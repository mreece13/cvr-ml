import argparse
import lightning as L
import torch
from model import CVAE, VoteDataProcessor
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from lightning.pytorch.plugins.environments import SLURMEnvironment
import signal

def main():
    parser = argparse.ArgumentParser(description='Train CVAE on ballot data')
    parser.add_argument('--data', type=str, default='data/colorado.parquet')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dims', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--eval-only', type=bool, default=False)
    args = parser.parse_args()

    # Build a file name using all arguments except eval-only
    args_dict = {k: v for k, v in vars(args).items() if k != 'eval_only' and k != 'data'}
    file_name_parts = []
    for k, v in args_dict.items():
        file_name_parts.append(f"{k}{v}")
    file_name = "_".join(file_name_parts)

    print(f"Generated file name: {file_name}")

    DATA_PATH = args.data
    if not DATA_PATH:
        raise RuntimeError('Set DATA_PATH (or pass --data) to your ballots file before running training')

    # 1) read ballots and build dataset
    p = VoteDataProcessor(filepath=DATA_PATH)
    Ks = p.get_n_classes_per_item()

    # 2) DataLoader
    ds = p.get_torch_dataset()
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=8, 
        persistent_workers=True
    )

    # 3) instantiate CVAE
    model = CVAE(
        dataloader=dl,
        nitems=p.nitems,
        latent_dims=args.latent_dims,
        hidden_layer_size=args.hidden_size,
        qm=None,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_classes_per_item=Ks,
        encoder_emb_dim=args.emb_dim,
        n_samples=args.n_samples,
    )

    # better numerial stability for matmul, and supported only on Engaging
    # only set option if CUDA detected
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    if not args.eval_only:
        if torch.cuda.is_available():
            trainer = L.Trainer(
                max_epochs=args.epochs, 
                accelerator='auto', 
                devices='auto', 
                gpus = -1,
                # precision="transformer-engine",
                plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
            )
        else:
            trainer = L.Trainer(
                max_epochs=args.epochs, 
                accelerator='auto', 
                devices='auto', 
            )

        trainer.fit(model)
    
    eval_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    evaluate_and_export(model, p, eval_dl, file_name)

def evaluate_and_export(model: torch.nn.Module,
                        processor: VoteDataProcessor,
                        dl: DataLoader,
                        file_name: str,
                        out_dir: str = "outputs",
                        ref_name_special: str = "US PRESIDENT_FEDERAL",
                        ref_candidate_special: str = "JOSEPH R BIDEN"):
    """
    Evaluate trained model and export:
      - voter_latents.csv : posterior means (mu) per ballot / row
      - item_parameters.csv: per-item per-class 2PL-like parameters using a chosen reference class

    Reference selection:
      - For race named `ref_name_special` use candidate named `ref_candidate_special`
      - Otherwise choose the first alphabetical candidate name as reference
    """
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # save index
    processor.index.to_frame(index=False).to_csv(os.path.join(out_dir, file_name + "_index.csv"), index=False)

    # 1) collect posterior means (mu) for every row in the dataset (no sampling)
    mu_list = []
    with torch.no_grad():
        for batch in dl:
            # dataset expected to yield (labels_long, mask_float) or (labels_long, mask_float, ids)
            x = batch[0].to(device)
            m = batch[1].to(device)
            # encoder returns (mu, log_sigma)
            mu, _ = model.encoder(x, m)
            mu_list.append(mu.cpu())

    mu_all = torch.cat(mu_list, dim=0).numpy()
    latent_dim = mu_all.shape[1]

    # save voter latents
    z_cols = [f"z{d}" for d in range(latent_dim)]
    df_latents = pd.DataFrame(mu_all, columns=z_cols)
    df_latents = pd.concat([processor.index.to_frame(index=False), df_latents.reset_index(drop=True)], axis=1)
    df_latents.to_csv(os.path.join(out_dir, file_name + "_voter_latents.csv"), index=False)
    print(f"Saved voter latents -> {os.path.join(out_dir, file_name + '_voter_latents.csv')}")

    # --- extract per-item per-class decoder parameters (match Decoder in model.py) ---
    Ks = processor.get_n_classes_per_item()
    nitems = processor.nitems
    latent_dim = mu_all.shape[1]

    # build readable candidate names per item from processor.candidate_maps if available
    candidate_names_per_item = []
    item_names = getattr(processor, "idx_to_race", None)
    for r in (item_names if item_names is not None else list(range(nitems))):
        cmap = processor.candidate_maps[r] if hasattr(processor, "candidate_maps") else {}
        # invert cmap into ordered list by index
        if cmap:
            inv = [None] * len(cmap)
            for name, idx in cmap.items():
                if 0 <= idx < len(inv):
                    inv[idx] = name
            # fallback for any None entries
            for j in range(len(inv)):
                if inv[j] is None:
                    inv[j] = f"class_{j}"
            candidate_names_per_item.append(inv)
        else:
            # fallback generic names using Ks if available
            Ki = Ks[len(candidate_names_per_item)] if Ks is not None and len(Ks) > len(candidate_names_per_item) else 1
            candidate_names_per_item.append([f"class_{k}" for k in range(Ki)])

    rows = []
    # Decoder stores per-item weights as Parameters of shape [latent_dims, Ki]
    for item_idx in range(len(model.decoder.weights_list)):
        W_param = model.decoder.weights_list[item_idx].detach().cpu().numpy()  # [D, Ki]
        b_param = model.decoder.bias_list[item_idx].detach().cpu().numpy()     # [Ki]
        # transpose to [Ki, D] so each row corresponds to a class weight vector
        W_per_class = W_param.T  # shape [Ki, latent_dim]
        Ki = W_per_class.shape[0]

        cands = candidate_names_per_item[item_idx] if len(candidate_names_per_item) > item_idx else [f"class_{k}" for k in range(Ki)]
        name_to_idx = {name: k for k, name in enumerate(cands)}
        item_label = item_names[item_idx] if item_names and len(item_names) > item_idx else None

        # decide reference class index
        if item_label == ref_name_special:
            ref_idx = name_to_idx.get(ref_candidate_special, 0)
        else:
            sorted_names = sorted(cands)
            ref_name = sorted_names[0]
            ref_idx = name_to_idx.get(ref_name, 0)

        # extract rows
        for k in range(Ki):
            w_k = W_per_class[k]            # shape [latent_dim]
            b_k = b_param[k]
            w_ref = W_per_class[ref_idx]
            b_ref = b_param[ref_idx]

            delta_w = w_k - w_ref
            delta_b = b_k - b_ref
            discr = float(np.linalg.norm(delta_w))
            difficulty = float(-delta_b / discr) if discr > 0 else float("nan")

            row = {
                "item_index": int(item_idx),
                "item_name": item_label if item_label is not None else "",
                "class_index": int(k),
                "class_name": cands[k],
                "ref_index": int(ref_idx),
                "bias": float(b_k),
                "discrimination": discr,
                "difficulty": difficulty,
            }
            for d in range(latent_dim):
                row[f"w_{d}"] = float(w_k[d])
            rows.append(row)

    df_items = pd.DataFrame(rows)
    out_items = os.path.join(out_dir, file_name + "_item_parameters.csv")
    df_items.to_csv(out_items, index=False)
    print(f"Saved item parameters -> {out_items}")

    # ------------------ extra exports: per-voter projections & pres. probs ------------------
    # find presidential item index (best effort by name match)
    pres_idx = None
    if item_names is not None:
        for i, nm in enumerate(item_names):
            if nm and 'PRESIDENT' in nm.upper():
                pres_idx = i
                break

    # only proceed if we found a presidential item
    if pres_idx is not None:
        # candidate name list for pres item
        pres_cands = candidate_names_per_item[pres_idx]
        # try to pick Biden and Trump indices
        biden_name = 'JOSEPH R BIDEN'
        trump_name = 'DONALD J TRUMP'
        biden_idx = None
        trump_idx = None
        for k, name in enumerate(pres_cands):
            if name.upper() == biden_name:
                biden_idx = k
            if name.upper() == trump_name:
                trump_idx = k
        
        if biden_idx is None or trump_idx is None:
            raise RuntimeError(f"Could not find both Biden and Trump in presidential candidate names: {pres_cands}")

        # build decoder W and b for pres item
        W_param = model.decoder.weights_list[pres_idx].detach().cpu().numpy()  # [D, Ki]
        b_param = model.decoder.bias_list[pres_idx].detach().cpu().numpy()     # [Ki]
        W_per_class = W_param.T  # [Ki, D]

        # compute Biden-Trump separating axis (delta w)
        if biden_idx is not None and trump_idx is not None:
            delta = W_per_class[biden_idx] - W_per_class[trump_idx]
            delta_norm = np.linalg.norm(delta)
            if delta_norm > 0:
                delta_unit = (delta / delta_norm)
            else:
                delta_unit = delta * 0.0
        else:
            delta_unit = None

        # compute per-voter probabilities and projections
        # mu_all is [N, D]
        # numerically-stable numpy softmax (avoid extra SciPy dependency)
        def _np_softmax(a, axis=1):
            a = a - np.max(a, axis=axis, keepdims=True)
            e = np.exp(a)
            return e / np.sum(e, axis=axis, keepdims=True)

        logits = mu_all.dot(W_param) + b_param  # mu_all [N,D] dot W_param [D,Ki] -> [N,Ki]
        probs = _np_softmax(logits, axis=1)

        # projection onto Biden-Trump axis
        if delta_unit is not None:
            proj_bt = mu_all.dot(delta_unit)
        else:
            proj_bt = np.full((mu_all.shape[0],), np.nan)

        # per-dimension projections (ideal points per latent dim)
        per_dim = mu_all.copy()  # each column is an 'ideal point' on that dimension

        # assemble voter dataframe with ids (if available)
        df_voters = pd.DataFrame(per_dim, columns=[f'z{d}' for d in range(per_dim.shape[1])])
        # attach ids if they exist in the earlier saved df_latents
        try:
            ids_cols = [c for c in df_latents.columns if c not in df_voters.columns]
            if ids_cols:
                df_voters = pd.concat([df_latents[ids_cols].reset_index(drop=True), df_voters.reset_index(drop=True)], axis=1)
        except Exception:
            pass

        # add projection and pres probs
        df_voters['proj_biden_minus_trump'] = proj_bt
        # ensure indices exist in pres_cands
        if biden_idx is not None:
            df_voters['p_biden'] = probs[:, biden_idx]
        if trump_idx is not None:
            df_voters['p_trump'] = probs[:, trump_idx]

        # also add predicted winner and max prob
        df_voters['pred_class'] = probs.argmax(axis=1)
        df_voters['pred_prob'] = probs.max(axis=1)

        out_voters = os.path.join(out_dir, file_name + '_voter_scores.csv')
        df_voters.to_csv(out_voters, index=False)
        print(f"Saved voter projections & pres. probs -> {out_voters}")

        # print which latent dimension individually best correlates with BT axis
        if delta_unit is not None:
            corrs = []
            for d in range(per_dim.shape[1]):
                c = np.corrcoef(per_dim[:, d], proj_bt)[0,1]
                corrs.append(abs(c))
            best_dim = int(np.nanargmax(corrs))
            print(f'Best single latent dim separating Biden/Trump: dim {best_dim} (abs corr={corrs[best_dim]:.3f})')
        else:
            print('Could not compute Biden-Trump separating axis (missing indices)')

if __name__ == '__main__':
    main()