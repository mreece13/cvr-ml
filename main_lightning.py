import argparse
import lightning as L
import torch
from model import CVAE, VoteDataProcessor
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from lightning.pytorch.plugins.environments import SLURMEnvironment

# better numerial stability for matmul, and supported only on Engaging
# only set option if CUDA detected
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description='Train CVAE on ballot data')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--latent-dims', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--emb-dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--eval-only', type=bool, default=False)
    args = parser.parse_args()

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

    if not args.eval_only:
        if torch.cuda.is_available():
            trainer = L.Trainer(
                max_epochs=args.epochs, 
                accelerator='auto', 
                devices='auto', 
                plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)]
            )
        else:
            trainer = L.Trainer(
                max_epochs=args.epochs, 
                accelerator='auto', 
                devices='auto', 
            )

        trainer.fit(model)
    
    # print('Diagnostic 1:')
    # for i, b in enumerate(model.decoder.bias_list):
    #     print(i, 'shape', tuple(b.shape), 'requires_grad', b.requires_grad, 'sum', float(b.sum().item()), 'nonzero', int((b!=0).sum().item()))

    # print('Diagnostic 2:')  
    # # get one batch
    # dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    # x, m = next(iter(dl))
    # x = x.to(next(model.parameters()).device)
    # m = m.to(x.device)

    # opt = torch.optim.SGD(model.parameters(), lr=1e-3)   # local optimizer for test
    # opt.zero_grad()
    # reco, mu, sigma, z = model(x, m)
    # loss, _ = model.loss(x, reco, m, mu, sigma, z)
    # print('loss before backward', float(loss.item()))
    # loss.backward()

    # for i, b in enumerate(model.decoder.bias_list):
    #     g = b.grad
    #     if g is None:
    #         print(f'bias[{i}] grad: None')
    #     else:
    #         print(f'bias[{i}] grad norm={g.norm().item():.6g} min={g.min().item():.6g} max={g.max().item():.6g}')

    # print('Diagnostic 3:')
    # # quick listing of parameter names Lightning will see
    # names = [name for name, p in model.named_parameters() if p.requires_grad]
    # print('trainable param count:', len(names))
    # print([n for n in names if 'decoder.bias' in n or 'bias_list' in n][:50])  # show any bias-related names

    # print('Diagnostic 4:')
    # # add a tiny random offset to biases to break perfect symmetry and re-check grads
    # for b in model.decoder.bias_list:
    #     b.data += torch.randn_like(b) * 1e-2

    # opt.zero_grad()
    # reco2, mu2, sigma2, z2 = model(x, m)
    # loss2, _ = model.loss(x, reco2, m, mu2, sigma2, z2)
    # loss2.backward()

    # for i, b in enumerate(model.decoder.bias_list):
    #     print(i, 'bias sum', float(b.sum().item()), 'grad norm', None if b.grad is None else b.grad.norm().item())
    # opt.step()
    # # check biases updated
    # for i, b in enumerate(model.decoder.bias_list):
    #     print(i, 'bias after step sum', float(b.sum().item()))

    # After training (or when running in eval-only), run a small extraction/sanity check
    # use a fresh dataloader without shuffling for evaluation
    eval_dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --- small sanity extraction: per-item per-class weights/biases and nonzero counts ---
    # try:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model.to(device)
    #     model.eval()

    #     nitems = p.nitems
    #     Ks = p.get_n_classes_per_item()
    #     latent_dim = model.latent_dims

    #     total_params = 0
    #     total_nonzero = 0
    #     per_item = []
    #     for i in range(nitems):
    #         W = model.decoder.weights_list[i].detach().cpu()
    #         b = model.decoder.bias_list[i].detach().cpu()
    #         # W shape [D, Ki] -> per-class rows [Ki, D]
    #         Ki = W.shape[1]
    #         nonzero_w = (W != 0).sum().item()
    #         nonzero_b = (b != 0).sum().item()
    #         total_params += W.numel() + b.numel()
    #         total_nonzero += nonzero_w + nonzero_b
    #         per_item.append({
    #             'item_index': i,
    #             'Ki': Ki,
    #             'w_nonzero': int(nonzero_w),
    #             'b_nonzero': int(nonzero_b),
    #             'w_total': int(W.numel()),
    #             'b_total': int(b.numel())
    #         })

    #     print('Decoder params sanity check:')

    #     for info in per_item:
    #         print(f" item {info['item_index']}: Ki={info['Ki']} w_nonzero={info['w_nonzero']}/{info['w_total']} b_nonzero={info['b_nonzero']}/{info['b_total']}")

    #     print(f' total nonzero params: {total_nonzero}/{total_params}')
    # except Exception as e:
    #     print('Warning: unable to run decoder sanity check:', e)

    evaluate_and_export(model, p, eval_dl)

def evaluate_and_export(model: torch.nn.Module,
                        processor: VoteDataProcessor,
                        dl: DataLoader,
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

    # a_est = model.decoder.weights.t().detach().numpy()
    # d_est = model.decoder.bias.t().detach().numpy()

    # 1) collect posterior means (mu) for every row in the dataset (no sampling)
    mu_list = []
    ids_list = []
    with torch.no_grad():
        for batch in dl:
            # dataset expected to yield (labels_long, mask_float) or (labels_long, mask_float, ids)
            x = batch[0].to(device)
            m = batch[1].to(device)
            extras = batch[2:] if len(batch) > 2 else []
            # encoder returns (mu, log_sigma)
            mu, _ = model.encoder(x, m)
            mu_list.append(mu.cpu())
            if extras:
                ids_list.append(extras[0].cpu())

    mu_all = torch.cat(mu_list, dim=0).numpy()
    latent_dim = mu_all.shape[1]

    # save voter latents
    z_cols = [f"z{d}" for d in range(latent_dim)]
    df_latents = pd.DataFrame(mu_all, columns=z_cols)
    # attach ids if available
    if ids_list:
        try:
            ids_all = torch.cat(ids_list, dim=0).numpy()
            ids_df = pd.DataFrame(ids_all)
            df_latents = pd.concat([ids_df.reset_index(drop=True), df_latents.reset_index(drop=True)], axis=1)
        except Exception:
            pass
    df_latents.to_csv(os.path.join(out_dir, "voter_latents.csv"), index=False)
    print(f"Saved voter latents -> {os.path.join(out_dir, 'voter_latents.csv')}")

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
    out_items = os.path.join(out_dir, "item_parameters.csv")
    df_items.to_csv(out_items, index=False)
    print(f"Saved item parameters -> {out_items}")

if __name__ == '__main__':
    main()