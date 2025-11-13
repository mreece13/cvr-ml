"""
Explore voter ideal points (latent means) from a trained CVAE model.

Features:
- Compute ideal points (mu) for every voter in the dataset
- Plot density overlays for specified candidates within a given race (e.g., Biden vs Trump)
- Extra plots: 2D scatter/KDE by candidate, violin/box plots, pairplot across latent dims

Usage (examples):

python explore_ideal_points.py \
  --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
  --data data/colorado2.parquet \
  --race "US PRESIDENT_FEDERAL" \
  --candidates "JOSEPH R BIDEN,DONALD J TRUMP" \
  --output-dir ideal_points

# Only compute latents once then reuse the cached CSV
python explore_ideal_points.py \
  --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
  --data data/colorado2.parquet \
  --race "US PRESIDENT_FEDERAL" \
  --candidates "JOSEPH R BIDEN,DONALD J TRUMP" \
  --output-dir ideal_points \
  --latents-csv outputs/voter_latents.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from model import CVAE, VAEDataModule

sns.set_context("talk")

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_eval_loader(dm: VAEDataModule, batch_size: int) -> torch.utils.data.DataLoader:
    # Build a deterministic DataLoader aligned with the dm.index order
    dataset = torch.utils.data.TensorDataset(dm.data_tensor, dm.mask_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

def compute_voter_latents(model: CVAE,
                          dm: VAEDataModule,
                          batch_size: int = 1024,
                          save_csv: Optional[Path] = None,
                          subsample: Optional[int] = None) -> pd.DataFrame:
    """
    Compute per-voter latent means (ideal points) for all ballots.

    Returns a DataFrame with key columns plus columns mu_0..mu_{D-1}.
    If subsample is provided, returns a random subset of that many rows.
    """
    model.eval()
    model.to(device())

    loader = build_eval_loader(dm, batch_size)

    all_mu = []
    with torch.no_grad():
        for xb, mb in loader:
            xb = xb.to(device())
            mb = mb.to(device())
            mu, _ = model.encoder(xb.float(), mb.float())
            all_mu.append(mu.detach().cpu().numpy())

    mu_arr = np.concatenate(all_mu, axis=0)

    # Build DataFrame with keys
    idx_df = dm.index.to_pandas()  # polars -> pandas
    out = idx_df.copy()
    for d in range(mu_arr.shape[1]):
        out[f"mu_{d}"] = mu_arr[:, d]

    # Optional subsample
    if subsample is not None and subsample < len(out):
        out = out.sample(n=subsample, random_state=42).reset_index(drop=True)

    if save_csv is not None:
        out.to_csv(save_csv, index=False)
        print(f"Saved voter latents to: {save_csv}")

    return out


def normalize_mu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure latent columns are named mu_0, mu_1, ...
    If columns like z0, z1, z2, z3 exist (older exports that saved both mu and log_sigma),
    only keep the first half (the mu values) and rename them to mu_0, mu_1.
    """
    df = df.copy()
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    if mu_cols:
        # standard already; make sure numeric
        for c in mu_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    # try z0, z1, ... -> mu_0, mu_1
    # If we have z0, z1, z2, z3 for a 2D model, the first 2 are mu, the last 2 are log_sigma
    import re
    z_pat = re.compile(r"^z(\d+)$")
    z_cols = []
    for c in df.columns:
        m = z_pat.match(c)
        if m:
            z_cols.append((int(m.group(1)), c))
    
    if z_cols:
        z_cols.sort()  # sort by index
        # Assume first half are mu, second half are log_sigma
        n_latent = len(z_cols) // 2
        mu_z_cols = z_cols[:n_latent]
        
        renames: dict[str, str] = {}
        for i, (_, col_name) in enumerate(mu_z_cols):
            renames[col_name] = f"mu_{i}"
        
        df = df.rename(columns=renames)
        for c in renames.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")
        print(f"Detected {len(z_cols)} z* columns; kept first {n_latent} as mu: {renames}")
        
        # Drop the log_sigma columns
        sigma_cols = [col_name for _, col_name in z_cols[n_latent:]]
        if sigma_cols:
            df = df.drop(columns=sigma_cols)
            print(f"Dropped log_sigma columns: {sigma_cols}")
    
    return df

def extract_voter_choices(dm: VAEDataModule, race_name: str) -> pd.Series:
    """
    Return a pandas Series of candidate names per voter for the given race.
    Only ballots where mask==1 for the race are assigned; others are NaN.
    """
    if race_name not in dm.race_to_idx:
        raise ValueError(f"Race '{race_name}' not found. Available: e.g., {dm.idx_to_race[:5]} ...")

    ridx = dm.race_to_idx[race_name]

    # class index per voter
    class_idx = dm.data_tensor[:, ridx].numpy()
    mask = dm.mask_tensor[:, ridx].numpy().astype(bool)

    # idx -> name mapping
    class_to_name = dm.get_all_candidates_for_race(ridx)

    names = np.full(len(class_idx), None, dtype=object)
    for k, v in class_to_name.items():
        names[class_idx == k] = v

    names[~mask] = None
    return pd.Series(names, name="candidate")

def resolve_candidate_names(dm: VAEDataModule, race_name: str, requested: List[str]) -> List[str]:
    """
    Resolve candidate names to exact matches (case-insensitive). If a requested name
    doesn't match exactly, try substring matching; if still not found, ignore with a warning.
    """
    ridx = dm.race_to_idx[race_name]
    cmap = dm.get_all_candidates_for_race(ridx)
    available = list(cmap.values())

    resolved = []
    for q in requested:
        # exact (case-insensitive)
        matches = [c for c in available if c.lower() == q.lower()]
        if not matches:
            # substring
            matches = [c for c in available if q.lower() in c.lower()]
        if matches:
            resolved.append(matches[0])
        else:
            print(f"Warning: candidate '{q}' not found in race '{race_name}'. Skipping.")
    return resolved

def build_plot_frame(latents_df: pd.DataFrame, choices: pd.Series) -> pd.DataFrame:
    df = latents_df.copy()
    return df.join(choices.reset_index(drop=True))

def plot_density_by_candidate(df: pd.DataFrame,
                              race_name: str,
                              candidates: List[str],
                              out_dir: Path,
                              dims: Optional[List[int]] = None,
                              fill: bool = True) -> None:
    """
    For each requested latent dim, plot density overlays of mu_d for each candidate.
    """
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    dims = dims or list(range(len(mu_cols)))

    # Filter to only rows with candidate in requested set
    fdf = df[df["candidate"].isin(candidates)].copy()

    for d in dims:
        col = f"mu_{d}"
        plt.figure(figsize=(10, 6))
        for cand in candidates:
            sns.kdeplot(data=fdf[fdf["candidate"] == cand], x=col, fill=fill, label=cand, common_norm=False, alpha=0.4)
        plt.title(f"Density of latent dim {d} by candidate\n{race_name}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"density_dim{d}_{slugify(race_name)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

def plot_2d_kde(df: pd.DataFrame,
                race_name: str,
                candidates: List[str],
                out_dir: Path,
                dims: List[int] = [0, 1]) -> None:
    """2D KDE per candidate on selected dims (defaults to [0,1])."""
    colx = f"mu_{dims[0]}"
    coly = f"mu_{dims[1]}"

    fdf = df[df["candidate"].isin(candidates)].copy()

    # Individual panels for clarity
    g = sns.FacetGrid(fdf, col="candidate", hue="candidate", sharex=True, sharey=True, col_wrap=3, height=4)
    g.map_dataframe(sns.kdeplot, x=colx, y=coly, fill=True, thresh=0.05, levels=30, cmap="mako")
    g.map_dataframe(sns.scatterplot, x=colx, y=coly, s=5, alpha=0.2)
    g.set_titles(col_template="{col_name}")
    g.figure.suptitle(f"2D latent KDE by candidate\n{race_name}", y=1.05)
    for ax in g.axes.flatten():
        ax.set_xlabel(colx)
        ax.set_ylabel(coly)
    out_path = out_dir / f"kde2d_dim{dims[0]}_{dims[1]}_{slugify(race_name)}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_violin(df: pd.DataFrame,
                race_name: str,
                candidates: List[str],
                out_dir: Path,
                dims: Optional[List[int]] = None) -> None:
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    dims = dims or list(range(len(mu_cols)))

    fdf = df[df["candidate"].isin(candidates)].copy()

    for d in dims:
        col = f"mu_{d}"
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=fdf, x="candidate", y=col, inner="quartile", cut=0)
        plt.title(f"Violin plot of latent dim {d} by candidate\n{race_name}")
        plt.xlabel("Candidate")
        plt.ylabel(col)
        plt.xticks(rotation=20)
        plt.tight_layout()
        out_path = out_dir / f"violin_dim{d}_{slugify(race_name)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

def plot_pairplot(df: pd.DataFrame,
                  race_name: str,
                  candidates: List[str],
                  out_dir: Path,
                  max_dims: int = 3) -> None:
    # Keep only numeric mu_ columns and cap to max_dims
    mu_cols_all = [c for c in df.columns if c.startswith("mu_")]
    mu_cols = [c for c in mu_cols_all if pd.api.types.is_numeric_dtype(df[c])][:max_dims]

    if len(mu_cols) < 2:
        print(f"Skipping pairplot: need >= 2 numeric latent dims, found {len(mu_cols)} ({mu_cols}).")
        return

    fdf = df[df["candidate"].isin(candidates)][mu_cols + ["candidate"]]
    if fdf.empty:
        print("Skipping pairplot: no rows for requested candidates.")
        return

    # Pairplot can be heavy; cap rows for speed
    if len(fdf) > 50000:
        fdf = fdf.sample(50000, random_state=42)

    try:
        g = sns.pairplot(fdf, vars=mu_cols, hue="candidate", corner=True, plot_kws={"s": 8, "alpha": 0.3})
        g.fig.suptitle(f"Pairplot of latent dims (<= {max_dims}) by candidate\n{race_name}", y=1.02)
        out_path = out_dir / f"pairplot_{slugify(race_name)}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")
    except ValueError as e:
        print(f"Skipping pairplot due to error: {e}")

def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:80]

def main():
    parser = argparse.ArgumentParser(description="Explore voter ideal points from a trained CVAE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--candidates", type=str, default=None, help="Comma-separated candidate names to compare")
    parser.add_argument("--output-dir", type=str, default="ideal_points")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--subsample", type=int, default=None, help="Optional number of voters to sample for plotting")
    parser.add_argument("--latents-csv", type=str, default=None, help="Optional path to load/save computed latents CSV")
    parser.add_argument("--max-pairplot-dims", type=int, default=3)

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datamodule from data file
    dm = VAEDataModule(filepath=args.data, batch_size=args.batch_size)
    dm.prepare_data()
    dm.setup()

    # Inspect checkpoint hyperparameters and auto-detect model settings
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hparams = ckpt.get("hyper_parameters", {})
    if hparams:
        print("Loaded hparams from checkpoint:")
        for k in [
            "latent_dims",
            "hidden_layer_size",
            "encoder_emb_dim",
            "n_samples",
            "beta",
            "learning_rate",
            "batch_size",
            "nitems",
        ]:
            if k in hparams:
                print(f"  - {k}: {hparams[k]}")

        # Shape sanity check
        saved_nitems = hparams.get("nitems", None)
        if saved_nitems is not None and saved_nitems != dm.nitems:
            print(
                f"WARNING: checkpoint was trained with nitems={saved_nitems} "
                f"but current data has nitems={dm.nitems}. Proceeding with current data shapes."
            )

    # Load model with shapes from current datamodule, letting other hparams come from ckpt
    model = CVAE.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        nitems=dm.nitems,
        n_classes_per_item=dm.n_classes_per_item,
    )
    model.eval()
    
    # Report the actual latent dimension from the loaded model
    actual_latent_dim = model.latent_dims
    print(f"\nModel loaded with latent_dims = {actual_latent_dim}")
    if hparams.get("latent_dims") and hparams["latent_dims"] != actual_latent_dim:
        print(f"WARNING: checkpoint hparams say latent_dims={hparams['latent_dims']}, but model has {actual_latent_dim}")

    # Compute or load latents
    latents_csv_path = Path(args.latents_csv) if args.latents_csv else None
    if latents_csv_path and latents_csv_path.exists():
        latents_df = pd.read_csv(latents_csv_path)
        print(f"Loaded voter latents from: {latents_csv_path}")
        if args.subsample is not None and args.subsample < len(latents_df):
            latents_df = latents_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
    else:
        latents_df = compute_voter_latents(model, dm, batch_size=args.batch_size, save_csv=latents_csv_path, subsample=args.subsample)

    # Normalize latent column names (handles legacy z0,z1 -> mu_0,mu_1)
    latents_df = normalize_mu_columns(latents_df)
    
    # Report what latent columns we found after normalization
    mu_cols_found = [c for c in latents_df.columns if c.startswith("mu_")]
    print(f"Found {len(mu_cols_found)} latent dimension columns: {mu_cols_found}")

    # Choices for the race
    choices = extract_voter_choices(dm, args.race)

    # Merge and drop missing
    df = build_plot_frame(latents_df, choices)
    df = df[df["candidate"].notna()].reset_index(drop=True)

    # Resolve candidate names
    requested = [s.strip() for s in args.candidates.split(",")] if args.candidates else []
    if requested:
        candidates = resolve_candidate_names(dm, args.race, requested)
    else:
        # default: top 2 by count in the race
        top = df["candidate"].value_counts().head(2).index.tolist()
        print(f"No candidates provided; using top two by frequency: {top}")
        candidates = top

    # Basic stats
    print("\n=== Summary ===")
    print(f"Race: {args.race}")
    print("Candidates and counts:")
    print(df[df["candidate"].isin(candidates)]["candidate"].value_counts())

    # Plots
    plot_density_by_candidate(df, args.race, candidates, out_dir)
    if sum(c.startswith("mu_") for c in df.columns) >= 2:
        plot_2d_kde(df, args.race, candidates, out_dir, dims=[0, 1])
    plot_violin(df, args.race, candidates, out_dir)
    # Only attempt pairplot when we have at least 2 latent dims
    if sum(c.startswith("mu_") for c in df.columns) >= 2:
        plot_pairplot(df, args.race, candidates, out_dir, max_dims=args.max_pairplot_dims)
    else:
        print("Skipping pairplot: < 2 latent dimensions available.")

    # Save merged frame for reference
    merged_path = out_dir / f"ideal_points_{slugify(args.race)}.csv"
    df.to_csv(merged_path, index=False)
    print(f"Saved merged ideal points with labels: {merged_path}")


if __name__ == "__main__":
    main()
