"""
Script to extract and analyze candidate embeddings from a trained CVAE model.

Usage:
    python analyze_embeddings.py --checkpoint path/to/checkpoint.ckpt --output-dir embedding_analysis
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

from model import CVAE, VAEDataModule


def extract_candidate_embeddings(model, datamodule):
    """
    Extract all candidate embeddings from the trained encoder.
    
    Returns:
        embeddings_df: DataFrame with columns [race_idx, race_name, class_idx, 
                       candidate_name, emb_0, emb_1, ..., emb_E]
    """
    rows = []
    encoder = model.encoder
    
    for race_idx in range(datamodule.nitems):
        race_name = datamodule.idx_to_race[race_idx]
        candidates = datamodule.get_all_candidates_for_race(race_idx)
        
        # Get embedding matrix for this race: shape [Ki, emb_dim]
        emb_matrix = encoder.embeddings[race_idx].weight.data.cpu().numpy()
        
        # Get projection weights (optional, but part of the learned representation)
        proj_layer = encoder.item_proj[race_idx]
        with torch.no_grad():
            # Apply the projection to get the "effective" embedding
            emb_tensor = torch.from_numpy(emb_matrix)
            projected = torch.nn.functional.elu(proj_layer(emb_tensor)).numpy()
        
        for class_idx, candidate_name in candidates.items():
            row = {
                'race_idx': race_idx,
                'race_name': race_name,
                'class_idx': class_idx,
                'candidate_name': candidate_name,
            }
            # Add embedding dimensions
            for dim_idx in range(projected.shape[1]):
                row[f'emb_{dim_idx}'] = projected[class_idx, dim_idx]
            rows.append(row)
    
    return pd.DataFrame(rows)


def compute_candidate_similarities(embeddings_df, race_name=None):
    """
    Compute pairwise cosine similarities between candidates.
    If race_name is provided, only compute within that race.
    """
    if race_name:
        df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
    else:
        df = embeddings_df.copy()
    
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    emb_matrix = df[emb_cols].values
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(emb_matrix)
    
    # Create labeled dataframe
    sim_df = pd.DataFrame(
        sim_matrix,
        index=df['candidate_name'].values,
        columns=df['candidate_name'].values
    )
    
    return sim_df


def find_similar_candidates(embeddings_df, candidate_name, race_name=None, top_k=10):
    """
    Find the top-k most similar candidates to a given candidate.
    """
    if race_name:
        df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
    else:
        df = embeddings_df.copy()
    
    # Get target candidate embedding
    target = df[df['candidate_name'] == candidate_name]
    if len(target) == 0:
        print(f"Candidate '{candidate_name}' not found")
        return None
    
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    target_emb = target[emb_cols].values[0]
    
    # Compute similarities to all candidates
    similarities = []
    for idx, row in df.iterrows():
        cand_emb = row[emb_cols].values
        sim = 1 - cosine(target_emb, cand_emb)  # cosine similarity
        similarities.append({
            'candidate': row['candidate_name'],
            'race': row['race_name'],
            'similarity': sim
        })
    
    sim_df = pd.DataFrame(similarities).sort_values('similarity', ascending=False)
    return sim_df.head(top_k)


def visualize_embeddings_2d(embeddings_df, output_dir, race_name=None, method='pca'):
    """
    Visualize embeddings in 2D using PCA or t-SNE.
    """
    if race_name:
        df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
        title_suffix = f" - {race_name}"
    else:
        df = embeddings_df.copy()
        title_suffix = " - All Races"
    
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    emb_matrix = df[emb_cols].values
    
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(emb_matrix)
        var_explained = reducer.explained_variance_ratio_
        title = f"PCA of Candidate Embeddings{title_suffix}\n(Var: {var_explained[0]:.2%}, {var_explained[1]:.2%})"
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df)-1))
        coords = reducer.fit_transform(emb_matrix)
        title = f"t-SNE of Candidate Embeddings{title_suffix}"
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    if race_name:
        # Single race: label all points
        plt.scatter(coords[:, 0], coords[:, 1], s=100, alpha=0.6)
        for i, row in df.iterrows():
            plt.annotate(
                row['candidate_name'][:30],  # truncate long names
                (coords[df.index.get_loc(i), 0], coords[df.index.get_loc(i), 1]),
                fontsize=8,
                alpha=0.7
            )
    else:
        # Multiple races: color by race
        unique_races = df['race_name'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_races)))
        race_to_color = dict(zip(unique_races, colors))
        
        for race in unique_races:
            mask = df['race_name'] == race
            race_coords = coords[mask]
            plt.scatter(
                race_coords[:, 0], 
                race_coords[:, 1], 
                label=race[:40], 
                s=100, 
                alpha=0.6,
                color=race_to_color[race]
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.title(title, fontsize=14)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    
    filename = f"embeddings_{method}{'_' + race_name.replace('/', '_') if race_name else '_all'}.png"
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def cluster_candidates(embeddings_df, race_name=None, n_clusters=3, output_dir=None):
    """
    Cluster candidates and identify clusters (e.g., progressive, moderate, conservative).
    """
    if race_name:
        df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
        title_suffix = f" - {race_name}"
    else:
        df = embeddings_df.copy()
        title_suffix = " - All Races"
    
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    emb_matrix = df[emb_cols].values
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(emb_matrix)
    
    # Print cluster assignments
    print(f"\n{'='*80}")
    print(f"CLUSTERING RESULTS{title_suffix}")
    print(f"{'='*80}")
    for cluster_id in range(n_clusters):
        cluster_members = df[df['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_members)} candidates):")
        for _, row in cluster_members.iterrows():
            print(f"  - {row['candidate_name']} ({row['race_name']})")
    
    # Visualize clusters with PCA
    if output_dir:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(emb_matrix)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            coords[:, 0], 
            coords[:, 1], 
            c=df['cluster'], 
            cmap='tab10', 
            s=100, 
            alpha=0.6
        )
        plt.colorbar(scatter, label='Cluster')
        plt.title(f"Candidate Clusters (K-means, k={n_clusters}){title_suffix}")
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        filename = f"clusters_k{n_clusters}{'_' + race_name.replace('/', '_') if race_name else '_all'}.png"
        plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()
    
    return df


def plot_similarity_heatmap(embeddings_df, race_name, output_dir):
    """
    Plot a heatmap of candidate similarities within a race.
    """
    sim_df = compute_candidate_similarities(embeddings_df, race_name=race_name)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title(f"Candidate Similarity Matrix - {race_name}")
    plt.tight_layout()
    
    filename = f"similarity_heatmap_{race_name.replace('/', '_')}.png"
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_dendrogram(embeddings_df, race_name, output_dir):
    """
    Plot hierarchical clustering dendrogram for candidates in a race.
    """
    df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    emb_matrix = df[emb_cols].values
    
    # Compute linkage
    linkage_matrix = linkage(emb_matrix, method='ward')
    
    plt.figure(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        labels=df['candidate_name'].values,
        leaf_rotation=90,
        leaf_font_size=10
    )
    plt.title(f"Hierarchical Clustering of Candidates - {race_name}")
    plt.xlabel("Candidate")
    plt.ylabel("Distance")
    plt.tight_layout()
    
    filename = f"dendrogram_{race_name.replace('/', '_')}.png"
    plt.savefig(Path(output_dir) / filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def analyze_decoder_weights(model, datamodule, output_dir):
    """
    Analyze decoder weight vectors to see how latent dimensions map to candidates.
    """
    decoder = model.decoder
    
    results = []
    for race_idx in range(datamodule.nitems):
        race_name = datamodule.idx_to_race[race_idx]
        candidates = datamodule.get_all_candidates_for_race(race_idx)
        
        # Decoder weights for this race: [latent_dims, Ki]
        weights = decoder.weights_list[race_idx].data.cpu().numpy()
        
        for class_idx, candidate_name in candidates.items():
            row = {
                'race_name': race_name,
                'candidate_name': candidate_name,
            }
            # Each latent dimension's weight for this candidate
            for dim in range(weights.shape[0]):
                row[f'latent_dim_{dim}'] = weights[dim, class_idx]
            results.append(row)
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = Path(output_dir) / "decoder_weights.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved decoder weights to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze candidate embeddings from trained CVAE')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data file (required to load datamodule)')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for datamodule (only used if creating new datamodule)')
    parser.add_argument('--output-dir', type=str, default='embedding_analysis', 
                       help='Directory to save analysis outputs')
    parser.add_argument('--race', type=str, default=None, 
                       help='Specific race to analyze (default: analyze all)')
    parser.add_argument('--reference-candidate', type=str, default=None,
                       help='Find similar candidates to this reference')
    parser.add_argument('--n-clusters', type=int, default=3,
                       help='Number of clusters for k-means')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Create datamodule - we need the data file to rebuild it
    if not args.data:
        raise ValueError("Please provide --data argument with path to data file.")
    
    print(f"Creating datamodule from file: {args.data}")
    datamodule = VAEDataModule(filepath=args.data, batch_size=args.batch_size)
    datamodule.prepare_data()
    datamodule.setup()
    print(f"Datamodule loaded with {datamodule.nitems} races")
    
    # Load model
    model = CVAE.load_from_checkpoint(
        args.checkpoint, 
        map_location='cpu',
        dataloader=None, 
        nitems=datamodule.nitems,
        n_classes_per_item=datamodule.n_classes_per_item, 
        latent_dims=2, 
        hidden_layer_size=64, 
        qm=None, 
        learning_rate=1e-3, 
        batch_size=args.batch_size
    )
    model.eval()
    print("Model loaded successfully")
    
    # Extract embeddings
    print("\nExtracting candidate embeddings...")
    embeddings_df = extract_candidate_embeddings(model, datamodule)
    
    # Save embeddings
    embeddings_path = output_dir / "candidate_embeddings.csv"
    embeddings_df.to_csv(embeddings_path, index=False)
    print(f"Saved embeddings to: {embeddings_path}")
    
    # Analyze decoder weights
    print("\nAnalyzing decoder weights...")
    decoder_df = analyze_decoder_weights(model, datamodule, output_dir)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    if args.race:
        # Analyze specific race
        print(f"\nAnalyzing race: {args.race}")
        visualize_embeddings_2d(embeddings_df, output_dir, race_name=args.race, method='pca')
        visualize_embeddings_2d(embeddings_df, output_dir, race_name=args.race, method='tsne')
        plot_similarity_heatmap(embeddings_df, args.race, output_dir)
        plot_dendrogram(embeddings_df, args.race, output_dir)
        cluster_candidates(embeddings_df, race_name=args.race, 
                          n_clusters=args.n_clusters, output_dir=output_dir)
    else:
        # Analyze all races
        visualize_embeddings_2d(embeddings_df, output_dir, method='pca')
        visualize_embeddings_2d(embeddings_df, output_dir, method='tsne')
        
        # Cluster all candidates
        cluster_candidates(embeddings_df, n_clusters=args.n_clusters, output_dir=output_dir)
        
        # Analyze top races
        top_races = embeddings_df['race_name'].value_counts().head(3).index
        for race in top_races:
            print(f"\nAnalyzing top race: {race}")
            plot_similarity_heatmap(embeddings_df, race, output_dir)
            plot_dendrogram(embeddings_df, race, output_dir)
    
    # Find similar candidates if reference provided
    if args.reference_candidate:
        print(f"\nFinding candidates similar to: {args.reference_candidate}")
        similar = find_similar_candidates(
            embeddings_df, 
            args.reference_candidate, 
            race_name=args.race,
            top_k=10
        )
        if similar is not None:
            print(similar.to_string(index=False))
            similar.to_csv(output_dir / f"similar_to_{args.reference_candidate.replace(' ', '_')}.csv", 
                          index=False)
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
