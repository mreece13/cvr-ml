# Quick Embedding Analysis Notebook
# Run this interactively to explore your model's learned candidate embeddings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from model import CVAE, VAEDataModule

# ============================================================================
# CONFIGURATION - Update these paths
# ============================================================================
CHECKPOINT_PATH = 'lightning_logs/lightning_logs/datacombined_sample.parquet_batch_size512_latent_dims4_hidden_size64_emb_dim16/checkpoints/last.ckpt'
DATA_PATH = 'data/combined_sample.parquet'  # Path to your data file
OUTPUT_DIR = "embedding_analysis"
REPRESENTATION = 'sparse'  # 'dense' or 'sparse' - sparse is recommended for large datasets

# ============================================================================
# Load Model and DataModule
# ============================================================================
print("Loading checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)

# Try to detect representation mode from checkpoint hyperparameters
hparams = checkpoint.get('hyper_parameters', {})
representation_mode = hparams.get('representation', REPRESENTATION)
print(f"Using representation mode: {representation_mode}")

print(f"Creating datamodule from file: {DATA_PATH}")
datamodule = VAEDataModule(
    filepath=DATA_PATH, 
    batch_size=512,
    representation=representation_mode
)
datamodule.prepare_data()
datamodule.setup()
print(f"✓ Datamodule loaded with {datamodule.nitems} races")
if hasattr(datamodule, 'dropped_races') and len(datamodule.dropped_races) > 0:
    print(f"  Note: {len(datamodule.dropped_races)} races with duplicates were dropped")

# Auto-detect model hyperparameters from checkpoint
latent_dims = hparams.get('latent_dims', 2)
hidden_layer_size = hparams.get('hidden_layer_size', 64)
encoder_emb_dim = hparams.get('encoder_emb_dim', 16)
learning_rate = hparams.get('learning_rate', 1e-3)
batch_size = hparams.get('batch_size', 512)

print(f"Detected hyperparameters:")
print(f"  latent_dims: {latent_dims}")
print(f"  hidden_layer_size: {hidden_layer_size}")
print(f"  encoder_emb_dim: {encoder_emb_dim}")

model = CVAE.load_from_checkpoint(
    CHECKPOINT_PATH, 
    map_location='cpu',
    nitems=datamodule.nitems,
    n_classes_per_item=datamodule.n_classes_per_item, 
    latent_dims=latent_dims, 
    hidden_layer_size=hidden_layer_size,
    encoder_emb_dim=encoder_emb_dim,
    qm=None, 
    learning_rate=learning_rate, 
    batch_size=batch_size
)
model.eval()
print("✓ Model loaded")

# ============================================================================
# Extract All Embeddings
# ============================================================================
def get_embeddings_df(model, datamodule):
    """Extract all candidate embeddings into a DataFrame"""
    rows = []
    for race_idx in range(datamodule.nitems):
        race_name = datamodule.idx_to_race[race_idx]
        candidates = datamodule.get_all_candidates_for_race(race_idx)
        
        # Get projected embeddings
        emb_matrix = model.encoder.embeddings[race_idx].weight.data
        proj_layer = model.encoder.item_proj[race_idx]
        
        with torch.no_grad():
            projected = torch.nn.functional.elu(proj_layer(emb_matrix)).cpu().numpy()
        
        for class_idx, candidate_name in candidates.items():
            row = {
                'race_idx': race_idx,
                'race_name': race_name,
                'class_idx': class_idx,
                'candidate_name': candidate_name,
                'embedding': projected[class_idx]
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

embeddings_df = get_embeddings_df(model, datamodule)
print(f"✓ Extracted embeddings for {len(embeddings_df)} candidates across {datamodule.nitems} races")

# ============================================================================
# Quick Summary
# ============================================================================
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)
print(f"Total races: {datamodule.nitems}")
print(f"Total candidates: {len(embeddings_df)}")
print(f"Embedding dimension: {embeddings_df.iloc[0]['embedding'].shape[0]}")
print("\nTop 10 races by candidate count:")
print(embeddings_df['race_name'].value_counts().head(10))

# ============================================================================
# Example 1: Find Similar Candidates
# ============================================================================
def find_similar(candidate_name, race_name=None, top_k=5):
    """Find candidates most similar to the given candidate"""
    from scipy.spatial.distance import cosine
    
    # Filter by race if specified
    df = embeddings_df.copy()
    if race_name:
        df = df[df['race_name'] == race_name]
    
    # Get target embedding
    target_row = embeddings_df[embeddings_df['candidate_name'] == candidate_name]
    if len(target_row) == 0:
        print(f"Candidate '{candidate_name}' not found")
        return None
    
    target_emb = target_row.iloc[0]['embedding']
    
    # Compute similarities
    similarities = []
    for idx, row in df.iterrows():
        sim = 1 - cosine(target_emb, row['embedding'])
        similarities.append({
            'candidate': row['candidate_name'],
            'race': row['race_name'],
            'similarity': sim
        })
    
    result = pd.DataFrame(similarities).sort_values('similarity', ascending=False)
    return result.head(top_k)

# Example usage:
print("\n" + "="*80)
print("EXAMPLE: Find candidates similar to 'joseph r biden'")
print("="*80)
similar = find_similar("joseph r biden", top_k=10)
if similar is not None:
    print(similar.to_string(index=False))

# ============================================================================
# Example 2: Visualize a Specific Race
# ============================================================================
def visualize_race(race_name, method='pca'):
    """Visualize candidate embeddings for a specific race"""
    df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
    
    if len(df) < 2:
        print(f"Not enough candidates in race: {race_name}")
        return
    
    # Stack embeddings
    emb_matrix = np.stack(df['embedding'].values)
    
    # Reduce to 2D
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(emb_matrix)
        var = reducer.explained_variance_ratio_
        title = f"{race_name}\nPCA (Var: {var[0]:.1%}, {var[1]:.1%})"
    else:
        from sklearn.manifold import TSNE
        perplexity = min(30, len(df) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = reducer.fit_transform(emb_matrix)
        title = f"{race_name}\nt-SNE"
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=150, alpha=0.6, c='steelblue', edgecolors='black', linewidth=1)
    
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        plt.annotate(
            row['candidate_name'][:30],
            (coords[idx, 0], coords[idx, 1]),
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
        )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} 1')
    plt.ylabel(f'{method.upper()} 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    filename = f"{OUTPUT_DIR}/quick_{method}_{race_name.replace('/', '_')[:50]}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()

# Example: Visualize presidential race
print("\n" + "="*80)
print("EXAMPLE: Visualize 'us president_federal' race")
print("="*80)
try:
    visualize_race("us president_federal", method='pca')
except Exception as e:
    print(f"Could not visualize: {e}")

# ============================================================================
# Example 3: Cluster Candidates
# ============================================================================
def cluster_race(race_name, n_clusters=3):
    """Cluster candidates in a race using K-means"""
    from sklearn.cluster import KMeans
    
    df = embeddings_df[embeddings_df['race_name'] == race_name].copy()
    emb_matrix = np.stack(df['embedding'].values)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(emb_matrix)
    
    print(f"\nClusters for {race_name} (k={n_clusters}):")
    print("-" * 80)
    for cluster_id in range(n_clusters):
        members = df[df['cluster'] == cluster_id]['candidate_name'].tolist()
        print(f"\nCluster {cluster_id} ({len(members)} candidates):")
        for member in members:
            print(f"  • {member}")
    
    return df

# Example usage:
print("\n" + "="*80)
print("EXAMPLE: Cluster 'us president_federal' candidates")
print("="*80)
try:
    clustered = cluster_race("us president_federal", n_clusters=4)
except Exception as e:
    print(f"Could not cluster: {e}")

# ============================================================================
# Example 4: Analyze Decoder Weights
# ============================================================================
def get_decoder_weights(model, datamodule, race_name):
    """Get decoder weight matrix for a race"""
    race_idx = datamodule.race_to_idx[race_name]
    candidates = datamodule.get_all_candidates_for_race(race_idx)
    
    # Weights shape: [latent_dims, n_candidates]
    weights = model.decoder.weights_list[race_idx].data.cpu().numpy()
    
    df_rows = []
    for class_idx, cand_name in candidates.items():
        row = {'candidate': cand_name}
        for dim in range(weights.shape[0]):
            row[f'dim_{dim}'] = weights[dim, class_idx]
        df_rows.append(row)
    
    return pd.DataFrame(df_rows)

print("\n" + "="*80)
print("EXAMPLE: Decoder weights for 'us president_federal'")
print("="*80)
try:
    decoder_df = get_decoder_weights(model, datamodule, "us president_federal")
    print(decoder_df.to_string(index=False))
    print("\nInterpretation:")
    print("  • Each dim_X column shows how latent dimension X affects this candidate")
    print("  • Positive = higher latent value → higher probability of voting for this candidate")
    print("  • Compare across candidates to understand ideological dimensions")
except Exception as e:
    print(f"Could not get decoder weights: {e}")

# ============================================================================
# Interactive Helpers
# ============================================================================
print("\n" + "="*80)
print("READY FOR INTERACTIVE EXPLORATION")
print("="*80)
print("\nAvailable functions:")
print("  • find_similar(candidate_name, race_name=None, top_k=5)")
print("  • visualize_race(race_name, method='pca')")
print("  • cluster_race(race_name, n_clusters=3)")
print("  • get_decoder_weights(model, datamodule, race_name)")
print("\nAvailable data:")
print("  • embeddings_df: DataFrame with all candidate embeddings")
print("  • model: The trained CVAE model")
print("  • datamodule: The VAEDataModule")
print("\nExample races to explore:")
for race in embeddings_df['race_name'].value_counts().head(5).index:
    print(f"  • {race}")

