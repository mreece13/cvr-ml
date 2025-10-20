# Analyzing Candidate Embeddings

This guide explains how to interpret the learned embeddings from your CVAE model to discover patterns like "progressive vs conservative candidates" or candidate clusters.

## What Gets Learned

The [`EmbeddingEncoder`](model.py ) learns three types of representations:

1. **Candidate Embeddings** (`encoder.embeddings[race_i]`): Raw embedding vectors for each candidate in each race
2. **Projection Weights** (`encoder.item_proj[race_i]`): Per-race transformations that adapt embeddings
3. **Decoder Weights** (`decoder.weights_list[race_i]`): How latent voter positions map to candidate probabilities

## Quick Start

### 1. Install Analysis Dependencies

```bash
pip install scikit-learn matplotlib seaborn scipy
```

### 2. Run Analysis Script

```bash
# Analyze all races
python analyze_embeddings.py --checkpoint lightning_logs/version_2/checkpoints/epoch=19.ckpt

# Analyze specific race (e.g., presidential)
python analyze_embeddings.py \
    --checkpoint lightning_logs/version_2/checkpoints/epoch=19.ckpt \
    --race "US PRESIDENT_FEDERAL" \
    --n-clusters 2

# Find similar candidates
python analyze_embeddings.py \
    --checkpoint lightning_logs/version_2/checkpoints/epoch=19.ckpt \
    --reference-candidate "JOSEPH R BIDEN" \
    --race "US PRESIDENT_FEDERAL"
```

## Outputs

The script generates:

### 1. **candidate_embeddings.csv**
All candidate embeddings in tabular form:
```
race_name,candidate_name,emb_0,emb_1,...,emb_15
US PRESIDENT_FEDERAL,JOSEPH R BIDEN,0.234,-0.567,...,0.891
US PRESIDENT_FEDERAL,DONALD J TRUMP,-0.456,0.789,...,-0.234
...
```

### 2. **Visualization Plots**
- `embeddings_pca_*.png`: 2D PCA projection showing candidate positions
- `embeddings_tsne_*.png`: 2D t-SNE showing nonlinear structure
- `similarity_heatmap_*.png`: Cosine similarity matrix between candidates
- `dendrogram_*.png`: Hierarchical clustering tree
- `clusters_k3_*.png`: K-means cluster assignments

### 3. **decoder_weights.csv**
How each latent dimension affects candidate probabilities:
```
race_name,candidate_name,latent_dim_0,latent_dim_1
US PRESIDENT_FEDERAL,JOSEPH R BIDEN,1.234,-0.456
US PRESIDENT_FEDERAL,DONALD J TRUMP,-1.567,0.678
```

## Interpretation Guide

### Finding Progressive/Conservative Clusters

**Method 1: Visual Inspection**
1. Look at `embeddings_pca_US_PRESIDENT_FEDERAL.png`
2. Candidates that cluster together have similar voting patterns
3. The axis directions may correspond to ideological dimensions

**Method 2: K-means Clustering**
```bash
python analyze_embeddings.py \
    --checkpoint model.ckpt \
    --race "US PRESIDENT_FEDERAL" \
    --n-clusters 2  # Try 2 for binary (left/right) or 3 for (left/center/right)
```

Output will show:
```
Cluster 0 (3 candidates):
  - JOSEPH R BIDEN (US PRESIDENT_FEDERAL)
  - HOWIE HAWKINS (US PRESIDENT_FEDERAL)
  - ...

Cluster 1 (2 candidates):
  - DONALD J TRUMP (US PRESIDENT_FEDERAL)
  - ...
```

**Method 3: Similarity Search**
Find candidates most similar to a known progressive:
```bash
python analyze_embeddings.py \
    --checkpoint model.ckpt \
    --reference-candidate "JOSEPH R BIDEN"
```

### Understanding Decoder Weights

Decoder weights show **how voter latent positions predict candidate choices**.

For a 2D latent space (e.g., social/economic axes):
```python
# decoder_weights.csv excerpt
candidate,latent_dim_0,latent_dim_1
BIDEN,     1.5,        0.8    # High on dim 0, positive on dim 1
TRUMP,    -1.2,        0.9    # Low on dim 0, positive on dim 1
SANDERS,   1.8,       -0.3    # High on dim 0, negative on dim 1
```

**Interpretation**:
- `latent_dim_0` might be **social liberalism** (Biden/Sanders high, Trump low)
- `latent_dim_1` might be **economic interventionism** (Biden/Trump positive, Sanders negative)

### Cross-Race Analysis

To find candidates with similar appeal across different races:

```python
# Load embeddings
import pandas as pd
embeddings = pd.read_csv('embedding_analysis/candidate_embeddings.csv')

# Group by candidate name (if same person runs in multiple races)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Get embeddings for all candidates
emb_cols = [c for c in embeddings.columns if c.startswith('emb_')]
X = embeddings[emb_cols].values

# Compute similarity matrix
sim = cosine_similarity(X)

# Find cross-race patterns
for i, row_i in embeddings.iterrows():
    for j, row_j in embeddings.iterrows():
        if i < j and row_i['race_name'] != row_j['race_name']:
            if sim[i, j] > 0.8:  # High similarity threshold
                print(f"{row_i['candidate_name']} ({row_i['race_name']}) similar to "
                      f"{row_j['candidate_name']} ({row_j['race_name']}): {sim[i,j]:.3f}")
```

## Manual Analysis in Python

If you want to do custom analysis:

```python
import torch
from model import CVAE, VoteDataProcessor
import numpy as np

# Load model
checkpoint = torch.load('model.ckpt')
processor = VoteDataProcessor.from_state_dict(checkpoint['data_processor'])
model = CVAE.load_from_checkpoint('model.ckpt', map_location='cpu', 
                                   dataloader=None, nitems=processor.nitems,
                                   n_classes_per_item=processor.get_n_classes_per_item())

# Get embeddings for a specific race
race_idx = 0  # First race
race_name = processor.idx_to_race[race_idx]
candidates = processor.get_all_candidates_for_race(race_idx)

# Extract embeddings (raw + projected)
raw_embeddings = model.encoder.embeddings[race_idx].weight.data.cpu().numpy()
proj_layer = model.encoder.item_proj[race_idx]

with torch.no_grad():
    projected_embeddings = torch.nn.functional.elu(
        proj_layer(torch.from_numpy(raw_embeddings))
    ).numpy()

# Analyze
print(f"Race: {race_name}")
for class_idx, cand_name in candidates.items():
    emb = projected_embeddings[class_idx]
    print(f"  {cand_name}: {emb[:3]}... (norm: {np.linalg.norm(emb):.3f})")

# Compute pairwise distances
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(projected_embeddings, metric='cosine'))

# Find most similar pair
min_dist = np.inf
for i in range(len(candidates)):
    for j in range(i+1, len(candidates)):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            most_similar = (candidates[i], candidates[j])

print(f"\nMost similar candidates: {most_similar[0]} & {most_similar[1]} "
      f"(distance: {min_dist:.3f})")
```

## Advanced: Latent Space Interpretation

The voter latent positions (`mu` from encoder) and decoder weights together reveal ideology:

```python
# Get voter latents from your exports
voter_latents = pd.read_csv('outputs/batch_size512_..._voter_latents.csv')

# Get decoder weights
decoder_weights = pd.read_csv('embedding_analysis/decoder_weights.csv')

# For US President race
pres_weights = decoder_weights[decoder_weights['race_name'] == 'US PRESIDENT_FEDERAL']

# Plot: latent dim 0 vs dim 1, colored by candidate preference
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
for _, row in pres_weights.iterrows():
    ax.scatter(row['latent_dim_0'], row['latent_dim_1'], s=200, alpha=0.7,
               label=row['candidate_name'])
    ax.annotate(row['candidate_name'], 
                (row['latent_dim_0'], row['latent_dim_1']),
                fontsize=10)

ax.set_xlabel('Latent Dimension 0')
ax.set_ylabel('Latent Dimension 1')
ax.set_title('Candidate Positions in Latent Space')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latent_space_interpretation.png', dpi=150)
```

## Tips

1. **Start with a single high-profile race** (e.g., presidential) to validate interpretability
2. **Try different cluster numbers** (k=2,3,4) to find natural groupings
3. **Compare to external labels** (e.g., party affiliation) to validate learned clusters
4. **Look at decoder weights** to understand what latent dimensions represent
5. **Check embedding norms**: Candidates with larger norms might be more "extreme" or distinctive

## Troubleshooting

**Q: Embeddings look random / no clear structure**
- Model may need more training epochs
- Try increasing `encoder_emb_dim` (default 16, try 32 or 64)
- Check if dataset is large enough (need substantial co-voting patterns)

**Q: All candidates cluster together**
- May need more latent dimensions
- Check if mask is too sparse (too much missing data)
- Try reducing regularization (increase `beta` parameter)

**Q: Clusters don't match expected ideology**
- Model learns from co-voting patterns, not external labels
- Clusters may reflect voting coalitions rather than ideology
- Consider: rural/urban, age demographics, not just left/right

## Next Steps

After identifying clusters, you can:
1. **Label clusters** based on known candidates
2. **Predict voter ideology** from their latent positions
3. **Analyze split-ticket voting** by comparing voter latents to decoder weights
4. **Study temporal changes** by training on different election cycles
