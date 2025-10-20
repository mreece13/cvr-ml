# Summary: Interpreting Candidate Embeddings from Your CVAE

## What You Have

Your CVAE model with [`EmbeddingEncoder`](model.py ) learns **candidate embeddings** - vector representations that capture voting patterns. Candidates with similar embeddings tend to be chosen by similar voters.

## Tools I've Created

### 1. **analyze_embeddings.py** (Comprehensive Analysis)
Full-featured script that generates visualizations and statistics.

**Installation:**
```bash
pip install scikit-learn matplotlib seaborn scipy
```

**Basic usage:**
```bash
# Analyze all races
python analyze_embeddings.py --checkpoint path/to/model.ckpt

# Analyze specific race with clustering
python analyze_embeddings.py \
    --checkpoint path/to/model.ckpt \
    --race "US PRESIDENT_FEDERAL" \
    --n-clusters 2

# Find similar candidates
python analyze_embeddings.py \
    --checkpoint path/to/model.ckpt \
    --reference-candidate "JOSEPH R BIDEN"
```

**Outputs:**
- `candidate_embeddings.csv` - All embeddings in table form
- `embeddings_pca_*.png` - 2D PCA visualization
- `embeddings_tsne_*.png` - 2D t-SNE visualization
- `similarity_heatmap_*.png` - Candidate similarity matrix
- `dendrogram_*.png` - Hierarchical clustering tree
- `clusters_k*.png` - K-means cluster visualization
- `decoder_weights.csv` - Latent→candidate mappings

### 2. **explore_embeddings.py** (Quick Interactive Script)
Lightweight Python script for rapid exploration.

**Usage:**
```bash
# Run interactively (in IPython or Jupyter)
python -i explore_embeddings.py

# Then use functions:
>>> find_similar("JOSEPH R BIDEN", top_k=10)
>>> visualize_race("US PRESIDENT_FEDERAL")
>>> cluster_race("US PRESIDENT_FEDERAL", n_clusters=3)
```

### 3. **Updated VoteDataProcessor Methods**
Added helper methods to [`model.py`](model.py ):
- `get_candidate_name(race_idx, class_idx)` - Look up candidate by indices
- `get_all_candidates_for_race(race_idx)` - Get all candidates in a race

## How to Answer: "Which Candidates Are Progressive?"

### Method 1: Visual Clustering (Easiest)
```bash
python analyze_embeddings.py \
    --checkpoint model.ckpt \
    --race "US PRESIDENT_FEDERAL" \
    --n-clusters 2
```

Look at the output plot - candidates that cluster together share voting patterns. Compare to known labels (e.g., party affiliation) to identify which cluster is "progressive."

### Method 2: Similarity Search
Start with a known progressive candidate:
```bash
python analyze_embeddings.py \
    --checkpoint model.ckpt \
    --reference-candidate "JOSEPH R BIDEN"
```

Top similar candidates are likely also progressive.

### Method 3: Decoder Weight Analysis
```python
import pandas as pd

# Load decoder weights
weights = pd.read_csv('embedding_analysis/decoder_weights.csv')
pres = weights[weights['race_name'] == 'US PRESIDENT_FEDERAL']

# Candidates with similar latent dimension patterns are ideologically similar
# E.g., if Biden has high latent_dim_0 and low latent_dim_1,
# other candidates with that pattern are likely progressive
print(pres[['candidate_name', 'latent_dim_0', 'latent_dim_1']])
```

### Method 4: Compute Pairwise Similarities
```python
from scipy.spatial.distance import cosine
import pandas as pd

embeddings = pd.read_csv('embedding_analysis/candidate_embeddings.csv')
race_df = embeddings[embeddings['race_name'] == 'US PRESIDENT_FEDERAL']

# Get embedding columns
emb_cols = [c for c in race_df.columns if c.startswith('emb_')]

# Compute all pairwise similarities
for i, row_i in race_df.iterrows():
    for j, row_j in race_df.iterrows():
        if i < j:
            emb_i = row_i[emb_cols].values
            emb_j = row_j[emb_cols].values
            similarity = 1 - cosine(emb_i, emb_j)
            print(f"{row_i['candidate_name'][:20]:20} <-> {row_j['candidate_name'][:20]:20}: {similarity:.3f}")
```

## Interpretation Tips

### What Embeddings Capture
- **Co-voting patterns**: Candidates chosen together by the same voters have similar embeddings
- **Voter coalitions**: Not necessarily ideology, but actual voting behavior
- **Cross-race patterns**: A candidate's embedding in one race reflects their position relative to that race's field

### What Decoder Weights Show
- **Latent dimension → candidate mapping**: How voter positions predict candidate choice
- **Ideological axes**: Each latent dimension may represent social/economic/geographic dimensions
- **Example**: If latent_dim_0 separates Biden (+) from Trump (-), it likely represents liberal-conservative

### Validation Strategies
1. **Compare to party labels**: Do clusters align with Democrat/Republican?
2. **Check known pairs**: Are Biden-Harris similar? Trump-Pence?
3. **Geographic patterns**: Do regional candidates cluster?
4. **Historical consistency**: Do 2020 candidates cluster like 2016?

## Example Workflow

```bash
# 1. Train your model (you've done this)
python main_lightning.py --data data/colorado.parquet --epochs 20 --latent-dims 2

# 2. Run comprehensive analysis
python analyze_embeddings.py \
    --checkpoint lightning_logs/version_X/checkpoints/epoch=19.ckpt \
    --output-dir analysis_results

# 3. Explore specific races
python analyze_embeddings.py \
    --checkpoint lightning_logs/version_X/checkpoints/epoch=19.ckpt \
    --race "US PRESIDENT_FEDERAL" \
    --n-clusters 2

# 4. Find similar candidates
python analyze_embeddings.py \
    --checkpoint lightning_logs/version_X/checkpoints/epoch=19.ckpt \
    --reference-candidate "JOSEPH R BIDEN"

# 5. Interactive exploration
python -i explore_embeddings.py
>>> visualize_race("US SENATE_FEDERAL")
>>> cluster_race("US SENATE_FEDERAL", n_clusters=3)
```

## Expected Patterns

If your model is working well, you should see:

✅ **Presidential race**: Clear 2-cluster structure (Biden voters vs Trump voters)  
✅ **Down-ballot races**: Similar clustering patterns (Democrats vs Republicans)  
✅ **Third-party candidates**: May form separate small clusters  
✅ **Cross-race consistency**: Progressive senate candidates cluster with progressive presidential candidates  
✅ **Geographic patterns**: Local candidates may cluster by region  

## Troubleshooting

**Problem**: All embeddings look random, no clear structure
- **Solution**: Train longer, increase `emb_dim`, check data quality

**Problem**: Only 1 cluster emerges
- **Solution**: Increase latent dimensions, reduce regularization, check for data imbalance

**Problem**: Clusters don't match party labels
- **Solution**: Model learns voting patterns, not party labels - this might be interesting! Maybe some Democrats vote like Republicans in your data.

## Next Steps

1. **Identify clusters**: Label them as "progressive", "moderate", "conservative" based on known candidates
2. **Analyze voter latents**: Use `voter_latents.csv` to see where voters fall ideologically
3. **Study split-ticket voting**: Find voters whose latent position is between clusters
4. **Temporal analysis**: Train on different elections and compare embedding shifts
5. **Prediction**: Use embeddings to predict candidate performance in new races

## Files You Need

- ✅ `model.py` - Updated with helper methods
- ✅ `analyze_embeddings.py` - Comprehensive analysis script
- ✅ `explore_embeddings.py` - Quick interactive exploration
- ✅ `EMBEDDING_ANALYSIS.md` - Detailed documentation
- ✅ Your trained checkpoint with data processor state

All set! Start with `python analyze_embeddings.py --checkpoint YOUR_CHECKPOINT.ckpt` and explore the visualizations.
