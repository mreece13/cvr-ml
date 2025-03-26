import polars as pl
import numpy as np
import torch
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

def prepare_voter_data_from_dataframe(df, voter_col='cvr_id', office_col='office', 
                                     district_col='district', candidate_col='candidate'):
    """Optimized version using polars and parallel processing"""
    # Convert to polars if needed
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    
    # Create race identifier efficiently
    df = df.with_columns(
        pl.concat_str([pl.col(office_col), pl.lit('_'), 
                      pl.col(district_col).cast(pl.Utf8)]).alias('race')
    )
    
    # Get unique values efficiently using polars
    unique_races = df.select('race').unique().sort('race').to_series().to_list()
    unique_voters = df.select(voter_col).unique().sort(voter_col).to_series().to_list()
    
    race_to_idx = {race: idx for idx, race in enumerate(unique_races)}
    voter_id_to_idx = {voter_id: idx for idx, voter_id in enumerate(unique_voters)}
    
    # Process races in parallel
    candidate_maps = [None] * len(unique_races)
    num_candidates_per_contest = [0] * len(unique_races)
    
    def process_race(race_idx):
        race = unique_races[race_idx]
        race_candidates = df.filter(pl.col('race') == race).select(candidate_col).unique().sort(candidate_col).to_series().to_list()
        candidate_map = {candidate: idx for idx, candidate in enumerate(race_candidates)}
        return race_idx, len(race_candidates), candidate_map
    
    with ThreadPoolExecutor() as executor:
        for race_idx, num_candidates, candidate_map in executor.map(process_race, range(len(unique_races))):
            candidate_maps[race_idx] = candidate_map
            num_candidates_per_contest[race_idx] = num_candidates
    
    # Initialize matrices
    num_voters = len(unique_voters)
    num_contests = len(unique_races)
    raw_data = np.full((num_voters, num_contests), -1)
    participation_mask = np.zeros((num_voters, num_contests), dtype=bool)
    
    # Create more efficient version of filling the matrices
    # Use polars for faster groupby
    voter_race_data = (df.select([voter_col, 'race', candidate_col])
                       .group_by([voter_col, 'race'])
                       .agg(pl.first(candidate_col).alias(candidate_col))
                       )
    
    # Convert to numpy for fast processing
    voter_ids = voter_race_data[voter_col].to_numpy()
    races = voter_race_data['race'].to_numpy()
    candidates = voter_race_data[candidate_col].to_numpy()
    
    # Process in batches
    BATCH_SIZE = 100000
    for i in range(0, len(voter_ids), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(voter_ids))
        
        for j in range(i, batch_end):
            voter_id = voter_ids[j]
            race = races[j]
            candidate = candidates[j]
            
            voter_idx = voter_id_to_idx.get(voter_id)
            race_idx = race_to_idx.get(race)
            
            if voter_idx is None or race_idx is None:
                continue
                
            candidate_idx = candidate_maps[race_idx].get(candidate)
            if candidate_idx is None:
                continue
                
            raw_data[voter_idx, race_idx] = candidate_idx
            participation_mask[voter_idx, race_idx] = True
    
    # Create metadata
    metadata = {
        'voter_id_to_idx': voter_id_to_idx,
        'idx_to_voter_id': {idx: voter_id for voter_id, idx in voter_id_to_idx.items()},
        'race_to_idx': race_to_idx,
        'idx_to_race': {idx: race for race, idx in race_to_idx.items()},
        'candidate_maps': candidate_maps
    }
    
    return raw_data, participation_mask, num_candidates_per_contest, metadata

def prepare_for_vae(raw_data, participation_mask, num_candidates_per_contest, keep_sparse=True):
    """Optimized version with option to keep tensors sparse"""
    num_voters, num_contests = raw_data.shape
    total_candidates = sum(num_candidates_per_contest)
    
    # Calculate starting indices for each contest
    contest_start_indices = [0]
    for i in range(num_contests - 1):
        contest_start_indices.append(contest_start_indices[-1] + num_candidates_per_contest[i])
    
    # Create sparse tensor more efficiently
    row_indices = []
    col_indices = []
    values = []
    
    # Process each contest separately - more vectorized
    for contest_idx in range(num_contests):
        # Get all participants at once
        participant_indices = np.where(participation_mask[:, contest_idx])[0]
        if len(participant_indices) == 0:
            continue
            
        # Get choices for these participants
        choices = raw_data[participant_indices, contest_idx]
        
        # Filter valid choices
        valid_mask = (choices >= 0) & (~np.isnan(choices))
        valid_participants = participant_indices[valid_mask]
        valid_choices = choices[valid_mask].astype(np.int32)
        
        # Calculate global indices
        global_choice_indices = contest_start_indices[contest_idx] + valid_choices
        
        # Add to sparse representation
        row_indices.extend(valid_participants)
        col_indices.extend(global_choice_indices)
        values.extend([1.0] * len(valid_participants))
    
    # Create sparse tensor
    indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
    values = torch.tensor(values, dtype=torch.float)
    input_data = torch.sparse_coo_tensor(indices, values, (num_voters, total_candidates))
    
    # Only convert to dense if needed
    if not keep_sparse:
        input_data = input_data.to_dense()
    
    # Create target indices more efficiently
    target_indices = []
    for contest_idx in range(num_contests):
        participant_indices = np.where(participation_mask[:, contest_idx])[0]
        choices = raw_data[participant_indices, contest_idx]
        
        target_indices.append((
            torch.tensor(participant_indices, dtype=torch.long),
            torch.tensor(choices, dtype=torch.long)
        ))
    
    participation_mask_tensor = torch.tensor(participation_mask, dtype=torch.float)
    
    return input_data, target_indices, participation_mask_tensor

def load_and_prepare_voter_data(df, voter_col='cvr_id', office_col='office', 
                               district_col='district', candidate_col='candidate', 
                               keep_sparse=True):
    """Complete optimized pipeline"""
    # Convert to polars for better performance
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    
    # Prepare data
    raw_data, participation_mask, num_candidates_per_contest, metadata = (
        prepare_voter_data_from_dataframe(df, voter_col, office_col, district_col, candidate_col)
    )
    
    # Convert to VAE format
    input_data, target_indices, participation_mask_tensor = (
        prepare_for_vae(raw_data, participation_mask, num_candidates_per_contest, keep_sparse)
    )
    
    # Print summary
    print(f"Prepared data from {len(metadata['voter_id_to_idx'])} voters across {len(metadata['race_to_idx'])} contests")
    print(f"Participation rate: {np.mean(participation_mask):.2f}")
    
    return input_data, target_indices, participation_mask_tensor, num_candidates_per_contest, metadata

def analyze_voter_embeddings(model, input_data, participation_mask_tensor, metadata, batch_size=10000):
    """Optimized voter embedding analysis with batching"""
    model.eval()
    
    # Determine number of voters
    num_voters = input_data.size(0)
    
    # Process in batches to avoid OOM
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, num_voters, batch_size):
            end_idx = min(i + batch_size, num_voters)
            
            # Handle both sparse and dense inputs
            if input_data.is_sparse:
                batch_input = input_data.index_select(0, torch.arange(i, end_idx, dtype=torch.long))
            else:
                batch_input = input_data[i:end_idx]
                
            batch_mask = participation_mask_tensor[i:end_idx]
            
            # Get embeddings
            mu, _ = model.encode(batch_input, batch_mask)
            all_embeddings.append(mu)
    
    # Combine all batches
    voter_traits = torch.cat(all_embeddings, dim=0).numpy()
    
    # Create DataFrame with polars (faster than pandas)
    columns = {f'trait_{i}': voter_traits[:, i] for i in range(voter_traits.shape[1])}
    columns['voter_id'] = [metadata['idx_to_voter_id'][idx] for idx in range(len(metadata['idx_to_voter_id']))]
    
    return pl.DataFrame(columns)

def save_prepared_data(save_dir, data_tuple):
    """Save all data with a simple approach"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each tensor component separately
    torch.save(data_tuple[0], os.path.join(save_dir, 'input_data.pt'))  # input_data tensor
    
    # Save everything else as a single pickle file
    with open(os.path.join(save_dir, 'other_data.pkl'), 'wb') as f:
        pickle.dump(data_tuple[1:], f)
    
    print(f"Data saved to {save_dir}")

def load_prepared_data(save_dir):
    """Load the saved data"""
    # Load tensor component
    input_data = torch.load(os.path.join(save_dir, 'input_data.pt'))
    
    # Load everything else
    with open(os.path.join(save_dir, 'other_data.pkl'), 'rb') as f:
        other_data = pickle.load(f)
    
    # Reconstruct the full tuple
    return (input_data,) + other_data