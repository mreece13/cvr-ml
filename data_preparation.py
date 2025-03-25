import pandas as pd
import numpy as np
import torch
from collections import defaultdict

def prepare_voter_data_from_dataframe(df, voter_col='cvr_id', office_col='office', district_col='district', candidate_col='candidate'):
    """
    Transform long-format voting data into the format required by the VAE model.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        Long-format dataframe with columns for voter ID, office, district, and candidate
    voter_col: str
        Column name for unique voter ID
    office_col: str
        Column name for office type
    district_col: str
        Column name for district
    candidate_col: str
        Column name for selected candidate
        
    Returns:
    --------
    raw_data: numpy.ndarray
        Matrix of voter choices, shape (num_voters, num_contests)
    participation_mask: numpy.ndarray
        Binary mask indicating which contests each voter participated in
    num_candidates_per_contest: list
        Number of candidates in each contest
    metadata: dict
        Dictionary containing mappings between IDs and indices
    """
    # Create race identifier by combining office and district
    df['race'] = df[office_col] + '_' + df[district_col].astype(str)
    race_col = 'race'
    
    # Get unique races, voters, and create mappings
    unique_races = sorted(df[race_col].unique())
    unique_voters = sorted(df[voter_col].unique())
    
    race_to_idx = {race: idx for idx, race in enumerate(unique_races)}
    voter_id_to_idx = {voter_id: idx for idx, voter_id in enumerate(unique_voters)}
    
    # Count candidates per race and create candidate mapping
    candidate_maps = []
    num_candidates_per_contest = []
    
    for race in unique_races:
        race_candidates = sorted(df[df[race_col] == race][candidate_col].unique())
        num_candidates = len(race_candidates)
        num_candidates_per_contest.append(num_candidates)
        
        # Map candidates to indices
        candidate_map = {candidate: idx for idx, candidate in enumerate(race_candidates)}
        candidate_maps.append(candidate_map)
    
    # Initialize matrices
    num_voters = len(unique_voters)
    num_contests = len(unique_races)
    raw_data = np.full((num_voters, num_contests), -1)
    participation_mask = np.zeros((num_voters, num_contests), dtype=bool)
    
    # Group by voter and race for more efficient processing
    voter_race_groups = df.groupby([voter_col, race_col])
    
    # Fill in the matrices
    for (voter_id, race), group in voter_race_groups:
        voter_idx = voter_id_to_idx[voter_id]
        race_idx = race_to_idx[race]
        
        # Use the first candidate if multiple (handling potential overvotes)
        candidate = group.iloc[0][candidate_col]
        candidate_idx = candidate_maps[race_idx][candidate]
        
        raw_data[voter_idx, race_idx] = candidate_idx
        participation_mask[voter_idx, race_idx] = True
    
    # Create metadata dictionary for future reference
    metadata = {
        'voter_id_to_idx': voter_id_to_idx,
        'idx_to_voter_id': {idx: voter_id for voter_id, idx in voter_id_to_idx.items()},
        'race_to_idx': race_to_idx,
        'idx_to_race': {idx: race for race, idx in race_to_idx.items()},
        'candidate_maps': candidate_maps
    }
    
    return raw_data, participation_mask, num_candidates_per_contest, metadata

def prepare_for_vae(raw_data, participation_mask, num_candidates_per_contest):
    """
    Convert raw data matrices into the tensors required by the VAE model.
    Uses the prepare_sparse_voter_data function from the VAE model.
    
    Parameters:
    -----------
    raw_data: numpy.ndarray
        Matrix of voter choices
    participation_mask: numpy.ndarray
        Binary mask indicating voter participation
    num_candidates_per_contest: list
        Number of candidates in each contest
        
    Returns:
    --------
    input_data: torch.Tensor
        One-hot encoded voter choices
    target_indices: list of tuples
        For each contest, indices of participating voters and their choices
    participation_mask_tensor: torch.Tensor
        Binary mask as tensor
    """
    num_voters, num_contests = raw_data.shape
    
    # Calculate total number of candidates across all contests
    total_candidates = sum(num_candidates_per_contest)
    
    # Initialize sparse matrix in COO format
    row_indices = []
    col_indices = []
    values = []
    
    # Calculate starting position for each contest
    contest_start_indices = [0]
    for i in range(num_contests - 1):
        contest_start_indices.append(contest_start_indices[-1] + num_candidates_per_contest[i])
    
    # Fill in the sparse matrix
    for voter_idx in range(num_voters):
        for contest_idx in range(num_contests):
            # Skip if voter didn't participate in this contest
            if not participation_mask[voter_idx, contest_idx]:
                continue
                
            # Get the chosen candidate
            candidate_idx = raw_data[voter_idx, contest_idx]
            
            # Skip if invalid
            if candidate_idx < 0 or np.isnan(candidate_idx):
                continue
                
            # Calculate position in the flattened representation
            col_idx = contest_start_indices[contest_idx] + int(candidate_idx)
            
            # Add to sparse representation
            row_indices.append(voter_idx)
            col_indices.append(col_idx)
            values.append(1.0)
    
    # Create sparse tensor and convert to dense
    i = torch.tensor([row_indices, col_indices], dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    input_data = torch.sparse.FloatTensor(i, v, (num_voters, total_candidates)).to_dense()
    
    # Create target indices for each contest
    target_indices = []
    for contest_idx in range(num_contests):
        # Get indices of voters who participated in this contest
        participant_indices = np.where(participation_mask[:, contest_idx])[0]
        
        # Get their choices
        choices = raw_data[participant_indices, contest_idx]
            
        # Convert to tensor
        target_indices.append((torch.tensor(participant_indices, dtype=torch.long),
                               torch.tensor(choices, dtype=torch.long)))
    
    # Convert participation mask to tensor
    participation_mask_tensor = torch.tensor(participation_mask, dtype=torch.float)
    
    return input_data, target_indices, participation_mask_tensor

def load_and_prepare_voter_data(df, voter_col='cvr_id', office_col='office', district_col='district', candidate_col='candidate'):
    """
    Complete pipeline to prepare voter data for the VAE model.
    
    Parameters:
    -----------
    df: pandas.DataFrame
    voter_col, office_col, district_col, candidate_col: str
        Column names for key fields
        
    Returns:
    --------
    prepared_data: tuple
        (input_data, target_indices, participation_mask_tensor, 
         num_candidates_per_contest, metadata)
    """
    
    # Prepare raw data
    raw_data, participation_mask, num_candidates_per_contest, metadata = prepare_voter_data_from_dataframe(df, voter_col, office_col, district_col, candidate_col)
    
    # Convert to format needed for VAE model
    input_data, target_indices, participation_mask_tensor = prepare_for_vae(raw_data, participation_mask, num_candidates_per_contest)
    
    # Print summary information
    print(f"Prepared data from {len(metadata['voter_id_to_idx'])} voters across {len(metadata['race_to_idx'])} contests")
    print(f"Number of candidates per contest: {num_candidates_per_contest}")
    print(f"Participation rate: {np.mean(participation_mask):.2f}")
    
    return input_data, target_indices, participation_mask_tensor, num_candidates_per_contest, metadata

def analyze_voter_embeddings(model, input_data, participation_mask_tensor, metadata):
    """
    Generate and analyze voter embeddings from the trained model.
    
    Parameters:
    -----------
    model: VoterChoiceVAE
        Trained VAE model
    input_data: torch.Tensor
        One-hot encoded voter choices
    participation_mask_tensor: torch.Tensor
        Binary mask indicating voter participation
    metadata: dict
        Dictionary containing mappings between IDs and indices
        
    Returns:
    --------
    voter_embeddings_df: pandas.DataFrame
        DataFrame with voter embeddings and voter IDs
    """
    model.eval()
    with torch.no_grad():
        # Get the latent space representation
        mu, _ = model.encode(input_data, participation_mask_tensor)
        voter_traits = mu.numpy()
    
    # Create DataFrame with embeddings
    voter_embeddings_df = pd.DataFrame(
        voter_traits, 
        columns=[f'trait_{i}' for i in range(voter_traits.shape[1])]
    )
    
    # Add voter IDs
    voter_embeddings_df['voter_id'] = [
        metadata['idx_to_voter_id'][idx] for idx in range(len(metadata['idx_to_voter_id']))
    ]
    
    return voter_embeddings_df