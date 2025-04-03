import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import ListedColormap
import torch

def visualize_voter_latent_space(model, input_data, participation_mask_tensor, metadata, 
                                sample_df, pres_race_name, trump_name, biden_name,
                                output_file=None):
    """
    Create a scatter plot visualization of the voter latent space.
    
    Parameters:
    -----------
    model : VoterChoiceVAE
        Trained VAE model
    input_data : torch.Tensor
        One-hot encoded voter choices
    participation_mask_tensor : torch.Tensor
        Binary mask indicating voter participation
    metadata : dict
        Dictionary containing mappings between IDs and indices
    sample_df : pandas.DataFrame
        Original dataframe with voting data
    pres_race_name : str
        Name of the presidential race (e.g., "President_1")
    trump_name : str
        Name of Trump in the data
    biden_name : str
        Name of Biden in the data
    output_file : str, optional
        Path to save the visualization (if None, will display plot)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    model.eval()
    
    # Generate embeddings
    with torch.no_grad():
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
    
    # Get presidential votes
    pres_votes = {}
    pres_df = sample_df.with_columns(
        pl.concat_str([pl.col('office'), pl.lit('_'), 
                      pl.col('district').cast(pl.Utf8)]).alias('race')
    ).filter(pl.col("race") == pres_race_name).to_pandas()
    
    for _, row in pres_df.iterrows():
        pres_votes[row['cvr_id']] = row['candidate']
    
    # Add presidential vote to embeddings
    voter_embeddings_df['pres_vote'] = voter_embeddings_df['voter_id'].map(pres_votes)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create a custom colormap for non-presidential voters
    blue = np.array([0.12, 0.47, 0.71, 1.0])  # Democratic blue
    red = np.array([0.84, 0.15, 0.16, 1.0])   # Republican red
    
    # Create the scatter plot
    scatter = plt.scatter(
        voter_embeddings_df['trait_0'], 
        voter_embeddings_df['trait_1'],
        c=voter_embeddings_df['pres_vote'].map({trump_name: 1, biden_name: 0}),
        cmap=ListedColormap([blue, red]),
        alpha=0.7,
        s=30
    )
    
    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=blue, markersize=10, label=f'Biden Voters'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=red, markersize=10, label=f'Trump Voters')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Label the axes
    plt.xlabel('Latent Dimension 1 (Liberal-Conservative)', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    
    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add title
    plt.title('Voter Latent Space from VAE Model', fontsize=14)
    
    # Add text annotations for the axes meaning
    if voter_embeddings_df['trait_0'].mean() < 0:
        liberal_side, conservative_side = 'Left', 'Right'
    else:
        liberal_side, conservative_side = 'Right', 'Left'
        
    plt.annotate(f'More Liberal', xy=(0.05, 0.02), xycoords='figure fraction', fontsize=10)
    plt.annotate(f'More Conservative', xy=(0.85, 0.02), xycoords='figure fraction', fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    return plt.gcf()

def visualize_latent_space_with_contests(model, input_data, participation_mask_tensor, metadata, 
                                         sample_df, num_candidates_per_contest, output_file=None):
    """
    Create a visualization of the latent space with discrimination vectors for each contest.
    
    Parameters:
    -----------
    model : VoterChoiceVAE
        Trained VAE model
    input_data : torch.Tensor
        One-hot encoded voter choices
    participation_mask_tensor : torch.Tensor
        Binary mask indicating voter participation
    metadata : dict
        Dictionary containing mappings between IDs and indices
    sample_df : pandas.DataFrame
        Original dataframe with voting data
    num_candidates_per_contest : list
        Number of candidates in each contest
    output_file : str, optional
        Path to save the visualization (if None, will display plot)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    model.eval()
    
    # Generate embeddings
    with torch.no_grad():
        mu, _ = model.encode(input_data, participation_mask_tensor)
        voter_traits = mu.numpy()
    
    # Create DataFrame with embeddings
    voter_embeddings_df = pd.DataFrame(
        voter_traits, 
        columns=[f'trait_{i}' for i in range(voter_traits.shape[1])]
    )
    
    # Get discrimination parameters
    discrimination_params, _ = model.get_irt_parameters()
    
    # Start plotting
    plt.figure(figsize=(12, 10))
    
    # Plot voter points with less emphasis
    plt.scatter(
        voter_embeddings_df['trait_0'], 
        voter_embeddings_df['trait_1'],
        alpha=0.3,
        s=20,
        color='gray'
    )
    
    # Add contest vectors
    # We'll use the first two dimensions of the latent space
    for contest_idx, race_name in metadata['idx_to_race'].items():
        contest_params = discrimination_params[contest_idx].numpy()
        
        # Only use first two dimensions
        if contest_params.shape[1] >= 2:
            # Extract office and district from race name
            parts = race_name.split('_')
            office = parts[0]
            district = parts[1] if len(parts) > 1 else ""
            
            # Get candidate names
            candidate_map = metadata['candidate_maps'][contest_idx]
            
            # Plot vectors for each candidate
            for candidate_idx, candidate_name in sorted(
                [(v, k) for k, v in candidate_map.items()]
            ):
                # Get the 2D vector for this candidate
                vector = contest_params[candidate_idx, :2]
                
                # Skip if vector is too small
                if np.linalg.norm(vector) < 0.1:
                    continue
                
                # Plot the vector
                plt.arrow(
                    0, 0, vector[0], vector[1],
                    head_width=0.05, head_length=0.1,
                    alpha=0.7,
                    length_includes_head=True
                )
                
                # Add label
                plt.text(
                    vector[0] * 1.1, 
                    vector[1] * 1.1, 
                    f"{office}-{district}: {candidate_name}",
                    fontsize=8
                )
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Latent Dimension 1 (Liberal-Conservative)', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.title('Voter Latent Space with Contest Discrimination Vectors', fontsize=14)
    
    # Add origin point
    plt.scatter([0], [0], color='black', s=50)
    plt.text(0.05, 0.05, 'Origin', fontsize=10)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Improve layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    return plt.gcf()

# Example usage
def create_latent_space_visualizations(model, input_data, participation_mask_tensor, 
                                      metadata, sample_df, num_candidates_per_contest,
                                      pres_race_name, trump_name, biden_name):
    """
    Generate and save visualizations of the latent space.
    
    Parameters:
    -----------
    model : VoterChoiceVAE
        Trained VAE model
    input_data, participation_mask_tensor, metadata:
        Standard model inputs and metadata
    sample_df : pandas.DataFrame
        Original dataframe with voting data
    num_candidates_per_contest : list
        Number of candidates in each contest
    pres_race_name, trump_name, biden_name:
        Names for presidential candidates
    """
    print("Creating latent space visualizations...")
    
    # Basic scatter plot of voters colored by presidential vote
    visualize_voter_latent_space(
        model, input_data, participation_mask_tensor, metadata,
        sample_df, pres_race_name, trump_name, biden_name,
        output_file="voter_latent_space.png"
    )
    
    # Advanced plot with contest vectors
    visualize_latent_space_with_contests(
        model, input_data, participation_mask_tensor, metadata,
        sample_df, num_candidates_per_contest,
        output_file="voter_latent_space_with_contests.png"
    )