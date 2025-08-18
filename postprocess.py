import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize_relative_discrimination_vectors(adjusted_params, reference_info, metadata, 
                                             voter_embeddings=None, output_file=None, contest_filter=None):
    """
    Visualize the adjusted discrimination vectors relative to reference candidates.
    
    Parameters:
    -----------
    adjusted_params : list
        List of adjusted discrimination parameter tensors
    reference_info : dict
        Information about reference candidates
    metadata : dict
        Dictionary containing mappings between races/candidates and indices
    voter_embeddings : pandas.DataFrame, optional
        DataFrame containing voter embeddings to plot in background
    output_file : str, optional
        Path to save the visualization
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Define color map for different contests
    contest_colors = plt.cm.tab10.colors
    
    # Plot contest vectors
    for contest_idx, contest_params in enumerate(adjusted_params):
        # Get contest info
        race_name = reference_info[contest_idx]['race']
        ref_name = reference_info[contest_idx]['name']

        if contest_filter is not None and contest_filter not in race_name:
            continue
        
        # Extract office and district
        parts = race_name.split('_')
        office = parts[0]
        district = parts[1] if len(parts) > 1 else ""
        
        # Get candidate names
        candidate_map = metadata['candidate_maps'][contest_idx]
        
        # Plot vectors for each non-reference candidate
        for candidate_idx, candidate_name in sorted([(v, k) for k, v in candidate_map.items()]):
            color = contest_colors[contest_idx % len(contest_colors)]
            
            # Skip reference candidate (should be at origin)
            if candidate_idx == reference_info[contest_idx]['index']:
                plt.text(
                    0, 0, 
                    f"{office}-{district}: {candidate_name}",
                    fontsize=8,
                    color=color
                )
                continue
                
            # Get vector (first two dimensions)
            if contest_params.size(1) >= 2:
                vector = contest_params[candidate_idx, :2].cpu().numpy()
                
                # Skip if vector is too small
                if np.linalg.norm(vector) < 0.05:
                    continue
                
                # Plot vector
                plt.arrow(
                    0, 0, vector[0], vector[1],
                    head_width=0.02, head_length=0.05,
                    alpha=0.8,
                    color=color,
                    length_includes_head=True
                )
                
                # Add label
                plt.text(
                    vector[0] * 1.05,
                    vector[1] * 1.05,
                    f"{office}-{district}: {candidate_name}",
                    fontsize=8,
                    color=color
                )
    
    # Add origin point representing all reference candidates
    plt.scatter([0], [0], color='black', s=50)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Latent Dimension 1 (Liberal-Conservative)', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.title('Discrimination Vectors Relative to Reference Candidates', fontsize=14)
    
    # Set equal aspect ratio
    plt.axis('equal')
    
    # Improve layout
    plt.tight_layout()
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    return plt.gcf()

def process_and_visualize_relative_vectors(model, input_data, participation_mask_tensor, metadata, sample_df, 
                                           reference_candidates=None, output_file=None, contest_filter=None):
    """
    Combined function to post-process parameters and visualize relative vectors.
    
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
    reference_candidates : dict, optional
        Dictionary mapping contest indices to reference candidate indices
    output_file : str, optional
        Path to save the visualization
        
    Returns:
    --------
    adjusted_params : list
        List of adjusted discrimination parameter tensors
    reference_info : dict
        Information about reference candidates
    """
    # Generate voter embeddings for background
    model.eval()
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
    
    # Post-process parameters
    adjusted_params, reference_info = postprocess_discrimination_parameters(
        model, metadata, reference_candidates
    )

    if model.latent_dim == 1:
        visualize_one_dimensional_latent_space(
            model, 
            input_data, 
            participation_mask_tensor, 
            metadata, 
            sample_df, 
            "US PRESIDENT_FEDERAL", 
            "DONALD J TRUMP", 
            "JOSEPH R BIDEN"
        )
    else:
        visualize_relative_discrimination_vectors(
            adjusted_params, 
            reference_info, 
            metadata, 
            voter_embeddings=voter_embeddings_df, 
            output_file=output_file,
            contest_filter=contest_filter
        )
    
    return adjusted_params, reference_info

def visualize_one_dimensional_latent_space(model, input_data, participation_mask_tensor, metadata, 
                                          sample_df, pres_race_name, trump_name, biden_name,
                                          reference_candidates=None, output_file=None):
    """
    Create a one-dimensional visualization of the latent space, showing voter distributions
    and discrimination parameters.
    
    Parameters:
    -----------
    model : VoterChoiceVAE
        Trained VAE model with latent_dim=1
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
    reference_candidates : dict, optional
        Dictionary mapping contest indices to reference candidate indices
    output_file : str, optional
        Path to save the visualization (if None, will display plot)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Check that model has latent_dim=1
    if model.latent_dim != 1:
        print(f"Warning: Model has latent_dim={model.latent_dim}, but this function is designed for latent_dim=1")
    
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
    pres_df = sample_df[(sample_df['office'] == pres_race_name.split('_')[0]) & 
                         (sample_df['district'] == pres_race_name.split('_')[1])]
    
    for _, row in pres_df.iterrows():
        pres_votes[row['cvr_id']] = row['candidate']
    
    # Add presidential vote to embeddings
    voter_embeddings_df['pres_vote'] = voter_embeddings_df['voter_id'].map(pres_votes)
    
    # Post-process discrimination parameters if reference_candidates provided
    if reference_candidates is not None:
        discrimination_params, reference_info = postprocess_discrimination_parameters(
            model, metadata, reference_candidates
        )
    else:
        discrimination_params, _ = model.get_irt_parameters()
        # Convert to list of NumPy arrays for consistency
        discrimination_params = [p.numpy() for p in discrimination_params]
    
    # Create figure with two panels (main plot and discrimination parameters)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                  gridspec_kw={'height_ratios': [3, 1]}, 
                                  sharex=True)
    
    # Panel 1: Voter density distributions
    # --------------------------------
    # Separate voters by presidential vote
    biden_voters = voter_embeddings_df[voter_embeddings_df['pres_vote'] == biden_name]['trait_0']
    trump_voters = voter_embeddings_df[voter_embeddings_df['pres_vote'] == trump_name]['trait_0']
    
    # Plot density distributions
    sns.kdeplot(biden_voters, ax=ax1, fill=True, color='blue', alpha=0.5, label=f'{biden_name} Voters')
    sns.kdeplot(trump_voters, ax=ax1, fill=True, color='red', alpha=0.5, label=f'{trump_name} Voters')
    
    # Add individual voter points at the bottom of the density plots
    y_offset = -0.01  # Small offset to position points below density curves
    ax1.scatter(biden_voters, [y_offset] * len(biden_voters), 
               color='blue', alpha=0.3, s=10, marker='|')
    ax1.scatter(trump_voters, [y_offset] * len(trump_voters), 
               color='red', alpha=0.3, s=10, marker='|')
    
    # Add legend and labels
    ax1.legend()
    ax1.set_title('Voter Distribution on Liberal-Conservative Dimension', fontsize=14)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Determine axis limits with some padding
    all_voters = np.concatenate([biden_voters, trump_voters])
    x_min, x_max = all_voters.min(), all_voters.max()
    x_range = x_max - x_min
    x_padding = x_range * 0.1  # 10% padding
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)
    
    # Add mean lines for each group
    # biden_mean = biden_voters.mean()
    # trump_mean = trump_voters.mean()
    # ax1.axvline(biden_mean, color='blue', linestyle='--', alpha=0.7)
    # ax1.axvline(trump_mean, color='red', linestyle='--', alpha=0.7)
    
    # Add mean value annotations
    # ax1.text(biden_mean, ax1.get_ylim()[1]*0.9, f'Mean: {biden_mean:.2f}', 
    #          color='blue', ha='center', fontsize=10)
    # ax1.text(trump_mean, ax1.get_ylim()[1]*0.8, f'Mean: {trump_mean:.2f}', 
    #          color='red', ha='center', fontsize=10)
    
    # Panel 2: Discrimination parameters
    # --------------------------------
    # Create a horizontal line for the parameter scale
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Define color map for different contests
    contest_colors = plt.cm.tab10.colors
    
    # Track candidates to avoid overlapping text
    used_positions = []
    min_spacing = 0.05  # Minimum spacing between labels
    
    # Plot discrimination parameters for each contest and candidate
    for contest_idx, contest_params in enumerate(discrimination_params):
        # Get race information
        race_name = metadata['idx_to_race'][contest_idx]
        parts = race_name.split('_')
        office = parts[0]
        district = parts[1] if len(parts) > 1 else ""
        
        # Get candidate names
        candidate_map = metadata['candidate_maps'][contest_idx]
        
        # Set color for this contest
        # color = contest_colors[contest_idx % len(contest_colors)]
        
        # Plot each candidate's parameter as a point
        for candidate_idx, candidate_name in sorted([(v, k) for k, v in candidate_map.items()]):
            # Get parameter (just the first dimension)
            param = contest_params[candidate_idx, 0]
            
            # Plot point
            ax2.scatter(param, 0, s=80, marker='o', zorder=3)
            
            # Add candidate label with contest prefix
            label = f"{office}-{district}: {candidate_name}"
            
            # Check for label position overlap
            position_ok = True
            for used_pos in used_positions:
                if abs(param - used_pos) < min_spacing:
                    position_ok = False
                    break
            
            # Add label if no overlap, otherwise try to find a better position
            if position_ok:
                ax2.annotate(label, (param, 0), xytext=(0, -20), textcoords='offset points', ha='center', fontsize=8, rotation=45)
                used_positions.append(param)
            else:
                # Try alternate position above the line
                ax2.annotate(label, (param, 0), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8, rotation=45)
    
    # Set axis labels
    ax2.set_ylabel('', visible=False)
    ax2.set_xlabel('Latent Dimension 1 (Liberal-Conservative)', fontsize=12)
    ax2.set_title('Candidate Discrimination Parameters', fontsize=12)
    
    # Disable y-axis ticks
    ax2.set_yticks([])
    
    # Improve layout
    plt.tight_layout()
    
    # Save or display the plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()
    
    return fig

def postprocess_discrimination_parameters(model, metadata, reference_candidates=None):
    """
    Post-process discrimination parameters by subtracting a reference candidate's parameters.
    
    Parameters:
    -----------
    model : VoterChoiceVAE
        Trained VAE model
    metadata : dict
        Dictionary containing mappings between races/candidates and indices
    reference_candidates : dict, optional
        Dictionary mapping contest indices to reference candidate indices
        If None, the first candidate in each contest will be used as reference
        
    Returns:
    --------
    adjusted_discrimination_params : list
        List of adjusted discrimination parameters arrays
    reference_info : dict
        Information about the reference candidates used
    """
    # Get original discrimination parameters
    discrimination_params, _ = model.get_irt_parameters()
    
    # Initialize adjusted parameters (we'll create new arrays to avoid modifying the model)
    adjusted_discrimination_params = []
    reference_info = {}
    
    # Process each contest
    for contest_idx, race_name in metadata['idx_to_race'].items():
        # Get parameters for this contest
        contest_params = discrimination_params[contest_idx].clone()
        
        # Determine reference candidate
        if reference_candidates is not None and contest_idx in reference_candidates:
            ref_candidate_idx = reference_candidates[contest_idx]
        else:
            # Default to first candidate
            ref_candidate_idx = 0
        
        # Get reference candidate name
        candidate_map = metadata['candidate_maps'][contest_idx]
        ref_candidate_name = [name for name, idx in candidate_map.items() if idx == ref_candidate_idx][0]
        
        # Store reference information
        reference_info[contest_idx] = {
            'index': ref_candidate_idx,
            'name': ref_candidate_name,
            'race': race_name
        }
        
        # Get reference candidate's parameters
        ref_params = contest_params[ref_candidate_idx]
        
        # Subtract reference parameters from all candidates
        for candidate_idx in range(contest_params.size(0)):
            contest_params[candidate_idx] = contest_params[candidate_idx] - ref_params
            
        # Add to adjusted parameters
        adjusted_discrimination_params.append(contest_params)
    
    return adjusted_discrimination_params, reference_info

def visualize_dimensions_as_points(model, metadata, reference_candidates=None, plot_reference=True,
                                  output_file=None, contest_filter=None, 
                                  dim_labels=None, figsize=(15, 10)):
    """
    Visualize each dimension separately as points, with contests on y-axis and values on x-axis.
    
    Parameters:
    -----------
    adjusted_params : list
        List of adjusted discrimination parameter tensors
    reference_info : dict
        Information about reference candidates
    metadata : dict
        Dictionary containing mappings between races/candidates and indices
    output_file : str, optional
        Path to save the visualization
    contest_filter : str, optional
        Only include contests containing this string
    dim_labels : list, optional
        Labels for each dimension (default: ["Liberal-Conservative", "Dimension 2", ...])
    figsize : tuple, optional
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    adjusted_params, reference_info = postprocess_discrimination_parameters(
        model, metadata, reference_candidates
    )

    # Get uncertainty estimates
    discrim_uncertainty = model.get_discrimination_uncertainty()
    diff_uncertainty = model.get_difficulty_uncertainty()

    # Determine number of dimensions from the first non-empty tensor
    for params in adjusted_params:
        if params.size(0) > 0:
            n_dims = params.size(1)
            break
    else:
        raise ValueError("No non-empty parameter tensors found")
    
    # Create default dimension labels if not provided
    if dim_labels is None:
        dim_labels = ["Liberal-Conservative" if i == 0 else f"Dimension {i+1}" for i in range(n_dims)]
    
    # Create figure with subplots (one per dimension)
    fig, axes = plt.subplots(1, n_dims, figsize=figsize, sharey=True)
    if n_dims == 1:
        axes = [axes]  # Handle single dimension case
    
    # Collect all contests and candidates to determine y-axis ordering
    contest_candidates = []
    
    for contest_idx, contest_params in enumerate(adjusted_params):
        # Get contest info
        race_name = reference_info[contest_idx]['race']
        
        # Skip if contest_filter is provided and not in race_name
        if contest_filter is not None and contest_filter not in race_name:
            continue
        
        # Extract office and district
        parts = race_name.split('_')
        office = parts[0]
        district = parts[1] if len(parts) > 1 else ""
        
        # Get candidate names
        candidate_map = metadata['candidate_maps'][contest_idx]
        
        # Add each candidate
        for candidate_idx, candidate_name in sorted([(v, k) for k, v in candidate_map.items()]):
            if not plot_reference and candidate_idx == reference_info[contest_idx]['index']:
                continue
            
            label = f"{office}-{district}: {candidate_name}"
            contest_candidates.append((contest_idx, candidate_idx, label))
    
    # Sort contest candidates by office, district, name for consistent ordering
    contest_candidates.sort(key=lambda x: x[2])
    
    # Create reversed mapping for y-positions
    y_positions = {candidate_tuple: i for i, candidate_tuple in enumerate(contest_candidates)}
    
    # Define color map for different contests
    contest_colors = plt.cm.tab10.colors
    
    # Plot each dimension
    for dim_idx in range(n_dims):
        ax = axes[dim_idx]
        
        # Set dimension title
        ax.set_title(dim_labels[dim_idx], fontsize=12)
        
        # Plot points for each contest/candidate
        for contest_idx, candidate_idx, label in contest_candidates:
            # Get parameter value for this dimension
            if candidate_idx == reference_info[contest_idx]['index']:
                value = 0
            else:
                value = adjusted_params[contest_idx][candidate_idx, dim_idx].item()
            
            # Get y-position
            y_pos = y_positions[(contest_idx, candidate_idx, label)]
            
            # Get color based on contest
            color = contest_colors[contest_idx % len(contest_colors)]
            
            # Plot point
            ax.scatter(value, y_pos, color=color, s=50)
            
            # Add candidate label on the first dimension plot only
            if dim_idx == 0:
                ax.text(-0.1, y_pos, label, fontsize=8, ha='right', va='center')
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # Set x-axis label
        ax.set_xlabel(f"Value", fontsize=10)
        
        # Remove y-axis ticks except for first plot
        if dim_idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticks([])  # No y-axis ticks for first plot either since we have text labels
    
    # Set common y-axis label
    fig.text(0.01, 0.5, 'Contests & Candidates', fontsize=12, va='center', rotation='vertical')
    
    # Add overall title
    plt.suptitle('Discrimination Parameters by Dimension', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Make room for y-axis label
    
    # Adjust horizontal spacing between subplots
    plt.subplots_adjust(wspace=0.05)
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    return output_file