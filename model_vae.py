import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import sparse

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VoterChoiceVAE(nn.Module):
    def __init__(self, num_contests, num_candidates_per_contest, hidden_dim, latent_dim, device=None):
        """
        VAE model for voter choice modeling inspired by 2-parameter IRT,
        handling variable contest participation across voters
        
        Parameters:
        -----------
        num_contests: int
            Total number of possible ballot contests/races
        num_candidates_per_contest: list
            List with number of candidates for each contest
        hidden_dim: int
            Dimension of hidden layers
        latent_dim: int
            Dimension of the latent space (voter traits/preferences)
        """
        super(VoterChoiceVAE, self).__init__()

        self.device = device if device is not None else get_device()
        self.num_contests = num_contests
        self.num_candidates_per_contest = num_candidates_per_contest
        self.latent_dim = latent_dim
        
        # Calculate total possible candidates across all contests
        self.total_candidates = sum(num_candidates_per_contest)
        
        # Encoder network (takes sparse input of voter choices)
        self.encoder = nn.Sequential(
            nn.Linear(self.total_candidates, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder networks - one per contest
        self.contest_decoders = nn.ModuleList()
        for num_candidates in num_candidates_per_contest:
            decoder = nn.Linear(latent_dim, num_candidates)
            # Initialize to mimic typical IRT parameter ranges
            nn.init.normal_(decoder.weight, mean=0.0, std=0.5)  # discrimination typically 0-3
            nn.init.normal_(decoder.bias, mean=0.0, std=1.0)    # difficulty typically -3 to 3
            self.contest_decoders.append(decoder)
        
        # Store the starting index for each contest in the flattened representation
        self.contest_start_indices = [0]
        for i in range(len(num_candidates_per_contest) - 1):
            self.contest_start_indices.append(
                self.contest_start_indices[-1] + num_candidates_per_contest[i]
            )
    
    def encode(self, x, mask=None):
        """
        Encode voter choices to latent trait distribution
        
        Parameters:
        -----------
        x: torch.Tensor
            One-hot encoded voter choices (can be sparse)
        mask: torch.Tensor
            Binary mask indicating which contests each voter participated in
            
        Returns:
        --------
        mu, logvar: torch.Tensor
            Parameters for the latent distribution
        """
        # If input is sparse, we accept it as is
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, mask=None):
        """
        Decode latent traits to contest choice probabilities
        
        Parameters:
        -----------
        z: torch.Tensor
            Latent voter traits
        mask: torch.Tensor
            Binary mask indicating which contests each voter participated in
            Shape: (batch_size, num_contests)
            
        Returns:
        --------
        contest_probs: list of torch.Tensor
            Predicted probabilities for each contest, only for contests the voter participated in
        """
        contest_probs = []
        
        for contest_idx, decoder in enumerate(self.contest_decoders):
            # Calculate logits for all voters
            logits = decoder(z)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            
            # If mask is provided, zero out probabilities for contests voter didn't participate in
            if mask is not None:
                # Extract mask for this contest
                contest_mask = mask[:, contest_idx].unsqueeze(1)
                # Apply mask (will be used in loss function)
                probs = probs * contest_mask
            
            contest_probs.append(probs)
        
        return contest_probs
    
    def forward(self, x, mask=None):
        """
        Full forward pass
        
        Parameters:
        -----------
        x: torch.Tensor
            One-hot encoded voter choices
        mask: torch.Tensor
            Binary mask indicating which contests each voter participated in
            
        Returns:
        --------
        contest_probs: list of torch.Tensor
            Predicted probabilities for each contest
        mu, logvar: torch.Tensor
            Latent space parameters
        """
        mu, logvar = self.encode(x, mask)
        z = self.reparameterize(mu, logvar)
        contest_probs = self.decode(z, mask)
        return contest_probs, mu, logvar
    
    def get_irt_parameters(self):
        """Extract IRT-equivalent parameters from the model"""
        discrimination_params = [decoder.weight.data for decoder in self.contest_decoders]
        difficulty_params = [decoder.bias.data for decoder in self.contest_decoders]
        
        return discrimination_params, difficulty_params

def prepare_sparse_voter_data(raw_data, participation_mask, num_candidates_per_contest):
    """
    Prepare voter data for the VAE model, handling sparse participation
    
    Parameters:
    -----------
    raw_data: numpy array or sparse matrix of shape (num_voters, num_contests)
              Each element is the index of the chosen candidate (0-indexed)
              Can contain -1 or NaN for contests voter didn't participate in
    participation_mask: numpy array of shape (num_voters, num_contests)
              Binary indicators of which contests each voter participated in
    num_candidates_per_contest: list 
              Number of candidates for each contest
    
    Returns:
    --------
    input_data: torch.Tensor
              Sparse one-hot encoded voter choices
    target_indices: list of torch.Tensor
              Indices of chosen candidates for each contest (only for participants)
    participation_mask_tensor: torch.Tensor
              Binary mask indicating participation in each contest
    """
    num_voters, num_contests = raw_data.shape
    
    # Create target indices for each contest, only for participating voters
    target_indices = []
    for contest_idx in range(num_contests):
        # Get indices of voters who participated in this contest
        participant_indices = np.where(participation_mask[:, contest_idx])[0]
        
        # Get their choices
        if isinstance(raw_data, np.ndarray):
            choices = raw_data[participant_indices, contest_idx]
        else:  # Sparse matrix
            choices = np.array(raw_data[participant_indices, contest_idx]).flatten()
            
        # Convert to tensor
        target_indices.append((torch.tensor(participant_indices, dtype=torch.long),
                               torch.tensor(choices, dtype=torch.long)))
    
    # Create sparse one-hot encoded input data
    # Calculate total number of candidates across all contests
    total_candidates = sum(num_candidates_per_contest)
    
    # Initialize sparse matrix in COO format
    row_indices = []
    col_indices = []
    values = []
    
    # Calculate starting position for each contest in the flattened representation
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
            if isinstance(raw_data, np.ndarray):
                candidate_idx = raw_data[voter_idx, contest_idx]
            else:  # Sparse matrix
                candidate_idx = raw_data[voter_idx, contest_idx]
            
            # Skip if the value is invalid
            if candidate_idx < 0 or np.isnan(candidate_idx):
                continue
                
            # Calculate position in the flattened representation
            col_idx = contest_start_indices[contest_idx] + int(candidate_idx)
            
            # Add to sparse representation
            row_indices.append(voter_idx)
            col_indices.append(col_idx)
            values.append(1.0)
    
    # Create sparse tensor
    i = torch.tensor([row_indices, col_indices], dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float)
    input_data = torch.sparse.FloatTensor(i, v, (num_voters, total_candidates)).to_dense()
    
    # Convert participation mask to tensor
    participation_mask_tensor = torch.tensor(participation_mask, dtype=torch.float)

    # At the end of the function, move tensors to device
    if device is None:
        device = get_device()
        
    input_data = input_data.to(device)
    participation_mask_tensor = participation_mask_tensor.to(device)
    
    # Update target_indices to be on the device
    device_target_indices = []
    for indices, choices in target_indices:
        device_target_indices.append((
            indices.to(device),
            choices.to(device)
        ))
    
    return input_data, target_indices, participation_mask_tensor

def voter_vae_loss(contest_probs, target_indices, participation_mask, mu, logvar, kl_weight=1.0):
    """
    Loss function for the voter choice VAE with variable contest participation
    
    Parameters:
    -----------
    contest_probs: list of torch.Tensor
        Predicted probabilities for each contest
    target_indices: list of (indices, choices) tuples
        For each contest, indices of participating voters and their choices
    participation_mask: torch.Tensor
        Binary mask indicating which contests each voter participated in
    mu, logvar: torch.Tensor
        Latent space parameters
    kl_weight: float
        Weight for KL divergence term
        
    Returns:
    --------
    total_loss: torch.Tensor
        Combined loss value
    recon_loss: torch.Tensor
        Reconstruction loss component
    kl_loss: torch.Tensor
        KL divergence loss component
    """
    # Reconstruction loss (categorical cross-entropy for each contest)
    recon_loss = 0.0
    total_contests = 0
    
    for contest_idx, ((voter_indices, targets), probs) in enumerate(zip(target_indices, contest_probs)):
        if len(voter_indices) == 0:
            continue  # Skip contests with no participants in this batch
            
        # Get predictions for participating voters
        participant_probs = probs[voter_indices]
        
        # Calculate cross entropy loss for this contest
        contest_loss = F.cross_entropy(participant_probs, targets)
        recon_loss += contest_loss
        total_contests += 1
    
    # Average loss over actual contests
    if total_contests > 0:
        recon_loss = recon_loss / total_contests
    
    # KL divergence - regularizes the latent space
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Scale KL loss by number of elements
    kl_loss = kl_loss / mu.size(0)
    
    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss
    
    return total_loss, recon_loss, kl_loss

# def train_voter_vae(model, input_data, target_indices, participation_mask, 
#                     num_epochs=100, batch_size=64, learning_rate=1e-3, kl_weight=1.0):
#     """
#     Train the voter choice VAE model
    
#     Parameters:
#     -----------
#     model: VoterChoiceVAE
#         The VAE model
#     input_data: torch.Tensor
#         One-hot encoded voter choices
#     target_indices: list of (indices, choices) tuples
#         For each contest, indices of participating voters and their choices
#     participation_mask: torch.Tensor
#         Binary mask indicating which contests each voter participated in
#     """
#     # Create dataset and dataloader
#     dataset = TensorDataset(input_data, participation_mask)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     model.train()
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         total_recon = 0.0
#         total_kl = 0.0
        
#         for batch_idx, (data, mask) in enumerate(dataloader):
#             optimizer.zero_grad()
            
#             # Forward pass
#             contest_probs, mu, logvar = model(data, mask)
            
#             # Extract targets for this batch
#             batch_start = batch_idx * batch_size
#             batch_end = min((batch_idx + 1) * batch_size, len(input_data))
            
#             # Create batch-specific target indices
#             batch_targets = []
#             for contest_idx, (voter_indices, targets) in enumerate(target_indices):
#                 # Find voters in this batch who participated in this contest
#                 batch_mask = (voter_indices >= batch_start) & (voter_indices < batch_end)
#                 batch_voter_indices = voter_indices[batch_mask] - batch_start
#                 batch_voter_targets = targets[batch_mask]
                
#                 batch_targets.append((batch_voter_indices, batch_voter_targets))
            
#             # Compute loss
#             loss, recon, kl = voter_vae_loss(contest_probs, batch_targets, mask, mu, logvar, kl_weight)
            
#             # Backward pass and optimize
#             loss.backward()
#             optimizer.step()
            
#             # Track metrics
#             total_loss += loss.item()
#             total_recon += recon.item()
#             total_kl += kl.item()
        
#         # Print progress
#         avg_loss = total_loss / len(dataloader)
#         avg_recon = total_recon / len(dataloader)
#         avg_kl = total_kl / len(dataloader)
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
#               f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}")
    
#     return model

def voter_vae_loss_constrained(contest_probs, target_indices, participation_mask, mu, logvar, 
                               model, pres_contest_idx, trump_idx, biden_idx, 
                               kl_weight=1.0, constraint_weight=10.0):
    """
    Loss function with soft constraint for presidential candidate ordering.
    
    Parameters:
    -----------
    (... standard parameters for voter_vae_loss ...)
    model: VoterChoiceVAE
        The VAE model
    pres_contest_idx: int
        Index of the presidential contest
    trump_idx: int
        Index of Trump within the presidential contest
    biden_idx: int
        Index of Biden within the presidential contest
    constraint_weight: float
        Weight for the constraint penalty term
    """
    # Calculate standard loss components
    recon_loss = 0.0
    total_contests = 0
    
    for contest_idx, ((voter_indices, targets), probs) in enumerate(zip(target_indices, contest_probs)):
        if len(voter_indices) == 0:
            continue  # Skip contests with no participants in this batch
            
        # Get predictions for participating voters
        participant_probs = probs[voter_indices]
        
        # Calculate cross entropy loss for this contest
        contest_loss = F.cross_entropy(participant_probs, targets)
        recon_loss += contest_loss
        total_contests += 1
    
    # Average loss over actual contests
    if total_contests > 0:
        recon_loss = recon_loss / total_contests
    
    # KL divergence - regularizes the latent space
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / mu.size(0)
    
    # Presidential candidate constraint penalty
    # Get the discrimination parameters for Trump and Biden
    trump_param = model.contest_decoders[pres_contest_idx].weight[trump_idx]
    biden_param = model.contest_decoders[pres_contest_idx].weight[biden_idx]
    
    # Calculate penalty: ReLU(Biden - Trump)
    # This gives zero when Trump > Biden and positive when Biden > Trump
    # Calculate penalty only for dimension 0 (first dimension)
    constraint_penalty = torch.relu(biden_param[0] - trump_param[0])
    
    # Total loss with constraint
    total_loss = recon_loss + kl_weight * kl_loss + constraint_weight * constraint_penalty
    
    return total_loss, recon_loss, kl_loss, constraint_penalty

def train_voter_vae_constrained(model, input_data, target_indices, participation_mask_tensor, 
                               metadata, pres_race_name, trump_name, biden_name,
                               num_epochs=100, batch_size=64, learning_rate=1e-3, 
                               kl_weight=1.0, constraint_weight=10.0):
    """
    Train the voter choice VAE model with presidential candidate constraint
    
    Parameters:
    -----------
    model: VoterChoiceVAE
        The VAE model
    input_data, target_indices, participation_mask_tensor:
        Standard model inputs
    metadata: dict
        Metadata containing mappings between races/candidates and indices
    pres_race_name: str
        Name of the presidential race (e.g., "President_1")
    trump_name: str
        Name of Trump in the data (e.g., "DONALD J TRUMP")
    biden_name: str
        Name of Biden in the data (e.g., "JOSEPH R BIDEN")
    """

    # Get the device
    device = model.device
    
    # Move input data to device
    input_data = input_data.to(device)
    participation_mask = participation_mask.to(device)
    
    # Move target indices to device
    device_target_indices = []
    for contest_idx, (voter_indices, targets) in enumerate(target_indices):
        device_target_indices.append((
            voter_indices.to(device),
            targets.to(device)
        ))
    
    # Get indices for constraint
    pres_contest_idx = metadata['race_to_idx'][pres_race_name]
    candidate_map = metadata['candidate_maps'][pres_contest_idx]
    trump_idx = candidate_map[trump_name]
    biden_idx = candidate_map[biden_name]
    
    print(f"Applying constraint: {trump_name} (idx {trump_idx}) > {biden_name} (idx {biden_idx})")
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(input_data, participation_mask_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_constraint = 0.0
        
        for batch_idx, (data, mask) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            contest_probs, mu, logvar = model(data, mask)
            
            # Extract targets for this batch
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(input_data))
            
            # Create batch-specific target indices
            batch_targets = []
            for contest_idx, (voter_indices, targets) in enumerate(target_indices):
                # Find voters in this batch who participated in this contest
                batch_mask = (voter_indices >= batch_start) & (voter_indices < batch_end)
                batch_voter_indices = voter_indices[batch_mask] - batch_start
                batch_voter_targets = targets[batch_mask]
                
                batch_targets.append((batch_voter_indices, batch_voter_targets))
            
            # Compute loss with constraint
            loss, recon, kl, constraint = voter_vae_loss_constrained(
                contest_probs, batch_targets, mask, mu, logvar, 
                model, pres_contest_idx, trump_idx, biden_idx,
                kl_weight, constraint_weight
            )
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_constraint += constraint.item()
        
        # Print progress
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_constraint = total_constraint / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, "
              f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, "
              f"Constraint: {avg_constraint:.4f}")
        
        # Check parameter values across all dimensions
        with torch.no_grad():
            trump_param = model.contest_decoders[pres_contest_idx].weight[trump_idx]
            biden_param = model.contest_decoders[pres_contest_idx].weight[biden_idx]
            
            print(f"  First dimension - Trump: {trump_param[0]:.4f}, Biden: {biden_param[0]:.4f}, "
                  f"Diff: {(trump_param[0] - biden_param[0]):.4f}")
            
            # if model.latent_dim > 1:
            #     print("  Other dimensions (unconstrained):")
            #     for dim in range(1, model.latent_dim):
            #         print(f"    Dim {dim}: Trump: {trump_param[dim]:.4f}, Biden: {biden_param[dim]:.4f}, "
            #              f"Diff: {(trump_param[dim] - biden_param[dim]):.4f}")
    
    return model

def post_process_latent_space(model, metadata, pres_race_name, trump_name, biden_name):
    """
    Post-process the model to ensure consistent orientation of latent dimensions.
    
    This function:
    1. Ensures the first dimension has Trump > Biden
    2. For other dimensions, flips them if necessary to ensure consistent orientation
       based on discrimination parameter signs
    """
    with torch.no_grad():
        # Get presidential race indices
        pres_contest_idx = metadata['race_to_idx'][pres_race_name]
        candidate_map = metadata['candidate_maps'][pres_contest_idx]
        trump_idx = candidate_map[trump_name]
        biden_idx = candidate_map[biden_name]
        
        # Get presidential decoder weights
        pres_decoder = model.contest_decoders[pres_contest_idx]
        
        # Ensure first dimension has Trump > Biden
        if pres_decoder.weight[trump_idx, 0] < pres_decoder.weight[biden_idx, 0]:
            print("Flipping first dimension to ensure Trump > Biden")
            # Flip signs for this dimension across all decoders
            for decoder in model.contest_decoders:
                decoder.weight[:, 0] *= -1
        
        # For other dimensions, check if majority of discriminations are negative
        # and flip if needed for consistency
        for dim in range(1, model.latent_dim):
            # Count positive and negative discrimination parameters
            positive_count = 0
            negative_count = 0
            
            for decoder in model.contest_decoders:
                for i in range(decoder.weight.size(0)):
                    if decoder.weight[i, dim] > 0:
                        positive_count += 1
                    else:
                        negative_count += 1
            
            # If more negative than positive, flip this dimension
            if negative_count > positive_count:
                print(f"Flipping dimension {dim} for consistent orientation")
                for decoder in model.contest_decoders:
                    decoder.weight[:, dim] *= -1
        
        print("Post-processing complete. Latent space now has consistent orientation.")
    
    return model