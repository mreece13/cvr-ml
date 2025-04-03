import torch
import polars as pl
from model_vae import *
from data_preparation import *

# Prepare data for the VAE model
save_dir = "intermediates"
load_data = True
save_data = True
voters_per_county = 2000 # used for sampling

if load_data:
    input_data, target_indices, participation_mask_tensor, num_candidates_per_contest, metadata = load_prepared_data(save_dir)
else:
    final_data = pl.scan_parquet("data/colorado_prepared.parquet")

    if voters_per_county > 0:
        ids = final_data.filter(pl.int_range(pl.len()).shuffle().over("county_name") < voters_per_county).select(['county_name', "cvr_id"])

        final_data = final_data.join(
            ids,
            on=["county_name", "cvr_id"],
            how="inner"
        )

    input_data, target_indices, participation_mask_tensor, num_candidates_per_contest, metadata = load_and_prepare_voter_data(final_data)
    
    if save_data:
        save_prepared_data(save_dir, (input_data, target_indices, participation_mask_tensor, num_candidates_per_contest, metadata))


# Create and train the model
# Model parameters
hidden_dim = 64
latent_dim = 2
num_epochs = 20
batch_size = 256
learning_rate = 1e-3

pres_race_name = "US PRESIDENT_FEDERAL"
trump_name = "DONALD J TRUMP"
biden_name = "JOSEPH R BIDEN"

model = VoterChoiceVAE(
    num_contests=len(num_candidates_per_contest),
    num_candidates_per_contest=num_candidates_per_contest,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim
)
model = model.to(get_device())

# Train with constraint
soft_model = train_voter_vae_constrained(
    model, 
    input_data, 
    target_indices,
    participation_mask_tensor,
    metadata, 
    pres_race_name, 
    trump_name, 
    biden_name,
    num_epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    kl_start_weight=0.1,
    kl_end_weight=2,
    kl_annealing='cyclical',
    constraint_weight=2
)

# Apply post-processing to ensure consistent orientation of all latent dimensions
soft_model = post_process_latent_space(soft_model, metadata, pres_race_name, trump_name, biden_name)

# Save only the model parameters
torch.save(soft_model.state_dict(), 'models/vae_sample2000_noCollapse.pt')