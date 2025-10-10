rm(list = ls())
gc()

library(tidyverse)
library(keras3)

source("utils.R")
source("model.R")

reticulate::use_virtualenv("r-tensorflow")

##################################
# Data Loading
##################################
data = arrow::read_parquet("data/colorado_sample.parquet")

# Assign unique IDs to races and candidates
ids <- data |>
  filter(candidate != "UNDERVOTE") |>
  count(race, candidate) |>
  arrange(race, desc(n)) |>
  group_by(race) |>
  mutate(candidate_id = 1:n(), race_id = cur_group_id())

# Join back to the original data
df <- data |>
  filter(candidate != "UNDERVOTE") |>
  left_join(ids, join_by(race, candidate))

# Create the votes matrix
votes_matrix <- df |>
  select(county_name, cvr_id, race_id, candidate_id, n) |>
  arrange(race_id, desc(n)) |>
  select(-n) |>
  pivot_wider(names_from = race_id, values_from = candidate_id, values_fill = 0) |>
  select(-cvr_id, -county_name) |>
  as.matrix()

# stan_data <- list(
#   N_voters = distinct(df, county_name, cvr_id) |> tally() |> pull(),
#   N_items = n_distinct(ids$race),
#   N_cands = length(ids$candidate),
#   votes = votes_matrix,
#   sizes = distinct(df, race, race_id, candidate) |> count(race_id) |> pull(n)
# )

##################################
# Build Model
##################################
# Model parameters
N_items <- ncol(votes_matrix)
N_dims <- 1
N_voters <- distinct(df, county_name, cvr_id) |> tally() |> pull()
Q = matrix(1, nrow = N_dims, ncol = N_items)

c(encoder, decoder, vae) %<-% build_vae_independent(
  N_items = N_items,
  N_dims = N_dims,
  Q = Q,
  enc_hid_arch = c(16L, 8L),
  # hid_enc_activations = c('relu', 'tanh'),
  output_activation = 'softmax',
  kl_weight = 1,
  learning_rate = 0.001
)

##################################
# Fit and Evaluate Model
##################################
# Training parameters
num_train <- floor(0.8 * N_voters)
train_idx <- sample(N_voters, num_train)
data_train <- votes_matrix[train_idx, ]
data_test <- votes_matrix[-train_idx, ]

history <- fit(
  vae,
  data_train,
  data_train,
  epochs = 5,
  validation_split = 0.15,
  shuffle = FALSE,
  verbose = 1,
  batch_size = 8
)

plot(history)
# history[2]$metrics$loss

# Get IRT parameter estimates
c(diff_est, disc_est) %<-% get_item_parameter_estimates(decoder)

test_theta_est <- get_ability_parameter_estimates(encoder, data_test)[[1]]
all_theta_est <- get_ability_parameter_estimates(encoder, votes_matrix)[[1]]
