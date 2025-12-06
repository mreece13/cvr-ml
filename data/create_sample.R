rm(list=ls())
gc()

library(tidyverse)
library(arrow)
library(here)

a = bind_rows(
  read_parquet(here("data/all.parquet")),
  read_parquet(here("data/colorado.parquet"))
)

write_parquet(a, here("data/combined.parquet"))

# pick random precincts to get good coverage
random_precincts <- a |> 
  distinct(state, county_name, precinct) |> 
  collect() |> 
  slice_sample(prop = 0.1, by = county_name)

# pick some random people
# randoms <- base_data |>
#   inner_join(random_precincts, join_by(state, county_name, precinct)) |>
#   distinct(state, county_name, precinct, cvr_id) |>
#   collect() |> 
#   slice_sample(n=500, by = c(county_name, precinct)) |> 
#   distinct(state, county_name, cvr_id)

a |> 
  semi_join(random_precincts, join_by(state, county_name, precinct)) |> 
  write_parquet(here("data/combined_precinctSample.parquet"))

a |> 
  semi_join(
    distinct(a, state, county_name, cvr_id) |> slice_sample(n=5000, by = c(state, county_name)),
    join_by(state, county_name, cvr_id)
  ) |> 
  write_parquet(here("data/combined_sample.parquet"))