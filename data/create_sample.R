rm(list=ls())
gc()

library(tidyverse)
library(arrow)
library(here)

here::i_am("data/create_sample.R")

a = bind_rows(
  read_parquet(here("data/all.parquet")),
  read_parquet(here("data/colorado.parquet"))
)

write_parquet(a, here("data/combined.parquet"))

a |> 
  semi_join(
    distinct(a, state, county_name, cvr_id) |> slice_sample(n=10000, by = c(state, county_name)),
    join_by(state, county_name, cvr_id)
  ) |> 
  write_parquet(here("data/combined_sample.parquet"))