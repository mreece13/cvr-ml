rm(list=ls())
gc()

library(tidyverse)
library(arrow)
library(here)

here::i_am("data/create_sample.R")

a = read_parquet(here("data/all.parquet"))

a |> 
  semi_join(
    distinct(a, state, county_name, cvr_id) |> slice_sample(n=10000, by = c(state, county_name)),
    join_by(state, county_name, cvr_id)
  ) |> 
  write_parquet(here("data/all_sample.parquet"))
