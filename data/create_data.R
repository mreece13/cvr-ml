rm(list=ls())
gc()

library(tidyverse)
library(arrow)

######
# create test file for VAE implementation
######

compare = readxl::read_excel("~/Dropbox (MIT)/Research/CVR_parquet/combined/compare.xlsx", sheet = "by-county") |> 
  filter(release == 1) |> 
  select(state, county_name)

small_cands_co = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
  filter(
    state == "COLORADO",
    magnitude == 1,
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  count(race, candidate) |> 
  filter(n <= 50)

small_cands = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |> 
  filter(
    state != "COLORADO",
    magnitude == 1,
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  count(race, candidate) |> 
  filter(n <= 50)

uncontested_co = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
  filter(
    state == "COLORADO",
    magnitude == 1,
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  distinct(race, candidate) |> 
  collect() |> 
  filter(n() == 1, .by = race)

uncontested = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |>
  filter(
    state != "COLORADO",
    magnitude == 1,
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  distinct(race, candidate) |> 
  collect() |> 
  filter(n() == 1, .by = race)

c = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
  filter(
    state == "COLORADO",
    (magnitude == 1 | is.na(magnitude)),
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(state, county_name, cvr_id, race, candidate, magnitude) |> 
  anti_join(small_cands_co, join_by(race, candidate)) |> 
  anti_join(uncontested_co, join_by(race, candidate)) |> 
  collect()

write_parquet(c, "~/Dropbox (MIT)/Research/cvr-ml/data/colorado.parquet")

a = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |>
  filter(
    state != "COLORADO",
    (magnitude == 1 | is.na(magnitude)),
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(state, county_name, cvr_id, race, candidate, magnitude) |> 
  semi_join(compare, join_by(state, county_name)) |>
  anti_join(small_cands, join_by(race, candidate)) |> 
  anti_join(uncontested, join_by(race, candidate))

write_parquet(a, "~/Dropbox (MIT)/Research/cvr-ml/data/all.parquet")

a |> 
  semi_join(
    distinct(a, state, county_name, cvr_id) |> slice_sample(n=10000, by = c(state, county_name)),
    join_by(state, county_name, cvr_id)
  ) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/all_sample.parquet")
