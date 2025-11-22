rm(list=ls())
gc()

library(tidyverse)
library(arrow)

######
# create test file for VAE implementation
######

newmeta = googlesheets4::read_sheet(
  ss = "https://docs.google.com/spreadsheets/d/1Pq9sNcCfLVi-qeXfBy7xEi5lPxpMJYy3LVUHQn_uEFI/edit?gid=1814631761#gid=1814631761",
  sheet = "metadata_2020",
  col_types = "cicccccci"
) |> 
  right_join(
    read_csv("../cvrs/metadata/newmeta_pass0.csv") |> select(election, state, contest_id, candidate_id, office, district, candidate), join_by(election, state, contest_id, candidate_id)
  ) |> 
  mutate(
    race = paste(office.y, district.y, sep = "_")
  ) |> 
  distinct(state, race, office.x, district.x, candidate.y, candidate.x, party, magnitude)

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
  semi_join(compare, join_by(state, county_name)) |> 
  anti_join(small_cands_co, join_by(race, candidate)) |> 
  anti_join(uncontested_co, join_by(race, candidate)) |> 
  mutate(
    state = str_to_lower(state),
    county_name = str_to_lower(county_name),
    candidate = str_to_lower(candidate),
    race = str_to_lower(race)
  ) |> 
  left_join(newmeta, join_by(state, race, candidate == candidate.y)) |> 
  mutate(
    race = paste(office.x, district.x, sep = "_")
  ) |>
  select(state, county_name, cvr_id, race, candidate = candidate.x, magnitude = magnitude.y) |> 
  mutate(
    magnitude = as.integer(magnitude)
  ) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/colorado.parquet")

open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |>
  filter(
    state != "COLORADO",
    (magnitude == 1 | is.na(magnitude)),
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO", "UNDERVOTE", "OVERVOTE")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(state, county_name, cvr_id, race, candidate, magnitude) |> 
  semi_join(compare, join_by(state, county_name)) |>
  anti_join(small_cands, join_by(race, candidate)) |> 
  anti_join(uncontested, join_by(race, candidate)) |> 
  mutate(
    state = str_to_lower(state),
    county_name = str_to_lower(county_name),
    candidate = str_to_lower(candidate),
    race = str_to_lower(race)
  ) |> 
  left_join(newmeta, join_by(state, race, candidate == candidate.y)) |> 
  mutate(
    race = paste(office.x, district.x, sep = "_")
  ) |>
  select(state, county_name, cvr_id, race, candidate = candidate.x, magnitude = magnitude.y) |> 
  mutate(
    magnitude = as.integer(magnitude)
  ) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/all.parquet")

## create new metadata file

c_meta = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
  filter(
    state == "COLORADO",
    (magnitude == 1 | is.na(magnitude)),
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT")),
    candidate != "WRITEIN"
  ) |> 
  distinct(state, office, district, candidate, party, magnitude) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  anti_join(small_cands_co, join_by(race, candidate)) |> 
  arrange(state, office, district, candidate) |> 
  select(-race) |> 
  collect()

a_meta = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |>
  filter(
    state != "COLORADO",
    (magnitude == 1 | is.na(magnitude)),
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO", "UNDERVOTE", "OVERVOTE")),
    candidate != "WRITEIN", candidate != "UNDERVOTE", candidate != "OVERVOTE"
  ) |> 
  distinct(state, office, district, candidate, party_detailed, magnitude) |> 
  arrange(state, office, district, candidate) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  anti_join(small_cands, join_by(race, candidate)) |> 
  select(-race) |> 
  rename(party = party_detailed) |>
  mutate(across(everything(), str_to_lower), magnitude = as.integer(magnitude)) |> 
  collect()

bind_rows(c_meta, a_meta) |> 
  mutate(
    contest_id = cur_group_id(),
    .by = c(state, office, district)
  ) |> 
  mutate(
    candidate_id = paste(contest_id, 1:n(), sep = "-"),
    .by = contest_id
  ) |> 
  mutate(
    across(everything(), str_to_lower),
    election = "2020 General"
  ) |> 
  write_csv("../cvrs/metadata/newmeta_pass0.csv")
  clipr::write_clip()

newmeta = googlesheets4::read_sheet(
  ss = "https://docs.google.com/spreadsheets/d/1Pq9sNcCfLVi-qeXfBy7xEi5lPxpMJYy3LVUHQn_uEFI/edit?gid=1814631761#gid=1814631761",
  sheet = "metadata_2020",
  col_types = "cicccccci"
) |> 
  right_join(
    read_csv("../cvrs/metadata/newmeta_pass0.csv") |> select(election, state, contest_id, candidate_id, office, district, candidate), join_by(election, state, contest_id, candidate_id)
  ) |> 
  mutate(
    race = paste(office.y, district.y, sep = "_")
  )
