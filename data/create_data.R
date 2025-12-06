rm(list=ls())
gc()

library(tidyverse)
library(arrow)

######
# create test file for VAE implementation
######

# newmeta = googlesheets4::read_sheet(
#   ss = "https://docs.google.com/spreadsheets/d/1Pq9sNcCfLVi-qeXfBy7xEi5lPxpMJYy3LVUHQn_uEFI/edit?gid=1814631761#gid=1814631761",
#   sheet = "metadata_2020",
#   col_types = "cicccccci"
# )

newmeta = read_csv("metadata/newmeta.csv") |> 
  group_by(contest_id, candidate_id) |> 
  fill(party, magnitude, .direction = "downup") |> 
  ungroup() |> 
  right_join(
    read_csv("metadata/newmeta_pass0.csv") |> 
      select(election, state, contest_id, candidate_id, office, district, candidate), 
    join_by(election, state, contest_id, candidate_id),
    suffix = c(".new", ".old")
  ) |> 
  group_by(state, office.new, district.new) |> 
  fill(magnitude, .direction = "downup") |> 
  group_by(state, office.new, district.new, candidate.new) |> 
  fill(party, .direction = "downup") |> 
  ungroup() |> 
  distinct(state, office.old, district.old, candidate.old, office.new, district.new, candidate.new, party, magnitude)

compare = readxl::read_excel("metadata/compare.xlsx", sheet = "by-county") |> 
  filter(release == 1) |> 
  select(state, county_name)

base_co = open_dataset("~/orcd/pool/supercloud-cvrs/data/pass3") |> 
  filter(
    magnitude == 1,
    !str_detect(candidate, regex("undervote|overvote|writein", ignore_case = TRUE))
  ) |> 
  semi_join(compare, join_by(state, county_name)) |>
  mutate(
    race = paste(office, district, sep = "_")
  )

base = open_dataset("~/orcd/pool/supercloud-cvrs/data/pass2") |> 
  filter(
    state != "COLORADO",
    magnitude == 1,
    !str_detect(candidate, regex("undervote|overvote|writein", ignore_case = TRUE))
  ) |> 
  semi_join(compare, join_by(state, county_name)) |>
  mutate(
    race = paste(office, district, sep = "_")
  )

small_cands_co = count(base_co, race, candidate) |> filter(n <= 50)
small_cands = count(base, race, candidate) |> filter(n <= 50)

uncontested_co = base_co 
  distinct(race, candidate) |> 
  collect() |> 
  filter(n() == 1, .by = race)

uncontested = base |> 
  distinct(race, candidate) |> 
  collect() |> 
  filter(n() == 1, .by = race)

base_co |> 
  select(state, county_name, precinct, cvr_id, race, office, district, candidate, magnitude) |>
  anti_join(small_cands_co, join_by(race, candidate)) |> 
  anti_join(uncontested_co, join_by(race, candidate)) |> 
  mutate(
    state = str_to_lower(state),
    county_name = str_to_lower(county_name),
    candidate = str_to_lower(candidate),
    race = str_to_lower(race),
    office = str_to_lower(office),
    district = str_to_lower(district)
  ) |> 
  left_join(newmeta, join_by(state, office == office.old, district == district.old, candidate == candidate.old)) |> 
  mutate(
    magnitude = as.integer(magnitude.y),
    race = paste(office.new, district.new, sep = "_")
  ) |> 
  select(state, county_name, precinct, cvr_id, race, candidate = candidate.new, magnitude) |> 
  distinct() |> 
  filter(
    magnitude == 1 | is.na(magnitude), candidate != "writein"
  ) |> 
  write_parquet(here("data/colorado.parquet"))

base |>
  mutate(
    magnitude = as.integer(magnitude)
  ) |> 
  filter(
    state != "COLORADO", county_name != "MARICOPA",
    (magnitude == 1 | is.na(magnitude)),
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO", "UNDERVOTE", "OVERVOTE")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(state, county_name, precinct, cvr_id, race, office, district, candidate, magnitude) |> 
  anti_join(small_cands, join_by(race, candidate)) |> 
  anti_join(uncontested, join_by(race, candidate)) |> 
  mutate(
    state = str_to_lower(state),
    county_name = str_to_lower(county_name),
    candidate = str_to_lower(candidate),
    race = str_to_lower(race),
    office = str_to_lower(office),
    district = str_to_lower(district)
  ) |> 
  left_join(newmeta, join_by(state, office == office.old, district == district.old, candidate == candidate.old)) |> 
  mutate(
    magnitude = as.integer(magnitude.y),
    race = paste(office.new, district.new, sep = "_")
  ) |> 
  select(state, county_name, precinct, cvr_id, race, candidate = candidate.new, magnitude) |> 
  filter(
    magnitude == 1 | is.na(magnitude), candidate != "writein"
  ) |> 
  write_parquet("data/all.parquet")

#### archive
## create new metadata file

# c_meta = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
#   filter(
#     state == "COLORADO",
#     (magnitude == 1 | is.na(magnitude)),
#     !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT")),
#     candidate != "WRITEIN"
#   ) |> 
#   distinct(state, office, district, candidate, party, magnitude) |> 
#   mutate(
#     race = paste(office, district, sep = "_")
#   ) |> 
#   anti_join(small_cands_co, join_by(race, candidate)) |> 
#   arrange(state, office, district, candidate) |> 
#   select(-race) |> 
#   collect()

# a_meta = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |>
#   filter(
#     state != "COLORADO",
#     (magnitude == 1 | is.na(magnitude)),
#     !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT") & candidate %in% c("YES", "NO", "UNDERVOTE", "OVERVOTE")),
#     candidate != "WRITEIN", candidate != "UNDERVOTE", candidate != "OVERVOTE"
#   ) |> 
#   distinct(state, office, district, candidate, party_detailed, magnitude) |> 
#   arrange(state, office, district, candidate) |> 
#   mutate(
#     race = paste(office, district, sep = "_")
#   ) |> 
#   anti_join(small_cands, join_by(race, candidate)) |> 
#   select(-race) |> 
#   rename(party = party_detailed) |>
#   mutate(across(everything(), str_to_lower), magnitude = as.integer(magnitude)) |> 
#   collect()

# bind_rows(c_meta, a_meta) |> 
#   mutate(
#     contest_id = cur_group_id(),
#     .by = c(state, office, district)
#   ) |> 
#   mutate(
#     candidate_id = paste(contest_id, 1:n(), sep = "-"),
#     .by = contest_id
#   ) |> 
#   mutate(
#     across(everything(), str_to_lower),
#     election = "2020 General"
#   ) |> 
#   write_csv("../cvrs/metadata/newmeta_pass0.csv")
#   clipr::write_clip()
