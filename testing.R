rm(list=ls())
gc()

library(tidyverse)
library(arrow)
library(data.table)

get_gsheet <- function(url, sheet, ...) {
  googlesheets4::read_sheet(
    ss = url,
    sheet = sheet,
    ...
  )
}

metadata = get_gsheet(
  "https://docs.google.com/spreadsheets/d/1Pq9sNcCfLVi-qeXfBy7xEi5lPxpMJYy3LVUHQn_uEFI/edit?gid=1550619837#gid=1550619837",
  "metadata",
  col_types = "c"
) |>
  mutate(
    across(office:prop_text, str_to_lower),
    district = str_replace(district, "county_name", str_to_lower(county)),
    district = str_replace(district, "statewide", str_to_lower(state)),
    magnitude = as.integer(magnitude)
  ) |>
  group_by(raw_candidate) |>
  fill(party) |>
  ungroup() |>
  group_by(candidate, party) |>
  fill(office, district, .direction = "updown") |>
  ungroup() |> 
  filter()

rcvr::clean_cvr(
  path = "~/Dropbox (MIT)/Research/cvrs/data/raw/Colorado/Garfield/cvr.csv",
  metadata = filter(metadata, state == "Colorado"),
  write_path = "~/Dropbox (MIT)/Research/cvrs/data/pass2/state=COLORADO/county_name=GARFIELD/part-0.parquet"
)

open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass2") |> 
  filter(
    state == "COLORADO",
    magnitude == "1",
    !(office %in% c("SUPREME COURT", "COURT OF APPEALS", "COUNTY JUDGE", "DISTRICT COURT")),
    candidate != "WRITEIN"
  ) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(state, county_name, cvr_id, race, candidate, magnitude) |> 
  # collect() |> 
  # filter(magnitude == "1") |> 
  # select(-magnitude) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/colorado.parquet")

d = read_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/colorado.parquet")

d |> 
  semi_join(
    distinct(d, county_name, cvr_id) |> slice_sample(n=1000, by = county_name),
    join_by(county_name, cvr_id)
  ) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/colorado_sample.parquet")

d |> 
  filter(n()>1, .by = c(state, county_name, cvr_id, race)) |> 
  distinct(county_name, race)

cands = open_dataset("../cvrs/data/pass2") |> 
  filter(state == "COLORADO") |> 
  distinct(office, district, candidate, party_detailed) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(race, candidate, party_detailed) |> 
  collect() |> 
  group_by(race, candidate) |> 
  fill(party_detailed, .direction = "downup") |> 
  ungroup() |> 
  distinct()


raw = read_parquet("data/colorado.parquet") |> 
  left_join(cands, join_by(race, candidate), relationship = "many-to-many")

xwalk = raw |> 
  distinct(state, county_name, cvr_id) |> 
  arrange(state, county_name) |> 
  bind_cols(
    fread("~/Downloads/index.csv", select=4),
    fread("outputs/batch_size512_latent_dims1_hidden_size64_emb_dim16_lr0.001_epochs20_n_samples1_voter_latents.csv")
  )

merged = raw |> 
  # mutate(id = cur_group_id(), .by = c(county_name, cvr_id)) |> 
  # filter(str_detect(race, "US PRESIDENT|US SENATE|US HOUSE")) |> 
  filter(race == "US PRESIDENT_FEDERAL") |> 
  mutate(
    topparty = case_when(
      party_detailed == "DEMOCRAT" ~ "Dem",
      party_detailed == "REPUBLICAN" ~ "Rep",
      .default = "Other"
    )
  ) |> 
  # distinct(state, county_name, cvr_id, topparty, id) |> 
  left_join(
    xwalk, join_by(state, county_name, cvr_id == cvr_id...3)
  ) |> 
  select(-cvr_id, cvr_id = cvr_id...4)

merged |> 
  filter(topparty != "Other") |>
  ggplot(aes(x = z0, fill = topparty)) +
  ggdist::stat_slab(alpha = 0.5) +
  theme_bw() +
  scale_fill_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E", "Other" = "grey50")
  )

merged |> 
  filter(topparty != "Other") |> 
  ggplot(aes(x = z0, y = z1, color = topparty)) +
  geom_density2d() +
  theme_bw() +
  scale_color_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E")
  )

ws = fread("outputs/batch_size512_latent_dims1_hidden_size64_emb_dim16_lr0.001_epochs20_n_samples1_item_parameters.csv")

ws[
  item_name == "US PRESIDENT_FEDERAL" & class_name %in% c("DONALD J TRUMP", "HOWIE HAWKINS", "JO JORGENSEN", "JOSEPH R BIDEN", "UNDERVOTE", "GLORIA LA RIVA"),
  list (class_name, bias, discrimination, difficulty, w_0, w_1, w_2, w_3)
  ]


s = fread("../cvr-ml/outputs/voter_scores.csv")

merged = raw |> 
  mutate(id = cur_group_id(), .by = c(county_name, cvr_id)) |> 
  filter(str_detect(race, "US PRESIDENT|US SENATE|US HOUSE")) |> 
  mutate(
    topparty = case_when(
      all(party_detailed == "DEMOCRAT") ~ "Dem",
      all(party_detailed == "REPUBLICAN") ~ "Rep",
      .default = "Other"
    ),
    .by = c(state, county_name, cvr_id)
  ) |> 
  distinct(state, county_name, cvr_id, topparty, id) |> 
  left_join(
    mutate(s, id = row_number()), join_by(id)
  )

merged |> 
  filter(topparty != "Other") |> 
  ggplot(aes(x = z0, color = topparty)) +
  geom_density() +
  theme_bw() +
  scale_color_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E")
  )

merged |> 
  summarize(
    p = mean(p_trump),
    .by = topparty
  )
