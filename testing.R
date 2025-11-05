rm(list=ls())
gc()

library(tidyverse)
library(arrow)
library(data.table)
devtools::load_all("~/Documents/github/rcvr/")

co_origmeta = googlesheets4::read_sheet(
  "https://docs.google.com/spreadsheets/d/12_OxXiSCrg6kO-R2bOXCzDxqvXmZr20-BhFsN52yetM/edit?gid=505279773#gid=505279773",
  "contests",
  col_types = "c"
) |> 
  filter(state == "COLORADO") |> 
  select(-order) |> 
  mutate(county_name = replace_na(county_name, "")) |>
  drop_na(contest) |>
  mutate(across(-contest, str_to_upper)) |>
  mutate(
    district = ifelse(office %in% c("STATE HOUSE", "STATE SENATE", "US HOUSE"),
      str_pad(district, width = 3, side = "left", pad = "0"),
      district
    ),
    district = str_replace(district, fixed("COUNTY_NAME"), county_name),
    district = str_replace(district, fixed("STATEWIDE"), state),
    contest = iconv(contest, from = "ascii", to = "UTF-8", sub = "")
  ) |> 
  mutate(across(-c(state, county_name), str_to_lower)) |> 
  mutate(
    contest = str_remove(contest, candidate) |> str_squish()
  )

co_files = googlesheets4::read_sheet(
  "https://docs.google.com/spreadsheets/d/12_OxXiSCrg6kO-R2bOXCzDxqvXmZr20-BhFsN52yetM/edit?gid=505279773#gid=505279773",
  "paths"
) |> 
  filter(state == "COLORADO", is.na(build)) |> 
  mutate(path = paste0("../cvrs/", path)) |> 
  select(path, state, county_name)

co_meta = co_files |> 
  mutate(
    m = map(path, \(p) rcvr::clean_cvr(p, generate_metadata = TRUE, metadata_only = TRUE))
  ) |> 
  select(-path) |> 
  unnest(cols = m) |> 
  select(state, county_name, raw_contest = contest, raw_candidate, raw_party, party, magnitude) |> 
  mutate(
    contest = janitor::make_clean_names(raw_contest, case = "title", allow_dupes = TRUE) |> str_to_lower(),
    candidate = str_to_lower(raw_candidate) |> str_remove_all(fixed('\"')) |> str_remove_all(fixed(".")) |> str_squish()
  ) |> 
  left_join(
    co_origmeta,
    join_by(state, county_name, contest, candidate)
  ) |>
  left_join(
    select(co_origmeta, state, county_name, contest, office, district, magnitude),
    join_by(state, county_name, contest)
  ) |> 
  mutate(
    party = coalesce(party_detailed, party, raw_party),
    magnitude = coalesce(magnitude.y, as.character(magnitude.x), magnitude),
    office = coalesce(office.x, office.y),
    district = coalesce(district.x, district.y)
  ) |> 
  select(
    state, county_name, raw_contest, raw_candidate, raw_party, office, district, candidate, party, magnitude
  ) |> 
  distinct()

co_meta_clean = googlesheets4::read_sheet(
  "https://docs.google.com/spreadsheets/d/1Pq9sNcCfLVi-qeXfBy7xEi5lPxpMJYy3LVUHQn_uEFI/edit?gid=1550619837#gid=1550619837",
  "metadata",
  col_types = "c"
) |>
  filter(election == "2020 General", state == "COLORADO") |> 
  mutate(
    across(office:prop_text, str_to_lower),
    district = str_replace(district, "county_name", str_to_lower(county)),
    district = str_replace(district, "statewide", str_to_lower(state)),
    magnitude = as.integer(magnitude)
  ) |>
  group_by(raw_candidate) |>
  fill(party) |>
  ungroup() |> 
  select(-type:-prop_text, -election) |> 
  distinct(state, county, contest, raw_candidate, office, district, candidate, party, magnitude)

clean = co_files |> 
  left_join(
    co_meta_clean, 
    join_by(state, county_name == county)
  ) |> 
  mutate(
    w = glue::glue("../cvrs/data/pass3/state={state}/county_name={county_name}/part-0.parquet")
  ) |> 
  nest(meta = -c(path, w))

pwalk(list(clean$path, clean$w, clean$meta),
  \(p, w, m) rcvr::clean_cvr(p, metadata = m, write_path = w)
)

######
# create test file for VAE implementation
######

small_cands = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
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

uncontested = open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
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

open_dataset("~/Dropbox (MIT)/Research/cvrs/data/pass3") |>
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
  anti_join(small_cands, join_by(race, candidate)) |> 
  anti_join(uncontested, join_by(race, candidate)) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/colorado2.parquet")

d |> 
  semi_join(
    distinct(d, county_name, cvr_id) |> slice_sample(n=1000, by = county_name),
    join_by(county_name, cvr_id)
  ) |> 
  write_parquet("~/Dropbox (MIT)/Research/cvr-ml/data/colorado_sample.parquet")

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
  left_join(
    fread("~/Dropbox (MIT)/Research/cvr-ideals/data/adams_voterlatents.csv", select = c(2,4,6)),
    join_by(county_name, cvr_id)
  ) |> 
  select(-cvr_id, cvr_id = cvr_id...4)

merged |> 
  filter(topparty != "Other") |>
  ggplot(aes(x = z0, fill = topparty)) +
  ggdist::stat_slab(alpha = 0.8) +
  theme_bw() +
  scale_fill_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E", "Other" = "grey50"),
    labels = c("Dem" = "Joseph R Biden", "Rep" = "Donald J Trump", "Other" = "Other")
  ) +
  labs(
    x = "Dim 0",
    y = "",
    fill = ""
  ) +
  theme(
    axis.text.x = element_text(size=12),
    axis.title.x = element_text(size=12),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank(),
    panel.grid = element_blank(),
    axis.text.y = element_blank(),
    panel.background  = element_blank(),
    panel.border = element_blank(),
    plot.background = element_blank(),
    axis.line.x.bottom = element_line(color = "black"),
    legend.box.background = element_rect(fill=NA, color=NA),
    legend.key = element_rect(fill=NA, color=NA, linewidth = 2),
    legend.text = element_text(color = "black", size = 12),
    legend.title = element_text(color = "black", size = 14),
    legend.position = "bottom",
    legend.direction = "horizontal",
  )

ggsave("figs/colorado_pres_latent_dim0_by_party.jpg", width = 6, height = 4, units = "in")

merged |> 
  filter(topparty != "Other") |>
  ggplot(aes(x = z1, fill = topparty)) +
  ggdist::stat_slab(alpha = 0.5) +
  theme_bw() +
  scale_fill_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E", "Other" = "grey50")
  )

merged |> 
  drop_na(mean) |> 
  mutate(
    mean_scaled = as.numeric(scale(mean)),
    z0_scaled = as.numeric(scale(z0)),
    # z1_scaled = as.numeric(scale(z1))
  ) |> 
  filter(topparty != "Other") |> 
  ggplot(aes(x = z0_scaled, y = mean_scaled)) + 
  geom_point(alpha = 0.2) +
  theme_bw() +
  labs(
    x = "VAE Estimates (Scaled)",
    y = "IRT Estimates (Scaled)"
  ) +
  coord_equal(clip = "on")

ggsave("figs/vae_vs_irt_colorado_dim1.jpg", width = 6, height = 6, units = "in")

merged |> 
  filter(topparty != "Other") |> 
  mutate(
    z0 = -1*z0
  ) |> 
  ggplot(aes(x = z0, y = z1, color = topparty)) +
  geom_density2d() +
  facet_grid(~ topparty) +
  theme_bw() +
  scale_color_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E")
  )

ws = fread("outputs/batch_size512_latent_dims2_hidden_size64_emb_dim16_lr0.001_epochs20_n_samples1_item_parameters.csv")

ws[
  item_name == "US PRESIDENT_FEDERAL" & class_name %in% c("DONALD J TRUMP", "HOWIE HAWKINS", "JO JORGENSEN", "JOSEPH R BIDEN", "UNDERVOTE", "GLORIA LA RIVA"),
  list (class_name, bias, discrimination, difficulty, w_0, w_1)
  ]
