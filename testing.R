rm(list=ls())
gc()

library(tidyverse)
library(arrow)
library(data.table)
library(patchwork)
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
  nest(meta = -c(path, w)) |> 
  filter(str_detect(path, "Garfield"))

tictoc::tic()
pwalk(list(clean$path, clean$w, clean$meta),
  \(p, w, m) rcvr::clean_cvr(p, metadata = m, write_path = w)
)
tictoc::toc()

candsCO = open_dataset("../cvrs/data/pass3") |> 
  filter(state == "COLORADO") |> 
  distinct(state, office, district, candidate, party) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  select(state, race, candidate, party) |> 
  distinct() |> 
  collect()

cands = open_dataset("../cvrs/data/pass2") |> 
  filter(state != "COLORADO") |> 
  distinct(state, office, district, candidate, party_detailed) |> 
  mutate(
    race = paste(office, district, sep = "_")
  ) |> 
  distinct(state, race, candidate, party_detailed) |> 
  rename(party = party_detailed) |> 
  mutate(across(everything(), str_to_lower)) |> 
  collect() |> 
  bind_rows(candsCO)

raw = read_parquet("data/colorado_adams.parquet") |> 
  left_join(cands, join_by(state, race, candidate), relationship = "many-to-many")

latents = fread("outputs/datacolorado_adams.parquet_batch_size512_latent_dims1_hidden_size64_emb_dim16_voter_scores.csv")
discs = fread("outputs/datacolorado_adams.parquet_batch_size512_latent_dims1_hidden_size64_emb_dim16_item_parameters.csv")

merged = raw |> 
  mutate(
    housedist = str_extract(race, "\\d+")
  ) |> 
  filter(str_detect(race, "us president|^us house")) |> 
  group_by(state, county_name, cvr_id) |>
  fill(housedist, .direction = "downup") |>
  ungroup() |>
  filter(race == "us president_federal") |> 
  mutate(
    topparty = case_when(
      party == "democrat" ~ "Dem",
      party == "republican" ~ "Rep",
      candidate == "joseph r biden" ~ "Dem",
      candidate == "donald j trump" ~ "Rep",
      .default = "Other"
    )
  ) |> 
  left_join(
    latents, join_by(state, county_name, cvr_id)
  ) |> 
  left_join(
    fread("~/Dropbox (MIT)/Research/cvr-ideals/data/adams_voterlatents.csv", select = c(2,4,6)) |> mutate(county_name = "adams"),
    join_by(county_name, cvr_id)
  )

p_ideals = merged |>
  drop_na(mean) |>
  filter(topparty != "Other", race == "us president_federal") |>
  select(topparty, mu_0, mean) |>
  pivot_longer(cols = c(mu_0, mean), names_to = "method", values_to = "value") |>
  mutate(
    value = (value - mean(value, na.rm = TRUE)) / sd(value, na.rm = TRUE),
    .by = method
  ) |>
  mutate(
    method = case_match(
      method,
      "mu_0" ~ "VAE Estimates",
      "mean" ~ "MCMC Estimates"
    )
  ) |>
  ggplot(aes(x = value, fill = topparty)) +
  ggdist::stat_slab() +
  facet_wrap(~method, ncol = 1) +
  theme_bw(base_size = 18, base_family = "Rubik") +
  scale_discrete_manual(
    aesthetics = c("color", "fill"),
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E", "Other" = "grey50"),
    labels = c("Dem" = "Joseph R Biden", "Rep" = "Donald J Trump", "Other" = "Other")
  ) +
  labs(
    x = "Dimension 1 Ideal Point",
    y = "",
    fill = "Top-Ticket Vote"
  ) +
  theme(
    panel.background = element_blank(),
    panel.border = element_blank(),
    plot.background = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.y = element_blank(),
    panel.grid = element_blank(),
    axis.text.y = element_blank(),
    legend.box.background = element_rect(fill = NA, color = NA),
    legend.key = element_rect(fill = NA, color = NA, linewidth = 2),
    legend.position = "inside",
    legend.position.inside = c(1.15, -0.24),
    legend.direction = "horizontal",
    strip.background = element_blank(),
    strip.text = element_text(size = 20, hjust = 0, face = "italic")
  )

ggsave("figs/us_pres_latent_dim0_by_party.jpg", width = 6, height = 4, units = "in")

cor(merged$mu_0, merged$mean, use = "complete.obs")

compares = drop_na(merged, mean)

p_compare = compares |> 
  mutate(
    mu_0 = (mu_0 - mean(mu_0, na.rm=TRUE)) / sd(mu_0, na.rm=TRUE),
    mean = (mean - mean(mean, na.rm=TRUE)) / sd(mean, na.rm=TRUE)
  ) |> 
  filter(topparty != "Other") |> 
  ggplot(aes(x = mu_0, y = mean)) + 
  geom_point(alpha = 0.3, aes(color = topparty, fill = topparty)) +
  geom_smooth(se = FALSE, color = "black") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  theme_bw(base_size=18, base_family = "Rubik") +
  # scale_discrete_manual(
  #   aesthetics = c("color", "fill"),
  #   values = c("Dem" = "#3791FF", "Rep" = "#F6573E", "Other" = "grey50"),
  #   labels = c("Dem" = "Joseph R Biden", "Rep" = "Donald J Trump", "Other" = "Other")
  # ) +
  scale_color_manual(
    guide = "none",
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E", "Other" = "grey50"),
    labels = c("Dem" = "Joseph R Biden", "Rep" = "Donald J Trump", "Other" = "Other")
  ) +
  labs(
    x = "VAE Estimates",
    y = "MCMC Estimates",
    color = "Top-Ticket Vote"
  ) +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    panel.background  = element_blank(),
    plot.background = element_blank(),
    panel.border = element_blank(),
    axis.line.y.left = element_line(color = "black", linewidth = 0.5)
  )

p_ideals + p_compare &
  theme(
    plot.margin = margin(b=35),
    axis.line.x.bottom = element_line(color = "black", linewidth = 0.5)
  )

ggsave("figs/validation.jpg", width = 10, height = 6, units = "in")

merged |> 
  filter(topparty != "Other") |> 
  ggplot(aes(x = mu_0, y = mu_1, color = topparty)) +
  geom_density2d() +
  # facet_grid(~ topparty) +
  theme_bw() +
  scale_color_manual(
    values = c("Dem" = "#3791FF", "Rep" = "#F6573E")
  )

library(sf)

house_summaries <- merged |> 
  summarize(
    mean0 = mean(mu_0, na.rm=TRUE),
    # mean1 = mean(mu_1, na.rm=TRUE),
    .by = c(state, housedist)
  ) |> 
  # NA housedist are for people who didn't vote in the House contest
  drop_na(housedist) |> 
  mutate(
    housedist = str_pad(housedist, 2, "left", "0")
  )

boundaries = tigris::congressional_districts(year = 2020) |> 
  left_join(distinct(tigris::fips_codes, state_code, state_name), join_by("STATEFP" == "state_code")) |> 
  mutate(
    state_name = str_to_lower(state_name)
  ) |> 
  select(state = state_name, CD116FP, geometry) |> 
  left_join(house_summaries, join_by(state, CD116FP == housedist)) |> 
  filter(
    !(max(mean0, na.rm=TRUE) == -Inf),
    .by = state
  )

states = tigris::states(year = 2020) |> 
  mutate(state = str_to_lower(NAME)) |> 
  select(state, geometry)

st_plot <- function(d, st){

  p = d |> 
    ggplot(aes(fill = mean0)) +
    geom_sf(color = "white") +
    geom_sf(data = filter(states, state == st), color = "black", fill = NA) +
    theme_void() +
    scale_fill_gradient2(
      low = "#3791FF",
      mid = "white",
      high = "#F6573E",
      midpoint = -0.02794232,
      na.value = "grey80",
      limits = c(min(boundaries$mean0, na.rm=TRUE), max(boundaries$mean0, na.rm=TRUE)),
      guide = "none"
    )
  
  return(p)
}

ps = boundaries |> 
  nest(data = -state) |> 
  filter(!(state %in% c("michigan", "wisconsin", "delaware", "utah"))) |> 
  mutate(
    plot = map2(data, state, st_plot)
  )

gg_county_grid <- cowplot::plot_grid(
  plotlist = ps$plot,
  labels = str_to_title(ps$state),
  align = "n",
  label_size = 14,
  hjust = 0.5,
  vjust = 1.0,
  label_x = 0.5,
  label_fontfamily = "Rubik",
  label_fontface = "plain",
  greedy = FALSE,
  scale = 0.90,
  ncol = 5
) +
  theme(
    text = element_text(family = "Rubik")
  )

ggsave("figs/us_house_latent_dim0.png", plot=gg_county_grid, width = 10, height = 6, units = "in", dpi=300)

## COLORADO STATE HOUSE

boundaries <- read_sf("../cvr-ideals/data/co_sldl_2011_to_2021/co_sldl_2011_to_2021.shp") |> 
  select(District_1) |> 
  left_join(house_summaries, join_by("District_1" == "housedist")) 

denver = c(33, 29, 35, 31, 27, 24, 23, 28, 22, 1, 4, 5, 2, 3, 38, 43, 45, 44, 37, 40, 9, 6, 8, 41, 42, 32, 7, 30, 36)
boundary_denver = st_union(filter(boundaries, District_1 %in% denver))
boundary_state = st_union(boundaries)

p_state = ggplot(boundaries, aes(fill = mean0)) +
  geom_sf(color = "white") +
  # ggrepel::geom_label_repel(
  #   data = filter(boundaries, is.na(mean0)),
  #   aes(label = District_1, geometry = geometry),
  #   stat = "sf_coordinates"
  # ) +
  geom_sf(data = boundary_denver, color = NA, fill = "white") +
  geom_sf(data = boundary_state, color = "black", fill = NA) +
  theme_void() +
  scale_fill_gradient2(
    low = "#3791FF",
    mid = "white",
    high = "#F6573E",
    midpoint = -0.02794232,
    na.value = "grey80",
    limits = c(min(boundaries$mean0, na.rm=TRUE), max(boundaries$mean0, na.rm=TRUE))
  ) +
  labs(
    fill = "Average Ideal Point",
    title = "Mean Ideal Point in Colorado State House Districts"
  )

p_denver <- boundaries |>
  filter(District_1 %in% denver) |>
  ggplot(aes(fill = mean0)) +
  geom_sf(color = "white") +
  geom_sf(data = boundary_denver, color = "black", fill = NA) +
  theme_void() +
  scale_fill_gradient2(
    low = "#3791FF",
    mid = "white",
    high = "#F6573E",
    midpoint = -0.02794232,
    na.value = "grey80",
    limits = c(min(boundaries$mean0, na.rm=TRUE), max(boundaries$mean0, na.rm=TRUE))
  ) +
  labs(
    fill = "Average Ideal Point"
  )

p_state + p_denver + plot_layout(guides = "collect") & theme(
  legend.text = element_blank(),
  legend.ticks = element_blank(),
  legend.position = "bottom",
  legend.direction = "horizontal",
  text = element_text(family = "Rubik", size = 14)
)

ggsave("figs/colorado_house_latent_dim0_map.jpg", width = 10, height = 6, units = "in", dpi=300)

precs = read_csv("../precinct_project/2020-precincts/2020-co-precinct-general.csv") |> 
  filter(office == "STATE HOUSE") |> 
  summarize(
    v = sum(votes, na.rm=TRUE),
    .by = c(district, party_simplified)
  ) |> 
  mutate(
    total = sum(v, na.rm=TRUE),
    .by = district
  ) |> 
  mutate(p = v / total) |> 
  select(district, party_simplified, p) |>
  pivot_wider(names_from = party_simplified, values_from = p) |> 
  mutate(
    diff = REPUBLICAN - DEMOCRAT 
  ) |> 
  select(district, diff) |> 
  mutate(district = as.numeric(district)) |> 
  right_join(boundaries, by = join_by("district" == "District_1")) |> 
  st_as_sf() |> 
  mutate(
    same = (diff < 0 & mean0 < -0.02794232) | (diff > 0 & mean0 > -0.02794232)
  )

p_state_precs = ggplot(precs, aes(fill = diff)) +
  geom_sf(color = "white") +
  geom_sf(data = boundary_denver, color = NA, fill = "white") +
  geom_sf(data = boundary_state, color = "black", fill = NA) +
  ggrepel::geom_label_repel(
    data = filter(precs, !same, !(district %in% denver)),
    aes(label = district, geometry = geometry),
    stat = "sf_coordinates"
  ) +
  theme_void() +
  scale_fill_gradient2(
    low = "#3791FF",
    mid = "white",
    high = "#F6573E",
    midpoint = 0,
    na.value = "grey80",
    limits = c(-1, 1)
  ) +
  labs(
    fill = "Average Ideal Point",
    title = "Precinct-Level State House Results"
  ) +
  guides(fill = "none")

p_denver_precs = precs |>
  filter(district %in% denver) |>
  ggplot(aes(fill = diff)) +
  geom_sf(color = "white") +
  geom_sf(data = boundary_denver, color = "black", fill = NA) +
  ggrepel::geom_label_repel(
    data = filter(precs, !same, district %in% denver),
    aes(label = district, geometry = geometry),
    stat = "sf_coordinates"
  ) +
  theme_void() +
  scale_fill_gradient2(
    low = "#3791FF",
    mid = "white",
    high = "#F6573E",
    midpoint = 0,
    na.value = "grey80",
    limits = c(-1, 1)
  ) +
  labs(
    fill = "Average Ideal Point"
  ) +
  guides(fill = "none")

(p_state + p_denver) / (p_state_precs + p_denver_precs) + 
  plot_layout(guides = "collect") &
  theme(
    legend.text = element_blank(),
    legend.ticks = element_blank(),
    legend.position = "bottom",
    legend.direction = "horizontal",
    text = element_text(family = "Rubik", size = 14)
  )

ggsave("figs/colorado_house_latent_dim0_map.jpg", width = 10, height = 11, units = "in", dpi=300)

## propositions

read_csv("embedding_analysis/decoder_weights.csv") |> 
  filter(str_detect(race_name, "^prop_\\d+$|^prop_colorado (b|c|76|77)|^prop_ee"), candidate_name == "yes") |> 
  select(-candidate_name) |> 
  pivot_longer(cols = -c(race_name)) |> 
  mutate(
    name = case_match(
      name,
      "latent_dim_0" ~ "Latent 1",
      "latent_dim_1" ~ "Latent 2",
      "latent_dim_2" ~ "Latent 0",
      "latent_dim_3" ~ "Latent 3",
    ),
    race_name = paste("Prop", str_to_upper(str_remove_all(race_name, "prop_|colorado "))),
    race_desc = case_match(
      race_name, 
      "Prop EE" ~ "Prop EE: Tobacco Tax",
      "Prop C" ~ "Prop C: Gaming Licenses",
      "Prop B" ~ "Prop B: Increase Property Taxes",
      "Prop 77" ~ "Prop 77: Specific Gaming",
      "Prop 76" ~ "Prop 76: Only Citizen Voting",
      "Prop 118" ~ "Prop 118: Paid Leave",
      "Prop 117" ~ "Prop 117: Business Taxation",
      "Prop 116" ~ "Prop 116: Lower Income Tax",
      "Prop 115" ~ "Prop 115: Ban Abortion",
      "Prop 114" ~ "Prop 114: Gray Wolves",
      "Prop 113" ~ "Prop 113: Popular Vote"
    )
  ) |>  
  drop_na(race_desc) |> 
  ggplot(aes(x = value, y = race_desc)) +
  geom_point() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  facet_wrap(~ name, nrow = 1) +
  labs(
    x = "Discrimination Parameter",
    y = ""
  ) +
  scale_x_continuous(
    n.breaks = 3,
    limits = c(-1.7, 1.7),
    breaks = c(-1, 0, 1)
  ) +
  theme_bw(base_size = 16, base_family = "Rubik") +
  theme(
    panel.grid.major = element_line(linetype = "dashed", color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_line(linetype = "dashed", color = "grey90", linewidth = 0.3),
    strip.background = element_rect(fill = NA, color = NA),
    strip.text = element_text(face = "bold", size=18)
  )

ggsave("figs/prop4d_loadings.jpg", width = 10, height = 6, units = "in", dpi=300)

### uncontested comparisons

get_dist <- function(r, c) {

  precs = open_dataset("../cvrs/data/pass3") |> 
    mutate(race = paste(office, district, sep = "_")) |>
    filter(state == "COLORADO", race == r, candidate == c) |> 
    distinct(state, county_name, precinct)
  
  voters = open_dataset("../cvrs/data/pass3") |> 
    mutate(race = paste(office, district, sep = "_")) |>
    semi_join(precs, join_by(state, county_name, precinct)) |> 
    distinct(state, county_name, cvr_id) |> 
    mutate(state = str_to_lower(state), county_name = str_to_lower(county_name)) |> 
    collect()

  voters |> 
    left_join(latents, join_by(state, county_name, cvr_id)) |> 
    drop_na(mu_1) |> 
    pull(mu_1)

}

uncon_data = uncontested_co |> 
  filter(str_detect(race, "^district attorney")) |> 
  mutate(
    dist = map2(race, candidate, get_dist, .progress = TRUE)
  )

uncon_data |> 
  mutate(
    race = str_replace(race, "_", " ") |> str_to_title(),
    party = str_to_title(party)
  ) |> 
  filter(party != "Unafilliated") |> 
  slice_sample(n=6) |> 
  unnest(cols = dist) |> 
  ggplot(aes(x = dist, y = race, fill = party)) +
  ggdist::stat_halfeye(alpha = 0.8, color = "grey20") + 
  scale_fill_manual(
    values = c("Democrat" = "#3791FF", "Republican" = "#F6573E"),
  ) +
  labs(
    x = "Underlying Ideal Point Distribution",
    y = "",
    fill = "Candidate Party"
  ) +
  theme_bw(base_size = 18, base_family = "Rubik") +
  theme(
    panel.grid.major = element_line(linetype = "dashed", color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_line(linetype = "dashed", color = "grey90", linewidth = 0.3),
    strip.background = element_rect(fill = NA, color = NA),
    strip.text = element_text(face = "bold", size=18),
    legend.position = "bottom",
    legend.direction = "horizontal"
  )

ggsave("figs/uncontested_coda.jpg", width = 10, height = 6, units = "in", dpi=300)
ggsave("presentations/images/uncontested_coda.jpg", width = 10, height = 6, units = "in", dpi=300)
