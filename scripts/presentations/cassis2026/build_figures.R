#!/usr/bin/env Rscript

# Slide-specific figures for the CASSIS 2026 talk.
#
# Every visual is drawn with ggplot2 (cowplot is used only for composition).
# The script reads terminal BDB2020--2025 result tables and small, checksum-
# documented exports of genuine tracking plays. It never reads predictive
# score values from the Betty monitoring namespace.

if ("--help" %in% commandArgs(trailingOnly = TRUE)) {
  cat("Usage: Rscript scripts/presentations/cassis2026/build_figures.R [--only=figure_name[,figure_name...]]\n",
      "Rebuild presentation figures from retained exports and local study inputs.\n",
      "Existing export receipts are preserved; historical extraction code is not executed.\n", sep = "")
  quit(status = 0)
}

options(warn = 1)

suppressPackageStartupMessages({
  library(cowplot)
  library(data.table)
  library(digest)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(readr)
  library(scales)
  library(stringr)
  library(tidyr)
})

args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args_all, value = TRUE)
if (length(script_arg) != 1L) stop("Run this file with Rscript.")
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg)))
repo_dir <- normalizePath(file.path(script_dir, "../../.."))
deck_dir <- file.path(repo_dir, "presentations", "cassis2026")
figure_dir <- file.path(deck_dir, "figures")
example_dir <- file.path(deck_dir, "data", "examples")
inference_dir <- file.path(deck_dir, "data", "inference_90")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

# A partial build refreshes only the requested assets and their manifest entries.
# Example: Rscript scripts/presentations/cassis2026/build_figures.R --only=rushing_accuracy,manzone_results
figure_args <- commandArgs(trailingOnly = TRUE)
if (length(figure_args) > 1L ||
    (length(figure_args) == 1L && !grepl("^--only=.+", figure_args))) {
  stop("Optional argument: --only=figure_name[,figure_name...]")
}
requested_figures <- if (length(figure_args)) {
  unique(strsplit(sub("^--only=", "", figure_args), ",", fixed = TRUE)[[1]])
} else NULL

# Helvetica is metrically stable in both ggplot2's layout pass and ragg's
# renderer; the surrounding deck uses its bundled Roboto web fonts.
font_family <- "Helvetica"

ink <- "#222222"
muted <- "#686868"
light <- "#E7E7E7"
very_light <- "#F7F7F7"
cmu_red <- "#C41230"
blue <- "#35618F"
teal <- "#287271"
gold <- "#D4A72C"
purple <- "#855C9E"

# Captions and legends must remain readable when a figure occupies only part
# of a 16:9 slide. These sizes are used for both main and backup figures.
caption_text_size <- 19
legend_text_size <- 16

model_colors <- c(
  "Linear-Structure" = "#666666",
  "Boosted-Structure" = cmu_red,
  "Regularized regression" = "#666666",
  "Boosted trees" = cmu_red,
  "RelNet" = teal,
  "AttnRelNet" = gold,
  "Sumer Transformer" = blue,
  "L2 logistic" = "#666666",
  "LightGBM" = cmu_red,
  "Zoo CNN" = purple
)

theme_deck <- function(base_size = 23) {
  theme_minimal(base_size = base_size, base_family = font_family) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(family = font_family, face = "bold", color = ink),
      plot.subtitle = element_text(family = font_family, color = muted),
      plot.caption = element_text(family = font_family, color = muted, size = rel(0.62), hjust = 0),
      axis.title = element_text(family = font_family, color = ink),
      axis.text = element_text(family = font_family, color = ink),
      strip.text = element_text(family = font_family, face = "bold", color = ink),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "#E9E9E9", linewidth = 0.45),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(family = font_family, color = ink),
      plot.margin = margin(10, 20, 10, 20)
    )
}

generated_files <- character()
available_figures <- character()

save_figure <- function(plot, name, width = 14.2, height = 6.15, dpi = 190) {
  available_figures <<- c(available_figures, name)
  png_path <- file.path(figure_dir, paste0(name, ".png"))
  if (!is.null(requested_figures) && !name %in% requested_figures) {
    if (!file.exists(png_path)) stop("Partial build requires existing asset: ", png_path)
    return(invisible(png_path))
  }
  ggsave(
    png_path, plot = plot, width = width, height = height, units = "in",
    dpi = dpi, bg = "white", device = ragg::agg_png
  )
  generated_files <<- c(generated_files, png_path)
  invisible(png_path)
}

verify_sidecar <- function(path) {
  sidecar <- paste0(path, ".sha256")
  if (!file.exists(sidecar)) stop("Missing checksum sidecar: ", sidecar)
  expected <- strsplit(readLines(sidecar, warn = FALSE)[1], "[[:space:]]+")[[1]][1]
  actual <- digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
  if (!identical(tolower(expected), tolower(actual))) {
    stop("Checksum mismatch: ", path)
  }
  invisible(TRUE)
}

required_examples <- c(
  "bdb2025_game_2022090800_play_1563_snap.csv",
  "bdb2025_game_2022090800_play_2979_snap.csv",
  "bdb2025_man_zone_actual_play_metadata.csv",
  "bdb2025_man_zone_actual_plays_checksums.json",
  "bdb2025_man_zone_actual_plays_provenance.json",
  "bdb2020_chubb_handoff_tracking.csv",
  "bdb2020_chubb_predictions.csv",
  "bdb2020_chubb_example.json",
  "bdb2020_chubb_provenance.json"
)
missing_examples <- required_examples[!file.exists(file.path(example_dir, required_examples))]
if (length(missing_examples)) {
  stop("Missing real-play exports: ", paste(missing_examples, collapse = ", "))
}

for (name in c(
  "bdb2020_chubb_handoff_tracking.csv",
  "bdb2020_chubb_predictions.csv",
  "bdb2020_chubb_example.json",
  "bdb2020_chubb_provenance.json"
)) verify_sidecar(file.path(example_dir, name))

bdb2025_checksums <- fromJSON(
  file.path(example_dir, "bdb2025_man_zone_actual_plays_checksums.json"),
  simplifyVector = TRUE
)$sha256
# Validate the retained files consumed by the renderer. The same receipt also
# records the historical extraction script and its original source/config
# receipts; those documentary entries do not make archived code a dependency
# of rendering an already extracted example.
bdb2025_render_inputs <- paste0(
  "presentations/cassis2026/data/examples/",
  c("bdb2025_game_2022090800_play_1563_snap.csv",
    "bdb2025_game_2022090800_play_2979_snap.csv",
    "bdb2025_man_zone_actual_play_metadata.csv",
    "bdb2025_man_zone_actual_plays_provenance.json")
)
for (relative_path in bdb2025_render_inputs) {
  if (!relative_path %in% names(bdb2025_checksums)) {
    stop("Example receipt does not bind renderer input: ", relative_path)
  }
  absolute_path <- file.path(repo_dir, relative_path)
  if (!file.exists(absolute_path)) stop("Missing checksum-bound renderer input: ", relative_path)
  actual <- digest(absolute_path, algo = "sha256", file = TRUE, serialize = FALSE)
  if (!identical(tolower(unname(bdb2025_checksums[[relative_path]])), tolower(actual))) {
    stop("Checksum mismatch: ", relative_path)
  }
}

rushing_run_dir <- file.path(
  repo_dir, "data/bdb_suite_runs/definitive/bdb2020_rushing_harmonized/full100"
)
rushing_final_dir <- file.path(rushing_run_dir, "final")
rushing_manifest_path <- file.path(rushing_run_dir, "manifest.json")
rushing_primary_path <- file.path(rushing_final_dir, "primary_metrics.csv")
rushing_summary_path <- file.path(rushing_final_dir, "summary.csv")
rushing_coverage_path <- file.path(inference_dir, "bdb2020/coverage_lower_bounds.csv")
rushing_sharpness_path <- file.path(inference_dir, "bdb2020/controlled_sharpness.csv")
rushing_width_paired_path <- file.path(inference_dir, "bdb2020/interval_width_contrasts.csv")
rushing_supported_path <- file.path(inference_dir, "bdb2020/supported_best.csv")
rushing_paired_path <- file.path(inference_dir, "bdb2020/paired_contrasts.csv")
bdb2025_metrics_path <- file.path(
  repo_dir, "results/bdb/full100_serial_bdb2025_attempt1/final/primary_metrics.csv"
)
bdb_primary_paths <- c(
  bdb2020_rushing_harmonized = rushing_primary_path,
  bdb2021_completion = file.path(deck_dir, "data/raw_terminal/bdb2021/primary_metrics.csv"),
  bdb2022_punt_returns = file.path(deck_dir, "data/raw_terminal/bdb2022/primary_metrics.csv"),
  bdb2023_sack = file.path(repo_dir, "results/bdb/full100_serial_bdb2023_attempt1/final/primary_metrics.csv"),
  bdb2024_tackle = file.path(repo_dir, "results/bdb/full100_serial_bdb2024_attempt1/final/primary_metrics.csv"),
  bdb2025_man_zone = bdb2025_metrics_path
)
bdb_supported_paths <- c(
  bdb2020_rushing_harmonized = rushing_supported_path,
  bdb2021_completion = file.path(inference_dir, "bdb2021/supported_best.csv"),
  bdb2022_punt_returns = file.path(inference_dir, "bdb2022/supported_best.csv"),
  bdb2023_sack = file.path(inference_dir, "bdb2023/supported_best.csv"),
  bdb2024_tackle = file.path(inference_dir, "bdb2024/supported_best.csv"),
  bdb2025_man_zone = file.path(inference_dir, "bdb2025/supported_best.csv")
)
bdb_paired_paths <- c(
  bdb2020_rushing_harmonized = rushing_paired_path,
  bdb2021_completion = file.path(inference_dir, "bdb2021/paired_contrasts.csv"),
  bdb2022_punt_returns = file.path(inference_dir, "bdb2022/paired_contrasts.csv"),
  bdb2023_sack = file.path(inference_dir, "bdb2023/paired_contrasts.csv"),
  bdb2024_tackle = file.path(inference_dir, "bdb2024/paired_contrasts.csv"),
  bdb2025_man_zone = file.path(inference_dir, "bdb2025/paired_contrasts.csv")
)

for (path in c(
  rushing_manifest_path, rushing_primary_path, rushing_summary_path,
  rushing_coverage_path, rushing_sharpness_path, rushing_supported_path,
  rushing_paired_path, rushing_width_paired_path, unname(bdb_primary_paths),
  unname(bdb_supported_paths), unname(bdb_paired_paths)
)) verify_sidecar(path)

snap_man <- read_csv(
  file.path(example_dir, "bdb2025_game_2022090800_play_1563_snap.csv"),
  show_col_types = FALSE
)
snap_zone <- read_csv(
  file.path(example_dir, "bdb2025_game_2022090800_play_2979_snap.csv"),
  show_col_types = FALSE
)
play_meta <- read_csv(
  file.path(example_dir, "bdb2025_man_zone_actual_play_metadata.csv"),
  show_col_types = FALSE
)
players <- read_csv(file.path(repo_dir, "data/bdb2025/raw/players.csv"), show_col_types = FALSE)
snap_man <- snap_man %>% left_join(players %>% select(nflId, position), by = "nflId")
snap_zone <- snap_zone %>% left_join(players %>% select(nflId, position), by = "nflId")

field_plot <- function(dat, xmin, xmax, los = NULL, label_players = TRUE, panel_title = NULL) {
  dat <- dat %>%
    mutate(
      object = if_else(is.na(nflId) | club == "football" | displayName == "football", "Football", club),
      label = if_else(label_players & displayName %in% c("Josh Allen", "Stefon Diggs"),
                      displayName, "")
    )
  yard_lines <- seq(0, 120, by = 5)
  p <- ggplot() +
    annotate("rect", xmin = xmin, xmax = xmax, ymin = 5, ymax = 48,
             fill = "#F2F5ED", color = "#A5AA9C", linewidth = 0.8) +
    geom_vline(xintercept = yard_lines, color = "white", linewidth = 0.65) +
    geom_hline(yintercept = c(5, 48), color = "#74786D", linewidth = 0.8) +
    geom_point(
      data = dat %>% filter(object != "Football"),
      aes(x = x, y = y, fill = object),
      shape = 21, size = 5.0, stroke = 0.9, color = "white"
    ) +
    geom_point(
      data = dat %>% filter(object == "Football"),
      aes(x = x, y = y), shape = 23, size = 4.3, fill = ink, color = "white", stroke = 0.8
    ) +
    geom_text(
      data = dat %>% filter(label != ""),
      aes(x = x, y = y + if_else(displayName == "Josh Allen", -3.2, 3.2), label = label),
      family = font_family, size = 4.5, fontface = "bold", color = ink
    ) +
    scale_fill_manual(values = c("BUF" = "#00338D", "LA" = "#FFA300"), drop = FALSE) +
    coord_fixed(xlim = c(xmin, xmax), ylim = c(5, 48), expand = FALSE, clip = "off") +
    labs(title = panel_title) +
    theme_void(base_family = font_family) +
    theme(
      plot.background = element_rect(fill = "white", color = NA),
      plot.title = element_text(size = 17, face = "bold", color = ink, hjust = 0.5, margin = margin(b = 6)),
      legend.position = "none",
      plot.margin = margin(8, 8, 8, 8)
    )
  if (!is.null(los)) {
    p <- p + geom_vline(xintercept = los, color = blue, linewidth = 1.15, linetype = "22")
  }
  p
}

# 1. Raw training-release scale at three nested levels. BDB2021, BDB2022,
# and BDB2024 counts come from checksum-bound source receipts. BDB2020,
# BDB2023, and BDB2025 counts were verified from the corresponding local raw
# files. Task-specific eligibility filters used later only reduce these totals.
release_counts <- tribble(
  ~year,     ~observations, ~plays, ~games,
  "2020",       682154,  31007,    688,
  "2021",     18309388,  19239,    253,
  "2022",     36769985,  19979,    764,
  "2023",      8314178,   8557,    122,
  "2024",     12187398,  12486,    136,
  "2025",     59327373,  16124,    136
)

median_counts <- release_counts %>%
  summarise(
    year = "Median",
    observations = median(observations),
    plays = median(plays),
    games = median(games)
  )

sample_units_long <- bind_rows(release_counts, median_counts) %>%
  mutate(
    year = factor(year, levels = rev(c(as.character(2020:2025), "Median"))),
    y = c(7, 6, 5, 4, 3, 2, 0.65),
    is_median = year == "Median"
  ) %>%
  pivot_longer(
    cols = c(games, plays, observations),
    names_to = "level", values_to = "count"
  ) %>%
  mutate(
    level = factor(
      level,
      levels = c("games", "plays", "observations"),
      labels = c("Games", "Plays", "Tracking observations")
    ),
    label = case_when(
      is_median & level == "Games" ~ "195",
      level == "Tracking observations" & count >= 1e6 ~
        paste0(number(count / 1e6, accuracy = 0.1), "M"),
      level == "Tracking observations" ~
        paste0(number(count / 1e3, accuracy = 1), "K"),
      TRUE ~ number(count, accuracy = 1, big.mark = ",")
    )
  )

sample_unit_ranges <- sample_units_long %>%
  filter(!is_median) %>%
  group_by(year, y) %>%
  summarise(x = min(count), xend = max(count), .groups = "drop")

sample_units <- ggplot(sample_units_long, aes(x = count, y = y)) +
  geom_segment(
    data = sample_unit_ranges,
    aes(x = x, xend = xend, y = y, yend = y),
    inherit.aes = FALSE, color = light, linewidth = 3.3, lineend = "round"
  ) +
  geom_hline(yintercept = 1.34, color = "#D0D0D0", linewidth = 0.8) +
  geom_point(
    data = sample_units_long %>% filter(!is_median),
    aes(color = level), size = 6.2
  ) +
  geom_point(
    data = sample_units_long %>% filter(is_median),
    aes(fill = level), shape = 23, color = "white", stroke = 0.9, size = 7.2
  ) +
  geom_text(
    aes(x = count * 1.27, label = label, color = level),
    hjust = 0, family = font_family, fontface = "bold", size = 5.0,
    show.legend = FALSE
  ) +
  scale_color_manual(values = c(
    "Games" = cmu_red,
    "Plays" = blue,
    "Tracking observations" = "#555555"
  )) +
  scale_fill_manual(values = c(
    "Games" = cmu_red,
    "Plays" = blue,
    "Tracking observations" = "#555555"
  ), guide = "none") +
  scale_x_log10(
    limits = c(70, 1.7e8),
    breaks = c(1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8),
    labels = c("100", "1K", "10K", "100K", "1M", "10M", "100M")
  ) +
  scale_y_continuous(
    breaks = c(7, 6, 5, 4, 3, 2, 0.65),
    labels = c(as.character(2020:2025), "Median"),
    limits = c(0.25, 7.55), expand = expansion(mult = c(0, 0))
  ) +
  labs(
    x = "Count on a log scale", y = NULL, color = NULL,
    caption = "BDB2020 has one handoff snapshot per rushing play. Later releases include 10 Hz tracking."
  ) +
  theme_deck(21) +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 17, face = "bold"),
    axis.text.x = element_text(size = 13),
    axis.title.x = element_text(size = 16),
    legend.position = "top",
    legend.justification = "left",
    legend.text = element_text(size = legend_text_size),
    legend.key.width = unit(1.25, "lines"),
    plot.caption = element_text(size = caption_text_size, hjust = 0),
    plot.margin = margin(5, 56, 10, 26)
  )
save_figure(sample_units, "sample_units")

# 3. Trust framework: intentionally typographic, but still generated in ggplot2.
trust <- tibble(
  x = 1:3,
  heading = c("Accuracy", "Uncertainty", "Stability"),
  question = c(
    "Are the predictions good?",
    "Are they calibrated\nand informative?",
    "Would another plausible\nsample change the conclusion?"
  )
)
trust_framework <- ggplot(trust, aes(x = x, y = 1)) +
  geom_segment(aes(x = x - 0.34, xend = x + 0.34, y = 1.55, yend = 1.55),
               linewidth = 2.3, color = c(ink, cmu_red, ink)) +
  geom_text(aes(label = heading), y = 1.28, family = font_family,
            fontface = "bold", size = 11, color = ink) +
  geom_text(aes(label = question), y = 0.75, family = font_family,
            size = 6.6, lineheight = 1.05, color = muted) +
  coord_cartesian(xlim = c(0.48, 3.52), ylim = c(0.30, 1.72), clip = "off") +
  theme_void(base_family = font_family) +
  theme(plot.background = element_rect(fill = "white", color = NA))
save_figure(trust_framework, "trust_framework")

# BDB2020 real-play inputs, regenerated from the frozen repeat 5 / 360-game cell.
chubb_track <- read_csv(file.path(example_dir, "bdb2020_chubb_handoff_tracking.csv"), show_col_types = FALSE)
chubb_pred <- read_csv(file.path(example_dir, "bdb2020_chubb_predictions.csv"), show_col_types = FALSE)
chubb_example <- fromJSON(file.path(example_dir, "bdb2020_chubb_example.json"), simplifyVector = TRUE)

example_pred <- chubb_pred %>%
  filter(model_id == "lightgbm_multiclass") %>%
  mutate(across(
    c(is_observed_yard, in_central_interval, in_conformal_interval),
    ~ as.logical(as.integer(.x))
  ))
if (nrow(example_pred) != 80L) stop("Expected 80 LightGBM yard classes for the Chubb play.")
obs_y <- unique(example_pred$yards[example_pred$is_observed_yard])
if (length(obs_y) != 1L) stop("Expected one observed rushing-yard class.")
pred_mean <- sum(example_pred$yards * example_pred$probability)
play_crps <- sum(example_pred$crps_term)

# 3. The genuine Nick Chubb handoff snapshot.
chubb_plot_data <- chubb_track %>%
  mutate(
    plot_group = case_when(
      IsRusher ~ "Nick Chubb",
      role == "football" ~ "Football",
      role == "offense" ~ "Cleveland",
      TRUE ~ "New York Jets"
    )
  )
chubb_los <- unique(chubb_plot_data$line_of_scrimmage_x_std)
if (length(chubb_los) != 1L) stop("Expected one line of scrimmage for the Chubb play.")
chubb_distance <- as.integer(chubb_example$play$distance)
chubb_down <- as.integer(chubb_example$play$down)
chubb_down_label <- c("1st", "2nd", "3rd", "4th")[[chubb_down]]
chubb_field_xmin <- max(0, floor(chubb_los - 10))
chubb_field_xmax <- min(120, chubb_field_xmin + 39)
chubb_play_label <- paste0(
  chubb_example$play$away_team, " at ", chubb_example$play$home_team,
  ", ", chubb_example$play$season, " Week ", chubb_example$play$week,
  ", ", chubb_down_label, " & ", chubb_distance
)

# A simpler rendering of the same genuine play introduces what tracking adds:
# joint player locations and movement at one moment. The later rushing slide
# adds the football outcome and first-down context.
tracking_intro <- ggplot() +
  annotate("rect", xmin = chubb_field_xmin, xmax = chubb_field_xmax, ymin = 5, ymax = 48,
           fill = "#F2F5ED", color = "#A5AA9C", linewidth = 0.8) +
  geom_vline(xintercept = seq(0, 120, by = 5), color = "white", linewidth = 0.65) +
  geom_vline(xintercept = chubb_los, color = blue, linetype = "22", linewidth = 1.05) +
  geom_hline(yintercept = c(5, 48), color = "#74786D", linewidth = 0.8) +
  geom_segment(
    data = chubb_plot_data %>% filter(plot_group != "Football"),
    aes(
      x = X_std, y = Y_std,
      xend = X_std + 0.68 * Sx, yend = Y_std + 0.68 * Sy,
      color = plot_group
    ),
    linewidth = 0.8, alpha = 0.72,
    arrow = arrow(length = grid::unit(0.10, "in"), type = "closed")
  ) +
  geom_point(
    data = chubb_plot_data %>% filter(plot_group != "Football"),
    aes(x = X_std, y = Y_std, fill = plot_group),
    shape = 21, size = 5.2, stroke = 0.9, color = "white"
  ) +
  geom_text(
    data = chubb_plot_data %>% filter(IsRusher),
    aes(x = X_std, y = Y_std - 3.4), label = "Nick Chubb",
    family = font_family, fontface = "bold", color = ink, size = 4.7
  ) +
  annotate(
    "text", x = chubb_los + 0.45, y = 46.2, label = "line of scrimmage",
    hjust = 0, color = blue, family = font_family, size = 4.1
  ) +
  scale_fill_manual(
    name = NULL,
    values = c("Cleveland" = "#311D00", "New York Jets" = "#125740", "Nick Chubb" = "#FF3C00")
  ) +
  scale_color_manual(
    values = c("Cleveland" = "#311D00", "New York Jets" = "#125740", "Nick Chubb" = "#FF3C00")
  ) +
  guides(
    color = "none",
    fill = guide_legend(override.aes = list(size = 5.2, color = "white"), nrow = 1)
  ) +
  coord_fixed(
    xlim = c(chubb_field_xmin, chubb_field_xmax), ylim = c(5, 48),
    expand = FALSE, clip = "off"
  ) +
  theme_void(base_family = font_family) +
  theme(
    legend.position = "bottom",
    legend.text = element_text(family = font_family, size = 13, color = ink),
    legend.spacing.x = grid::unit(0.18, "in"),
    plot.background = element_rect(fill = "white", color = NA),
    plot.margin = margin(4, 14, 2, 14)
  )
save_figure(tracking_intro, "tracking_intro_chubb", width = 6.8, height = 6.9)

chubb_field <- ggplot() +
  annotate("rect", xmin = chubb_field_xmin, xmax = chubb_field_xmax, ymin = 5, ymax = 48,
           fill = "#F2F5ED", color = "#A5AA9C", linewidth = 0.8) +
  geom_vline(xintercept = seq(0, 120, by = 5), color = "white", linewidth = 0.65) +
  geom_vline(xintercept = chubb_los, color = blue, linetype = "22", linewidth = 1.1) +
  geom_vline(xintercept = chubb_los + chubb_distance, color = gold, linetype = "22", linewidth = 1.0) +
  geom_vline(xintercept = chubb_los + obs_y, color = cmu_red, linetype = "22", linewidth = 1.15) +
  geom_hline(yintercept = c(5, 48), color = "#74786D", linewidth = 0.8) +
  geom_point(
    data = chubb_plot_data %>% filter(plot_group != "Football"),
    aes(x = X_std, y = Y_std, fill = plot_group), shape = 21, size = 5.5, stroke = 0.9, color = "white"
  ) +
  geom_point(
    data = chubb_plot_data %>% filter(plot_group == "Football"),
    aes(x = X_std, y = Y_std), shape = 23, size = 4.3, fill = ink, color = "white", stroke = 0.8
  ) +
  geom_segment(
    data = chubb_plot_data %>% filter(IsRusher),
    aes(x = X_std, xend = X_std, y = Y_std - 5.6, yend = Y_std - 0.8),
    color = ink, linewidth = 0.65
  ) +
  geom_text(
    data = chubb_plot_data %>% filter(IsRusher),
    aes(x = X_std, y = Y_std - 6.6), label = "Nick Chubb", family = font_family,
    hjust = 0.5, fontface = "bold", color = ink, size = 5.0
  ) +
  annotate("text", x = chubb_los - 0.4, y = 46.3,
           label = "line of scrimmage", hjust = 1, color = blue, family = font_family, size = 4.3) +
  annotate("text", x = chubb_los + chubb_distance + 0.4, y = 43.3,
           label = "first down", hjust = 0, color = gold, family = font_family, size = 4.3) +
  annotate("text", x = chubb_los + obs_y + 0.4, y = 40.3,
           label = "end of run", hjust = 0, color = cmu_red,
           family = font_family, fontface = "bold", size = 4.3) +
  annotate("text", x = chubb_field_xmin + 0.8, y = 6.7,
           label = chubb_play_label,
           hjust = 0, color = muted, family = font_family, size = 4.5) +
  annotate("text", x = chubb_field_xmax - 0.8, y = 6.7,
           label = paste0("+", obs_y, " yards"), hjust = 1,
           color = cmu_red, family = font_family, fontface = "bold", size = 6.4) +
  scale_fill_manual(values = c(
    "Cleveland" = "#311D00", "New York Jets" = "#125740",
    "Nick Chubb" = "#FF3C00"
  )) +
  coord_fixed(xlim = c(chubb_field_xmin, chubb_field_xmax), ylim = c(5, 48), expand = FALSE, clip = "off") +
  theme_void(base_family = font_family) +
  theme(legend.position = "none", plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(4, 18, 4, 18))
save_figure(chubb_field, "rushing_actual_play", width = 7.0, height = 6.15)

# 5. The frozen LightGBM worked-example distribution for that play.
rushing_distribution <- ggplot(example_pred, aes(x = yards, y = probability)) +
  geom_col(width = 0.92, fill = "#C9D6E4") +
  geom_vline(xintercept = obs_y, color = cmu_red, linewidth = 1.4) +
  geom_vline(xintercept = pred_mean, color = blue, linewidth = 1.1, linetype = "22") +
  annotate("text", x = obs_y, y = max(example_pred$probability) * 1.05,
           label = paste0("observed  ", obs_y, " yards"), hjust = -0.06,
           family = font_family, fontface = "bold", color = cmu_red, size = 6.4) +
  annotate("text", x = pred_mean - 1.3, y = max(example_pred$probability) * 1.05,
           label = paste0("predictive mean  ", number(pred_mean, accuracy = 0.1)), hjust = 1,
           family = font_family, color = blue, size = 5.7) +
  annotate("text", x = 27, y = max(example_pred$probability) * 0.18,
           label = paste0("CRPS = ", number(play_crps, accuracy = 0.0001)), hjust = 1,
           family = font_family, fontface = "bold", color = cmu_red, size = 7.3) +
  coord_cartesian(xlim = c(-12, 28), ylim = c(0, max(example_pred$probability) * 1.18), clip = "off") +
  scale_y_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.03))) +
  labs(x = "Eventual rushing yards", y = "Predicted probability",
       caption = "LightGBM example, 360 training games.") +
  theme_deck(23) +
  theme(panel.grid.major.x = element_blank(),
        plot.caption = element_text(hjust = 0, size = caption_text_size))
save_figure(rushing_distribution, "rushing_distribution", width = 9.2, height = 6.15)

# 6. Actual-play CRPS: the area between the predicted and observed CDFs.
rushing_crps <- ggplot(example_pred, aes(x = yards)) +
  geom_ribbon(aes(ymin = pmin(predicted_cdf, observation_cdf),
                  ymax = pmax(predicted_cdf, observation_cdf)),
              fill = cmu_red, alpha = 0.18) +
  geom_step(aes(y = predicted_cdf), color = blue, linewidth = 1.4, direction = "hv") +
  geom_step(aes(y = observation_cdf), color = ink, linewidth = 1.05, direction = "hv") +
  geom_vline(xintercept = obs_y, color = cmu_red, linewidth = 1.1, linetype = "22") +
  annotate("text", x = -10, y = 0.88, label = "predicted CDF", hjust = 0,
           family = font_family, fontface = "bold", color = blue, size = 5.0) +
  annotate("segment", x = obs_y + 7, xend = obs_y + 0.5, y = 0.82, yend = 0.98,
           color = ink, linewidth = 0.7) +
  annotate("text", x = obs_y + 7.4, y = 0.80, label = "observed outcome", hjust = 0,
           family = font_family, fontface = "bold", color = ink, size = 5.0) +
  annotate("text", x = 24, y = 0.20,
           label = paste0("CRPS = ", number(play_crps, accuracy = 0.0001)),
           hjust = 1, family = font_family, fontface = "bold", color = cmu_red, size = 6.3) +
  annotate("text", x = 11, y = 0.42,
           label = "CRPS(F,y)==integral((F(z)-bold(1)(y<=z))^2*dz,-infinity,infinity)",
           parse = TRUE, hjust = 0, family = font_family, color = ink, size = 5.4) +
  coord_cartesian(xlim = c(-12, 28), ylim = c(0, 1), clip = "off") +
  scale_y_continuous(labels = label_percent(), breaks = c(0, .25, .5, .75, 1)) +
  labs(x = "Rushing yards", y = "Cumulative probability",
       caption = "CRPS integrates the shaded CDF gap across 80 yard outcomes. Lower is better.") +
  theme_deck(23) +
  theme(panel.grid.major.x = element_blank(),
        plot.caption = element_text(hjust = 0, size = caption_text_size))
save_figure(rushing_crps, "rushing_crps_actual")

# 7. Arithmetic illustration of padding the same actual play's central
# interval. The frozen example supplies a visible three-yard amount, but this
# graphic does not present it as the terminal harmonized calibration result.
padding_pred <- chubb_pred %>%
  filter(model_id == "lightgbm_multiclass") %>%
  mutate(across(
    c(is_observed_yard, in_central_interval, in_conformal_interval),
    ~ as.logical(as.integer(.x))
  ))
if (nrow(padding_pred) != 80L) stop("Expected 80 LightGBM yard classes for the Chubb play.")
padding_obs_y <- unique(padding_pred$yards[padding_pred$is_observed_yard])
if (length(padding_obs_y) != 1L) stop("Expected one observed rushing-yard class for the Chubb play.")
central_rng <- range(padding_pred$yards[padding_pred$in_central_interval], na.rm = TRUE)
padding_q <- as.integer(chubb_example$models$lightgbm_multiclass$conformal_padding_classes)
padded_rng <- c(central_rng[1] - padding_q, central_rng[2] + padding_q)
padded_label <- paste0("Example padding\n(+", padding_q, " yards)")
interval_df <- tibble(
  kind = factor(c("Model’s central interval", padded_label),
                levels = c(padded_label, "Model’s central interval")),
  lo = c(central_rng[1], padded_rng[1]),
  hi = c(central_rng[2], padded_rng[2]),
  y = c(2, 1),
  label_y = c(2.34, 1.34),
  color = c("#8D8D8D", cmu_red)
)
make_rushing_interval <- function(show_padding = TRUE) {
  draw_df <- if (show_padding) {
    interval_df
  } else {
    interval_df %>% filter(as.character(kind) == "Model’s central interval")
  }
  row_labels <- if (show_padding) {
    c(padded_label, "Model’s central interval")
  } else {
    c("", "Model’s central interval")
  }

  ggplot(draw_df, aes(y = y)) +
    geom_segment(aes(x = lo, xend = hi, yend = y, color = kind),
                 linewidth = 8, lineend = "butt") +
    geom_point(aes(x = lo, color = kind), size = 5) +
    geom_point(aes(x = hi, color = kind), size = 5) +
    annotate("point", x = padding_obs_y, y = 2, shape = 21, fill = "white",
             color = ink, stroke = 1.3, size = 6) +
    annotate("text", x = padding_obs_y, y = 1.60,
             label = paste0("actual: ", padding_obs_y, " yards"),
             family = font_family, fontface = "bold", color = ink, size = 5.5) +
    geom_text(aes(x = lo, y = label_y, label = lo, color = kind),
              family = font_family, fontface = "bold", size = 5.3) +
    geom_text(aes(x = hi, y = label_y, label = hi, color = kind),
              family = font_family, fontface = "bold", size = 5.3) +
    scale_color_manual(values = c(
      "Model’s central interval" = "#8D8D8D",
      setNames(cmu_red, padded_label)
    )) +
    scale_y_continuous(breaks = c(1, 2), labels = row_labels) +
    scale_x_continuous(breaks = seq(-20, 30, by = 5)) +
    coord_cartesian(
      xlim = c(min(padded_rng[1], -5) - 2, max(padded_rng[2], 16) + 2),
      ylim = c(-0.05, 2.55), clip = "off"
    ) +
    labs(
      title = "Example play: Nick Chubb vs. the Jets",
      x = "Rushing yards", y = NULL
    ) +
    theme_deck(30) +
    theme(
      panel.grid.major.y = element_blank(),
      axis.text.y = element_text(size = 22, face = "bold"),
      axis.text.x = element_text(size = 17),
      axis.title.x = element_text(size = 22),
      plot.title = element_text(size = 23, face = "bold", hjust = 0,
                                margin = margin(b = 18)),
      plot.title.position = "plot",
      legend.position = "none"
    )
}
rushing_interval_central <- make_rushing_interval(show_padding = FALSE)
rushing_interval <- make_rushing_interval(show_padding = TRUE)
save_figure(rushing_interval_central, "rushing_central_actual", width = 10.0, height = 5.0)
save_figure(rushing_interval, "rushing_conformal_actual", width = 10.0, height = 5.0)

# 8. Actual game assignments across repeated whole-study reruns.
splits <- fromJSON(rushing_manifest_path, simplifyVector = FALSE)$task_design$split_manifests
split_to_df <- function(item) {
  bind_rows(
    tibble(game = as.character(unlist(item$train_game_ids)), role = "Training pool"),
    tibble(game = as.character(unlist(item$calibration_game_ids)), role = "Calibration"),
    tibble(game = as.character(unlist(item$test_game_ids)), role = "Test")
  ) %>% mutate(rerun = item[["repeat"]])
}
split_df <- bind_rows(lapply(splits[1:20], split_to_df))
first_order <- split_df %>% filter(rerun == 1) %>%
  mutate(role_order = match(role, c("Training pool", "Tuning", "Calibration", "Test"))) %>%
  arrange(role_order, game) %>% pull(game)
split_df <- split_df %>% mutate(game_index = match(game, first_order))
repeated_splits <- ggplot(split_df, aes(x = game_index, y = factor(rerun), fill = role)) +
  geom_tile() +
  scale_fill_manual(values = c(
    "Training pool" = "#D9D9D9", "Calibration" = blue, "Test" = cmu_red
  ), breaks = c("Training pool", "Calibration", "Test")) +
  scale_y_discrete(limits = rev) +
  labs(x = "The same 648 confirmatory games, reassigned each time", y = "Repeated split",
       caption = paste(
         "First 20 of 100 repetitions shown. Models share each split.",
         "40 development games stay fixed."
       )) +
  theme_deck(21) +
  theme(
    panel.grid = element_blank(), axis.text.x = element_blank(), axis.ticks.x = element_blank(),
    axis.text.y = element_text(size = 11), axis.title.y = element_text(size = 16),
    plot.caption = element_text(hjust = 0, size = caption_text_size),
    legend.position = "bottom", legend.text = element_text(size = legend_text_size)
  )
save_figure(repeated_splits, "repeated_game_splits")

# 9. The two input representations used by the model classes.
input_field <- field_plot(snap_man, xmin = 22, xmax = 57, los = 38, label_players = FALSE)
input_text <- tibble(
  x = c(0, 0, 0, 0, 0, 0),
  y = c(5.7, 4.8, 3.7, 2.8, 1.7, 0.8),
  label = c(
    "INFORMATION THROUGH THE SNAP",
    "Tracking and permitted play context",
    "ENGINEERED FOOTBALL FEATURES",
    "positions · motion · distance to nearest opponent · game situation",
    "PLAYER TRACKING BY FRAME",
    "20 frames × 23 objects × 24 channels"
  ),
  kind = c("eyebrow", "body", "head", "body", "head_red", "body")
)
input_panel <- ggplot(input_text, aes(x = x, y = y, label = label)) +
  geom_text(
    aes(color = kind, fontface = if_else(kind %in% c("head", "head_red", "eyebrow"), "bold", "plain")),
    hjust = 0, family = font_family, size = ifelse(input_text$kind == "eyebrow", 4.3,
                                                   ifelse(grepl("head", input_text$kind), 6.5, 5.1))
  ) +
  annotate("segment", x = 0, xend = 0.92, y = 4.15, yend = 4.15, linewidth = 1.2, color = ink) +
  annotate("segment", x = 0, xend = 0.92, y = 2.15, yend = 2.15, linewidth = 1.2, color = cmu_red) +
  annotate("text", x = 0, y = -0.05,
           label = "The outcome and movement after the snap are withheld. Player IDs align slots only;\nprimary models exclude player and team identity.",
           hjust = 0, vjust = 0, family = font_family, color = muted, size = 4.2, lineheight = 1.05) +
  scale_color_manual(values = c("eyebrow" = muted, "body" = ink, "head" = ink, "head_red" = cmu_red)) +
  coord_cartesian(xlim = c(0, 1), ylim = c(-0.1, 6.0), clip = "off") +
  theme_void(base_family = font_family) +
  theme(legend.position = "none", plot.background = element_rect(fill = "white", color = NA))
model_inputs <- plot_grid(input_field, input_panel, nrow = 1, rel_widths = c(1.05, 1.15))
save_figure(model_inputs, "model_input_views")

# 10. Neural interactions on the same Nick Chubb handoff used in the worked example.
# The rushing contract permits every distinct pair of the 22 observed players.
# Show the common neighborhood once and compare the aggregation computations.
# All weights remain symbolic; no fitted attention scores are implied.
rushing_nodes <- chubb_plot_data %>% filter(role != "football")
rushing_focal <- rushing_nodes %>% filter(IsRusher)
stopifnot(nrow(rushing_nodes) == 22L, nrow(rushing_focal) == 1L)
rushing_neighbors <- rushing_nodes %>% filter(!IsRusher)
typed_edges <- rushing_neighbors %>%
  transmute(
    source_id = NflId,
    x = X_std, y = Y_std,
    xend = rushing_focal$X_std, yend = rushing_focal$Y_std,
    type = if_else(role == "offense", "Rusher–blocker", "Rusher–defender")
  )
stopifnot(
  nrow(typed_edges) == 21L,
  sum(typed_edges$type == "Rusher–blocker") == 10L,
  sum(typed_edges$type == "Rusher–defender") == 11L
)

rushing_graph_xlim <- range(rushing_nodes$X_std) + c(-4.5, 4)
rushing_graph_ylim <- range(rushing_nodes$Y_std) + c(-4, 4)
rushing_common_graph <- ggplot() +
  annotate("rect", xmin = rushing_graph_xlim[1], xmax = rushing_graph_xlim[2],
           ymin = rushing_graph_ylim[1], ymax = rushing_graph_ylim[2],
           fill = "#F2F5ED", color = "#A5AA9C") +
  geom_vline(xintercept = seq(0, 120, 5), color = "white", linewidth = 0.55) +
  geom_vline(xintercept = chubb_los, color = blue, linetype = "22", linewidth = 0.55) +
  geom_segment(data = typed_edges,
               aes(x = x, y = y, xend = xend, yend = yend),
               linewidth = 0.65, color = muted, alpha = 0.55) +
  geom_point(data = rushing_nodes, aes(x = X_std, y = Y_std, fill = plot_group),
             shape = 21, size = 4.8, color = "white", stroke = 0.8) +
  geom_point(data = rushing_focal, aes(x = X_std, y = Y_std),
             shape = 21, size = 6.5, fill = "#FF3C00", color = "white", stroke = 1.1) +
  geom_segment(data = rushing_focal,
               aes(x = X_std, xend = X_std, y = Y_std - 5.4, yend = Y_std - 1.0),
               color = ink, linewidth = 0.55) +
  geom_text(data = rushing_focal, aes(x = X_std, y = Y_std - 6.5),
            label = "Nick Chubb", family = font_family, fontface = "bold", color = ink, size = 4.5) +
  scale_fill_manual(values = c(
    "Cleveland" = "#311D00", "New York Jets" = "#125740", "Nick Chubb" = "#FF3C00"
  )) +
  coord_fixed(xlim = rushing_graph_xlim, ylim = rushing_graph_ylim, expand = FALSE) +
  labs(
    title = "Same handoff",
    caption = "Chubb's links to the other 21 players"
  ) +
  theme_void(base_family = font_family) +
  theme(
    plot.background = element_rect(fill = "white", color = NA), legend.position = "none",
    plot.title = element_text(family = font_family, face = "bold", hjust = 0.5, size = 18,
                              margin = margin(b = 9)),
    plot.caption = element_text(family = font_family, hjust = 0.5, color = muted, size = 13,
                                margin = margin(t = 9)),
    plot.margin = margin(10, 16, 12, 12)
  )

# These are single-player aggregation steps, not complete network diagrams.
# The third row shows one Transformer head before its output projection.
aggregation_rows <- tibble(
  model = c("RelNet", "AttnRelNet", "Sumer Transformer"),
  y = c(8.5, 5.3, 2.1),
  construction = c("Pairwise message", "Pairwise message", "Player value"),
  construction_math = c(
    "m[ij] == g[theta](h[i], h[j], e[ij])",
    "m[ij] == g[theta](h[i], h[j], e[ij])",
    "v[j] == W[V]*h[j] + b[V]"
  ),
  aggregation = c("Uniform mean", "Attention-weighted mean", "Attention-weighted mean"),
  aggregation_math = c(
    "frac(1,21)*sum(m[ij], j != i)",
    "sum(alpha[ij]*m[ij], j != i)",
    "sum(beta[ij]*v[j], j == 1, 22)"
  ),
  construction_detail = c(
    "Player pair + football relation type",
    "Player pair + football relation type",
    "No supplied relation type"
  ),
  weight_detail = c(
    "Each message gets weight 1/21",
    "Softmax of Chubb–message scores",
    "Query–key softmax, separately per head"
  )
)
rushing_aggregation <- ggplot() +
  geom_hline(yintercept = c(6.2, 3.0), color = light, linewidth = 0.65) +
  geom_text(data = aggregation_rows, aes(x = 0.3, y = y + 1.1, label = model),
            hjust = 0, family = font_family, fontface = "bold", color = ink, size = 6.0) +
  geom_text(data = aggregation_rows, aes(x = 2.7, y = y + 0.35, label = construction),
            family = font_family, color = muted, size = 4.6) +
  geom_text(data = aggregation_rows, aes(x = 2.7, y = y - 0.30, label = construction_math),
            parse = TRUE, family = font_family, color = ink, size = 5.8) +
  geom_text(data = aggregation_rows, aes(x = 2.7, y = y - 1.0, label = construction_detail),
            family = font_family, color = muted, size = 4.0) +
  geom_segment(data = aggregation_rows,
               aes(x = 5.25, xend = 6.05, y = y - 0.30, yend = y - 0.30),
               color = muted, linewidth = 0.7,
               arrow = arrow(length = grid::unit(0.10, "in"), type = "closed")) +
  geom_text(data = aggregation_rows, aes(x = 8.8, y = y + 0.35, label = aggregation),
            family = font_family, color = ink, size = 4.6) +
  geom_text(data = aggregation_rows, aes(x = 8.8, y = y - 0.35, label = aggregation_math),
            parse = TRUE, family = font_family, color = ink, size = 6.0) +
  geom_text(data = aggregation_rows, aes(x = 8.8, y = y - 1.10, label = weight_detail),
            family = font_family, color = muted, size = 4.0) +
  coord_cartesian(xlim = c(0, 11.9), ylim = c(0.45, 10.05), expand = FALSE, clip = "off") +
  theme_void(base_family = font_family) +
  theme(plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(10, 16, 12, 5))
neural_edges <- plot_grid(
  rushing_common_graph, rushing_aggregation, nrow = 1,
  rel_widths = c(0.80, 1.65)
)
save_figure(neural_edges, "neural_aggregation_rushing")

# 11. Six completed football problems, described by question and evaluation target.
tasks <- tibble(
  year = paste0("BDB", 2020:2025),
  question = c("Rushing yards", "Pass completion", "Punt return yards", "Sack", "Tackle candidate", "Man / Zone"),
  moment = c("Handoff", "Pass release", "Punt received", "Snap", "Candidate event\n−10 frames", "20 frames through snap"),
  output = c("80-class distribution", "Probability", "Ordered distribution", "Probability", "Candidate probability", "Probability"),
  score = c("CRPS", "Brier", "CRPS", "Brier", "Brier", "Brier"),
  y = rev(seq_len(6))
)
task_long <- tasks %>%
  select(year, question, moment, output, score, y) %>%
  pivot_longer(cols = c(year, question, moment, output, score), names_to = "column", values_to = "value") %>%
  mutate(
    x = recode(column, year = 0.45, question = 1.55, moment = 3.05, output = 4.70, score = 6.15),
    hjust = if_else(column == "year", 0.5, 0)
  )
headers <- tibble(
  x = c(0.45, 1.55, 3.05, 4.70, 6.15), y = 6.82,
  label = c("YEAR", "QUESTION", "PREDICTION MOMENT", "MODEL OUTPUT", "ACCURACY"),
  hjust = c(0.5, 0, 0, 0, 0)
)
task_scope <- ggplot() +
  geom_hline(yintercept = seq(0.5, 6.5, by = 1), color = "#DDDDDD", linewidth = 0.55) +
  geom_text(data = headers, aes(x = x, y = y, label = label, hjust = hjust),
            family = font_family, fontface = "bold", color = muted, size = 4.5) +
  geom_text(data = task_long, aes(x = x, y = y, label = value, hjust = hjust,
                                  color = column == "year"),
            family = font_family, fontface = ifelse(task_long$column == "year", "bold", "plain"),
            size = 5.1) +
  scale_color_manual(values = c("TRUE" = cmu_red, "FALSE" = ink)) +
  coord_cartesian(xlim = c(0, 7.0), ylim = c(0.45, 7.0), clip = "off") +
  theme_void(base_family = font_family) +
  theme(legend.position = "none", plot.background = element_rect(fill = "white", color = NA),
        plot.margin = margin(5, 18, 5, 18))
save_figure(task_scope, "completed_problem_scope")

# 12. BDB2020 accuracy learning curves from the terminal harmonized five-role run.
model_display <- c(
  linear_structure = "Regularized regression",
  boosted_structure = "Boosted trees",
  relnet = "RelNet",
  attn_relnet = "AttnRelNet",
  set_transformer = "Sumer Transformer"
)
model_order <- unname(model_display)

rush_summary_wide <- read_csv(rushing_summary_path, show_col_types = FALSE)
rush_summary <- bind_rows(
  rush_summary_wide %>% transmute(
    model, n_train, metric = "crps", mean = primary_loss_mean,
    q10 = primary_loss_q10, q90 = primary_loss_q90
  ),
  rush_summary_wide %>% transmute(
    model, n_train, metric = "coverage", mean = coverage_mean,
    q10 = coverage_q10, q90 = coverage_q90
  ),
  rush_summary_wide %>% transmute(
    model, n_train, metric = "mean_width", mean = interval_width_mean,
    q10 = interval_width_q10, q90 = interval_width_q90
  )
) %>%
  mutate(
    model_label = unname(model_display[model]),
    model_label = factor(model_label, levels = model_order)
  )
rush_crps <- rush_summary %>% filter(metric == "crps")
rush_supported <- read_csv(rushing_supported_path, show_col_types = FALSE) %>%
  filter(metric == "primary_loss", supported_best) %>%
  transmute(n_train, model_label = unname(model_display[model]))
label_offsets <- tibble(
  model_label = model_order,
  label_offset = c(0.00028, 0.00010, -0.00003, -0.00020, 0.00008)
)
label_crps <- rush_crps %>%
  filter(n_train == max(n_train)) %>%
  mutate(model_label = as.character(model_label)) %>%
  left_join(label_offsets, by = "model_label") %>%
  mutate(label_y = mean + label_offset)
rushing_accuracy <- ggplot(rush_crps, aes(x = n_train, y = mean, color = model_label, fill = model_label)) +
  geom_ribbon(
    aes(ymin = q10, ymax = q90, alpha = "10th–90th percentiles across reruns"),
    color = NA, show.legend = c(fill = FALSE, alpha = TRUE)
  ) +
  geom_line(linewidth = 1.25, show.legend = FALSE) +
  geom_point(size = 3.5, show.legend = FALSE) +
  geom_point(data = rush_crps %>% semi_join(rush_supported, by = c("n_train", "model_label")),
             aes(shape = "Clear winner (90% max-t)"), fill = "white",
             stroke = 1.1, size = 4.2,
             show.legend = c(color = FALSE, fill = FALSE, shape = TRUE)) +
  geom_text(data = label_crps, aes(y = label_y, label = model_label), hjust = 0, nudge_x = 16,
            family = font_family, fontface = "bold", size = 5.2, show.legend = FALSE) +
  scale_color_manual(values = model_colors, guide = "none") +
  scale_fill_manual(values = model_colors, guide = "none") +
  scale_alpha_manual(values = c("10th–90th percentiles across reruns" = 0.10)) +
  scale_shape_manual(values = c("Clear winner (90% max-t)" = 21)) +
  scale_x_continuous(breaks = c(20, 40, 80, 160, 240, 360), limits = c(15, 455)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(x = "Training games", y = "Mean CRPS  (lower is better)") +
  theme_deck(21) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom", legend.justification = "left",
    legend.box = "horizontal", legend.text = element_text(size = legend_text_size),
    legend.key.width = grid::unit(1.05, "cm"),
    legend.margin = margin(0, 0, 0, 0)
  ) +
  guides(
    alpha = guide_legend(
      order = 1,
      override.aes = list(fill = "#B8C4D0", color = NA, alpha = 0.45)
    ),
    shape = guide_legend(
      order = 2,
      override.aes = list(color = ink, fill = "white", size = 4.5, stroke = 1.1)
    )
  )
save_figure(rushing_accuracy, "rushing_accuracy")

# 13. Coverage and interval width at the largest rushing training sample size.
rush_360 <- rush_summary %>%
  filter(n_train == 360, metric %in% c("coverage", "mean_width")) %>%
  select(model_label, metric, mean, q10, q90) %>%
  pivot_wider(names_from = metric, values_from = c(mean, q10, q90))
rush_360 <- rush_360 %>%
  mutate(model_label = factor(
    model_label,
    levels = rev(model_order)
  ))

coverage_panel <- ggplot(rush_360, aes(x = mean_coverage, y = model_label, color = model_label)) +
  annotate("rect", xmin = 0.90, xmax = Inf, ymin = -Inf, ymax = Inf,
           fill = "#ECF5ED", color = NA) +
  geom_vline(xintercept = 0.90, linetype = "22", color = muted, linewidth = 0.8) +
  geom_segment(aes(x = q10_coverage, xend = q90_coverage, yend = model_label), linewidth = 1.15) +
  geom_point(size = 5) +
  annotate("text", x = 0.901, y = 5.48, label = "90% target", hjust = 0,
           family = font_family, color = muted, size = 4.2) +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(limits = c(.885, .955), labels = label_percent(accuracy = 1),
                     breaks = c(.89, .90, .91, .92, .93, .94, .95)) +
  labs(x = "Empirical coverage", y = NULL, subtitle = "Coverage across repeated splits") +
  theme_deck(20) +
  theme(
    legend.position = "none", panel.grid.major.y = element_blank(),
    axis.text.y = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 16, face = "bold", color = ink)
  )

width_panel <- ggplot(rush_360, aes(x = mean_mean_width, y = model_label, color = model_label)) +
  geom_segment(aes(x = q10_mean_width, xend = q90_mean_width, yend = model_label), linewidth = 1.15) +
  geom_point(size = 5) +
  geom_text(aes(label = number(mean_mean_width, accuracy = 0.01)), nudge_y = 0.22,
            family = font_family, fontface = "bold", size = 4.2, show.legend = FALSE) +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(limits = c(11.5, 20.2), breaks = seq(12, 20, 2)) +
  labs(x = "Mean interval width (yards)", y = NULL, subtitle = "Width across repeated splits") +
  theme_deck(20) +
  theme(
    legend.position = "none", panel.grid.major.y = element_blank(),
    axis.text.y = element_blank(), axis.ticks.y = element_blank(),
    plot.subtitle = element_text(size = 16, face = "bold", color = ink)
  )

rushing_sharpness_panels <- plot_grid(
  coverage_panel, width_panel, nrow = 1, rel_widths = c(1.08, 0.92), align = "h"
)
rushing_sharpness_caption <- ggdraw() +
  draw_label(
    "360 training games. Horizontal bars show 10th–90th percentiles across 100 repetitions.",
    x = 0.01, y = 0.55, hjust = 0, vjust = 0.5, fontfamily = font_family,
    size = caption_text_size, color = muted
  )
rushing_sharpness <- plot_grid(
  rushing_sharpness_panels, rushing_sharpness_caption,
  ncol = 1, rel_heights = c(1, 0.10)
)
save_figure(rushing_sharpness, "rushing_coverage_width")

# 14. Observed leaders and problem-wide max-t clear-winner decisions for the
# five recurring roles, BDB2020--2025.
task_display <- c(
  bdb2020_rushing_harmonized = "2020: Rushing yards",
  bdb2021_completion = "2021: Pass completion",
  bdb2022_punt_returns = "2022: Punt return yards",
  bdb2023_sack = "2023: Sack probability",
  bdb2024_tackle = "2024: Tackle candidate",
  bdb2025_man_zone = "2025: Man / Zone"
)
winner_df <- purrr::map_dfr(names(bdb_primary_paths), function(task_id_value) {
  metrics <- read_csv(bdb_primary_paths[[task_id_value]], show_col_types = FALSE) %>%
    filter(
      branch == "fixed_main",
      is.na(ablation_id) | ablation_id == "",
      is.na(sensitivity_id) | sensitivity_id == ""
    ) %>%
    group_by(n_train, model) %>%
    summarise(mean_loss = mean(primary_loss), .groups = "drop") %>%
    group_by(n_train) %>%
    slice_min(mean_loss, n = 1, with_ties = FALSE) %>%
    ungroup()
  decisions <- read_csv(bdb_supported_paths[[task_id_value]], show_col_types = FALSE) %>%
    filter(supported_best) %>%
    transmute(n_train, supported_model = model)
  metrics %>%
    left_join(decisions, by = "n_train") %>%
    transmute(
      task_id = task_id_value,
      task = unname(task_display[[task_id_value]]),
      n_train,
      leader = model,
      leader_label = unname(model_display[model]),
      supported_best = !is.na(supported_model) & model == supported_model
    )
}) %>%
  group_by(task) %>% arrange(n_train, .by_group = TRUE) %>%
  mutate(training_size_index = row_number()) %>% ungroup() %>%
  mutate(
    task = factor(task, levels = rev(unname(task_display))),
    text_color = if_else(leader == "attn_relnet", ink, "white")
)
winner_matrix <- ggplot(winner_df, aes(x = training_size_index, y = task, fill = leader_label)) +
  geom_tile(width = 0.88, height = 0.78, color = "white", linewidth = 1.2) +
  geom_tile(
    data = winner_df %>% filter(supported_best),
    aes(linetype = "Clear winner (90% max-t)"),
    width = 0.88, height = 0.78, fill = NA, color = ink, linewidth = 2.1,
    show.legend = c(fill = FALSE, linetype = TRUE)
  ) +
  geom_text(aes(label = n_train, color = text_color),
            family = font_family, fontface = "bold", size = 5.2, show.legend = FALSE) +
  scale_fill_manual(values = c(
    "Regularized regression" = model_colors[["Linear-Structure"]],
    "Boosted trees" = model_colors[["Boosted-Structure"]],
    "RelNet" = model_colors[["RelNet"]],
    "AttnRelNet" = model_colors[["AttnRelNet"]],
    "Sumer Transformer" = model_colors[["Sumer Transformer"]]
  ), breaks = model_order, limits = model_order) +
  scale_linetype_manual(values = c("Clear winner (90% max-t)" = "solid")) +
  scale_color_identity() +
  scale_x_continuous(breaks = NULL, expand = expansion(add = 0.55)) +
  labs(x = NULL, y = NULL, fill = NULL, linetype = NULL) +
  theme_deck(22) +
  theme(
    panel.grid = element_blank(), axis.text.y = element_text(size = 17, face = "bold"),
    axis.text.x = element_text(size = 14), axis.title.x = element_text(size = 17),
    legend.position = "bottom", legend.justification = "left",
    legend.box = "vertical", legend.box.just = "left",
    legend.spacing.y = grid::unit(8, "pt"),
    legend.text = element_text(size = legend_text_size),
    legend.key.width = grid::unit(0.8, "cm"),
    legend.margin = margin(0, 0, 0, 0)
  ) +
  guides(
    fill = guide_legend(order = 1, nrow = 1, byrow = TRUE),
    linetype = guide_legend(
      order = 2,
      override.aes = list(fill = NA, color = ink, linewidth = 1.7)
    )
  )
save_figure(winner_matrix, "supported_winner_matrix")

# One row from the ranking matrix, enlarged to explain the encoding before the
# audience sees all six problems together.
rushing_row <- winner_df %>%
  filter(task_id == "bdb2020_rushing_harmonized") %>%
  mutate(
    training_size_label = factor(n_train, levels = sort(unique(n_train))),
    leader_short = recode(
      leader,
      linear_structure = "Regularized\nregression",
      boosted_structure = "Boosted\ntrees",
      relnet = "RelNet",
      attn_relnet = "AttnRelNet",
      set_transformer = "Sumer\nTransformer"
    )
  )

rushing_row_plot <- ggplot(
  rushing_row,
  aes(x = training_size_label, y = 1, fill = leader_label)
) +
  geom_tile(width = 0.84, height = 0.56, color = "white", linewidth = 1.3) +
  geom_tile(
    data = rushing_row %>% filter(supported_best),
    width = 0.84, height = 0.56, fill = NA, color = ink, linewidth = 2.3,
    show.legend = FALSE
  ) +
  geom_text(
    aes(label = leader_short, color = text_color),
    family = font_family, fontface = "bold", size = 5.8, lineheight = 0.92,
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = c(
      "Regularized regression" = model_colors[["Linear-Structure"]],
      "Boosted trees" = model_colors[["Boosted-Structure"]],
      "RelNet" = model_colors[["RelNet"]],
      "AttnRelNet" = model_colors[["AttnRelNet"]],
      "Sumer Transformer" = model_colors[["Sumer Transformer"]]
    ),
    breaks = model_order, limits = model_order, drop = FALSE
  ) +
  scale_color_identity() +
  scale_y_continuous(NULL, breaks = NULL, limits = c(0.65, 1.35)) +
  labs(
    x = "Training games",
    fill = NULL,
    caption = "Dark outline: clear winner under the problem’s paired 90% max-t intervals."
  ) +
  theme_deck(23) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(size = 17, face = "bold"),
    axis.title.x = element_text(size = 17),
    legend.position = "none",
    plot.caption = element_text(size = caption_text_size, hjust = 0),
    plot.margin = margin(25, 50, 15, 50)
  )
save_figure(rushing_row_plot, "rushing_winner_row", height = 4.5)

# 15. Observed leaders versus problem-wide max-t clear winners.
winner_counts <- winner_df %>%
  group_by(leader_label) %>%
  summarise(observed = n(), supported = sum(supported_best), .groups = "drop") %>%
  complete(
    leader_label = unname(model_display),
    fill = list(observed = 0L, supported = 0L)
  ) %>%
  mutate(leader_label = factor(leader_label, levels = rev(unname(model_display))))

counts_plot <- ggplot(winner_counts, aes(y = leader_label)) +
  geom_segment(aes(x = supported, xend = observed, yend = leader_label),
               color = "#C9C9C9", linewidth = 2.2) +
  geom_point(aes(x = observed, color = "Lowest mean loss"), size = 5.5) +
  geom_point(aes(x = supported, color = "90% max-t clear winner"), size = 5.5) +
  geom_text(aes(x = observed, label = observed), nudge_y = 0.32,
            family = font_family, fontface = "bold", size = 4.5, color = ink) +
  geom_text(aes(x = supported, label = supported), nudge_y = -0.32,
            family = font_family, fontface = "bold", size = 4.5, color = cmu_red) +
  scale_color_manual(values = c("Lowest mean loss" = ink, "90% max-t clear winner" = cmu_red)) +
  scale_x_continuous(limits = c(-0.5, 16), breaks = c(0, 4, 8, 12, 16)) +
  labs(
    x = "Problem × training sample size combinations",
    y = NULL,
    color = NULL
  ) +
  theme_deck(21) +
  theme(
    panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 15, face = "bold"),
    legend.position = "top", legend.justification = "left",
    legend.text = element_text(size = legend_text_size),
    axis.title.x = element_text(size = 15.5),
    plot.caption = element_text(hjust = 0, size = caption_text_size, color = muted),
    plot.margin = margin(8, 20, 8, 8)
  )
supported_total <- sum(winner_df$supported_best)
matched_attention <- purrr::map_dfr(unname(bdb_paired_paths), function(path) {
  read_csv(path, show_col_types = FALSE) %>%
    filter(
      (model_left == "relnet" & model_right == "attn_relnet") |
        (model_left == "attn_relnet" & model_right == "relnet")
    )
})
comparison_cells <- nrow(winner_df)
expected_matched_attention <- comparison_cells
if (nrow(matched_attention) != expected_matched_attention) {
  stop("Expected ", expected_matched_attention, " RelNet–AttnRelNet comparisons.")
}
matched_attention_supported <- sum(
  matched_attention$simultaneous_lower > 0 | matched_attention$simultaneous_upper < 0
)
selection_summary_plot <- ggplot() +
  annotate("text", x = 0.5, y = 0.83,
           label = paste0(supported_total, " / ", comparison_cells), family = font_family,
           fontface = "bold", color = cmu_red, size = 14) +
  annotate("text", x = 0.5, y = 0.64,
           label = "problem × training sample size combinations\nhave a clear winner",
           family = font_family, color = ink, size = 5.0) +
  annotate("segment", x = 0.18, xend = 0.82, y = 0.49, yend = 0.49,
           linewidth = 0.8, color = light) +
  annotate("text", x = 0.5, y = 0.34,
           label = paste0(matched_attention_supported, " / ", nrow(matched_attention)), family = font_family,
           fontface = "bold", color = ink, size = 11) +
  annotate("text", x = 0.5, y = 0.19,
           label = "RelNet–AttnRelNet 90% intervals exclude zero",
           family = font_family, color = ink, size = 4.2) +
  annotate("text", x = 0.5, y = 0.05,
           label = "An inconclusive difference does not establish equivalence.",
           family = font_family, color = muted, size = 4.2) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  theme_void(base_family = font_family) +
  theme(plot.background = element_rect(fill = "white", color = NA))
model_summary <- plot_grid(counts_plot, selection_summary_plot, nrow = 1, rel_widths = c(1.42, 0.78))
save_figure(model_summary, "model_selection_summary")

# 16. Two genuine Man/Zone snaps plus the possible class-conditional sets.
man_panel <- field_plot(snap_man, xmin = 22, xmax = 57, los = 38, label_players = FALSE,
                        panel_title = "Man coverage: Cover 1")
zone_panel <- field_plot(snap_zone, xmin = 37, xmax = 72, los = 57, label_players = FALSE,
                         panel_title = "Zone coverage: Cover 6 Right")
manzone_fields <- plot_grid(man_panel, zone_panel, nrow = 1)
manzone_steps <- tibble(
  x = c(1, 2.55, 4.10),
  heading = c("Model score", "Calibrated probability", "Prediction set"),
  detail = c(
    "Tracking through the snap",
    "Venn–Abers  P(Man | X)",
    "{Man}   {Zone}   {Man, Zone}"
  )
)
manzone_sets <- ggplot(manzone_steps, aes(x = x, y = 0.5)) +
  geom_label(aes(label = heading), y = 0.72, family = font_family, fontface = "bold",
             size = 5.3, linewidth = 0, fill = "white", color = ink) +
  geom_text(aes(label = detail), y = 0.31, family = font_family, size = 4.5,
            color = c(muted, blue, cmu_red)) +
  annotate("segment", x = 1.56, xend = 1.95, y = 0.52, yend = 0.52,
           arrow = arrow(length = grid::unit(0.13, "inches")), color = muted, linewidth = 0.8) +
  annotate("segment", x = 3.11, xend = 3.50, y = 0.52, yend = 0.52,
           arrow = arrow(length = grid::unit(0.13, "inches")), color = muted, linewidth = 0.8) +
  coord_cartesian(xlim = c(0.35, 4.75), ylim = c(0.05, 0.95), clip = "off") +
  theme_void(base_family = font_family) +
  theme(plot.background = element_rect(fill = "white", color = NA))
manzone_examples <- plot_grid(manzone_fields, manzone_sets, ncol = 1, rel_heights = c(1, 0.28))
save_figure(manzone_examples, "manzone_actual_plays")

# 17. BDB2025 across every training sample size: accuracy and the stability of
# the model ranking across complete game-level reruns.
bdb2025 <- read_csv(bdb2025_metrics_path, show_col_types = FALSE) %>%
  filter(branch == "fixed_main",
         is.na(ablation_id) | ablation_id == "",
         is.na(sensitivity_id) | sensitivity_id == "") %>%
  mutate(model_label = recode(model,
    linear_structure = "Regularized regression", boosted_structure = "Boosted trees",
    relnet = "RelNet", attn_relnet = "AttnRelNet", set_transformer = "Sumer Transformer"
  )) %>%
  mutate(model_label = factor(model_label, levels = model_order))
manzone_summary <- bdb2025 %>%
  group_by(n_train, model_label) %>%
  summarise(
    brier = mean(primary_loss), brier_lo = quantile(primary_loss, 0.10),
    brier_hi = quantile(primary_loss, 0.90), .groups = "drop"
  )

manzone_rank_share <- bdb2025 %>%
  group_by(`repeat`, n_train) %>%
  mutate(repeat_rank = min_rank(primary_loss)) %>%
  ungroup() %>%
  mutate(repeat_best = repeat_rank == 1L) %>%
  group_by(n_train, model_label) %>%
  summarise(best_share = mean(repeat_best), .groups = "drop")

bdb2025_max_t <- read_csv(
  bdb_supported_paths[["bdb2025_man_zone"]], show_col_types = FALSE
)
if (!identical(
  bdb2025_max_t %>% filter(supported_best) %>% pull(n_train),
  c(20, 30, 40, 50)
)) stop("Unexpected BDB2025 problem-wide max-t result.")
manzone_supported <- bdb2025_max_t %>%
  filter(metric == "primary_loss", supported_best) %>%
  transmute(n_train, model_label = unname(model_display[model]))

brier_plot <- ggplot(
  manzone_summary,
  aes(x = n_train, y = brier, color = model_label, fill = model_label)
) +
  geom_ribbon(aes(ymin = brier_lo, ymax = brier_hi), alpha = 0.055,
              color = NA, show.legend = FALSE) +
  geom_line(linewidth = 1.15) +
  geom_point(size = 3.1) +
  geom_point(
    data = manzone_summary %>% semi_join(manzone_supported, by = c("n_train", "model_label")),
    shape = 21, fill = "white", stroke = 1.1, size = 4.2, show.legend = FALSE
  ) +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  scale_x_continuous(breaks = seq(10, 60, 10)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01)) +
  labs(
    x = "Training games", y = "Mean Brier score",
    subtitle = "Brier score across repeated game splits"
  ) +
  theme_deck(18) +
  theme(
    legend.position = "none", plot.subtitle = element_text(size = 13.5),
    axis.title = element_text(size = 15), axis.text = element_text(size = 12),
    plot.margin = margin(8, 14, 6, 8)
  )

rank_plot <- ggplot(
  manzone_rank_share,
  aes(x = n_train, y = best_share, color = model_label)
) +
  geom_line(linewidth = 1.15) +
  geom_point(size = 3.1) +
  scale_color_manual(values = model_colors) +
  scale_x_continuous(breaks = seq(10, 60, 10)) +
  scale_y_continuous(
    labels = label_percent(accuracy = 1), limits = c(0, 0.75),
    breaks = c(0, 0.25, 0.50, 0.75), expand = expansion(mult = c(0, 0.03))
  ) +
  labs(
    x = "Training games", y = "Reruns with lowest Brier score",
    subtitle = "Frequency of ranking first across repeated game splits"
  ) +
  theme_deck(18) +
  theme(
    legend.position = "none", plot.subtitle = element_text(size = 13.5),
    axis.title = element_text(size = 15), axis.text = element_text(size = 12),
    plot.margin = margin(8, 8, 6, 14)
  )

manzone_legend <- get_legend(
  brier_plot +
    guides(color = guide_legend(nrow = 1), fill = "none") +
    theme(
      legend.position = "bottom", legend.text = element_text(size = legend_text_size),
      legend.key.width = grid::unit(1.15, "cm")
    )
)
manzone_note <- get_legend(
  rushing_accuracy +
    theme(legend.margin = margin(0, 14, 0, 14))
)
manzone_results <- plot_grid(
  plot_grid(brier_plot, rank_plot, nrow = 1, rel_widths = c(1.18, 1)),
  manzone_legend,
  manzone_note,
  ncol = 1,
  rel_heights = c(1, 0.11, 0.09)
)
save_figure(manzone_results, "manzone_results")

# Backup: all three rushing outcomes across every training sample size.
make_rush_panel <- function(metric_name, y_lab, percent = FALSE) {
  d <- rush_summary %>% filter(metric == metric_name)
  p <- ggplot(d, aes(x = n_train, y = mean, color = model_label, fill = model_label)) +
    geom_ribbon(aes(ymin = q10, ymax = q90), alpha = 0.08, color = NA) +
    geom_line(linewidth = 1.0) + geom_point(size = 2.6) +
    scale_color_manual(values = model_colors) + scale_fill_manual(values = model_colors) +
    scale_x_continuous(breaks = c(20, 80, 160, 240, 360)) +
    labs(x = "Training games", y = y_lab) + theme_deck(17) +
    theme(legend.position = "none", axis.text = element_text(size = 11.5),
          axis.title = element_text(size = 13), plot.margin = margin(4, 6, 4, 6))
  if (percent) p <- p + scale_y_continuous(labels = label_percent(accuracy = 1))
  if (metric_name == "crps") p <- p +
    geom_point(
      data = d %>% semi_join(rush_supported, by = c("n_train", "model_label")),
      shape = 21, fill = "white", stroke = 1.1, size = 4.2, show.legend = FALSE
    )
  p
}
full_rushing_panels <- list(
  make_rush_panel("crps", "Mean CRPS"),
  make_rush_panel("coverage", "Empirical coverage", percent = TRUE) +
    geom_hline(yintercept = .90, linetype = "22"),
  make_rush_panel("mean_width", "Mean interval width (yards)")
)
full_rushing_legend <- get_legend(
  full_rushing_panels[[1]] +
    guides(color = guide_legend(nrow = 1), fill = "none") +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = legend_text_size),
      legend.key.width = grid::unit(1.25, "cm")
    )
)
full_rushing_note <- get_legend(
  rushing_accuracy +
    geom_hline(aes(yintercept = 0.90, linetype = "90% coverage target"),
               color = ink, linewidth = 0.8) +
    scale_linetype_manual(values = c("90% coverage target" = "22")) +
    guides(
      linetype = guide_legend(order = 3, override.aes = list(color = ink))
    ) +
    theme(legend.margin = margin(0, 14, 0, 14))
)
full_rushing <- plot_grid(
  plot_grid(plotlist = full_rushing_panels, nrow = 1),
  full_rushing_legend,
  full_rushing_note,
  ncol = 1,
  rel_heights = c(1, 0.11, 0.10)
)
save_figure(full_rushing, "backup_full_rushing")

# Backup: an example of the simultaneous max-t decision rule (BDB2021, 10 games).
bdb2021_pairs <- read_csv(
  bdb_paired_paths[["bdb2021_completion"]], show_col_types = FALSE
) %>%
  filter(n_train == 10, model_left == "linear_structure") %>%
  mutate(
    comparator = recode(model_right,
      boosted_structure = "Boosted trees", relnet = "RelNet",
      attn_relnet = "AttnRelNet", set_transformer = "Sumer Transformer"
    ),
    comparator = factor(comparator, levels = rev(c(
      "Boosted trees", "RelNet", "AttnRelNet", "Sumer Transformer"
    )))
  )
simultaneous_example <- ggplot(bdb2021_pairs, aes(x = mean_difference, y = comparator)) +
  geom_vline(xintercept = 0, color = muted, linetype = "22") +
  geom_segment(aes(x = simultaneous_lower, xend = simultaneous_upper, yend = comparator,
                   color = "90% max-t interval"), linewidth = 1.2) +
  geom_point(color = cmu_red, size = 4.6, show.legend = FALSE) +
  scale_color_manual(values = c("90% max-t interval" = cmu_red)) +
  annotate("text", x = min(bdb2021_pairs$simultaneous_lower), y = 4.55,
           label = "Regularized regression − comparator", hjust = 0,
           family = font_family, color = muted, size = 4.5) +
  labs(x = "Mean paired Brier difference", y = NULL, color = NULL) +
  theme_deck(22) +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 17, face = "bold"),
        legend.position = "bottom", legend.justification = "left",
        legend.text = element_text(size = legend_text_size),
        legend.key.width = grid::unit(1.05, "cm"),
        legend.margin = margin(0, 0, 0, 0))
save_figure(simultaneous_example, "backup_simultaneous_inference")

# Reproducible figure manifest.
unknown_figures <- setdiff(requested_figures, available_figures)
if (length(unknown_figures)) {
  stop("Unknown figure names: ", paste(unknown_figures, collapse = ", "))
}
output_files <- sort(unique(generated_files))
manifest <- lapply(output_files, function(path) {
  list(
    file = basename(path),
    bytes = unname(file.info(path)$size),
    sha256 = digest(path, algo = "sha256", file = TRUE, serialize = FALSE),
    generated_by = "scripts/presentations/cassis2026/build_figures.R / ggplot2"
  )
})
manifest_path <- file.path(figure_dir, "manifest.json")
if (!is.null(requested_figures)) {
  if (!file.exists(manifest_path)) stop("Partial build requires an existing figure manifest.")
  previous_manifest <- fromJSON(manifest_path, simplifyVector = FALSE)$figures
  retained_manifest <- Filter(function(entry) {
    !entry$file %in% basename(output_files)
  }, previous_manifest)
  manifest <- c(manifest, retained_manifest)
  manifest <- manifest[order(vapply(manifest, `[[`, character(1), "file"))]
}
write_json(
  list(schema_version = "cassis2026-ggplot-figures-v1", figures = manifest),
  manifest_path, auto_unbox = TRUE, pretty = TRUE
)

message("Generated ", length(output_files), " ggplot2 assets in ", figure_dir)
