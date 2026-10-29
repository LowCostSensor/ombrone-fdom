# Run the staRdom (eemR, multiway) equivalents of the eemkit steps on the tables
# written by scripts/benchmark_export.py, for the Phase 5 benchmark.
#
# Usage:
#   Rscript scripts/benchmark_stardom.R corrections <input dir> <output dir>
#   Rscript scripts/benchmark_stardom.R parafac <input dir> <output dir> <rank> <nstart> <maxit> <cores>
#
# 'corrections': inner filter, blank, Raman units (in eemkit's order and in the
# staRdom tutorial order), scatter mask, and the indices of eemR computed on
# eemkit's own Raman-normalised EEMs (so only the index code differs).
# 'parafac': eem_parafac on eemkit's final corrected EEMs (same input matrix).

suppressMessages(suppressWarnings({
  library(staRdom)
  library(dplyr)
}))

new_eem <- getFromNamespace("eem", "eemR")

read_eems <- function(path) {
  d <- read.csv(path, check.names = FALSE)
  ex <- as.numeric(colnames(d)[-(1:2)])
  samples <- unique(d$sample)
  lst <- lapply(samples, function(s) {
    rows <- d[d$sample == s, ]
    x <- as.matrix(rows[, -(1:2)])
    dimnames(x) <- NULL
    new_eem(list(file = path, sample = s, x = x, ex = ex, em = rows$em, location = ""))
  })
  class(lst) <- "eemlist"
  lst
}

write_eems <- function(eems, path) {
  out <- bind_rows(lapply(eems, function(e) {
    df <- as.data.frame(e$x)
    colnames(df) <- format(e$ex, trim = TRUE)
    cbind(sample = e$sample, em = e$em, df)
  }))
  write.csv(out, path, row.names = FALSE, na = "NA")
}

as_eemlist <- function(lst) {
  class(lst) <- "eemlist"
  lst
}

# eem_remove_blank subtracts one blank from every sample; ours are paired
remove_paired_blank <- function(eems, blanks) {
  as_eemlist(lapply(seq_along(eems), function(i) {
    eem_remove_blank(as_eemlist(list(eems[[i]])), as_eemlist(list(blanks[[i]])))[[1]]
  }))
}

raman_units <- function(eems, area) {
  eem_raman_normalisation2(eems, blank = data.frame(raman_area = area$raman_area,
                                                     row.names = make.names(area$sample)))
}

run_corrections <- function(inp, out) {
  raw <- read_eems(file.path(inp, "raw_eem.csv"))
  blk <- read_eems(file.path(inp, "blank_eem.csv"))
  abs_data <- read.csv(file.path(inp, "absorbance.csv"), check.names = FALSE)
  stopifnot(identical(eem_names(raw), eem_names(blk)))

  # Raman area of each sample's own blank (staRdom: ex 350, em 371-428)
  area <- eem_raman_area(blk, blanks_only = FALSE)
  area$sample <- eem_names(raw)
  write.csv(area[c("sample", "raman_area")], file.path(out, "raman_area.csv"), row.names = FALSE)

  # eemkit order: IFE on the sample, then blank, then Raman units
  invisible(capture.output(ife <- eem_ife_correction(raw, abs_data, cuvl = 1)))
  a1 <- raman_units(remove_paired_blank(ife, blk), area)
  write_eems(a1, file.path(out, "after_raman_eemkit_order.csv"))
  # staRdom tutorial order: blank, then IFE, then Raman units
  invisible(capture.output(b2 <- eem_ife_correction(remove_paired_blank(raw, blk), abs_data,
                                                    cuvl = 1)))
  write_eems(raman_units(b2, area), file.path(out, "after_raman_stardom_order.csv"))

  # scatter: widths in staRdom's order raman1, raman2, rayleigh1, rayleigh2
  sc <- eem_rem_scat(as_eemlist(a1[1]), remove_scatter = c(TRUE, TRUE, TRUE, TRUE),
                     remove_scatter_width = c(8, 10, 10, 12), cores = 1)
  mask <- data.frame(em = sc[[1]]$em, is.na(sc[[1]]$x) & !is.na(a1[[1]]$x))
  colnames(mask) <- c("em", format(sc[[1]]$ex, trim = TRUE))
  write.csv(mask, file.path(out, "scatter_mask.csv"), row.names = FALSE)

  # indices on eemkit's Raman-normalised EEMs
  ek <- read_eems(file.path(inp, "eemkit_after_raman.csv"))
  idx <- eem_fluorescence_index(ek, verbose = FALSE) %>%
    left_join(eem_humification_index(ek, scale = TRUE, verbose = FALSE), by = "sample") %>%
    left_join(eem_biological_index(ek, verbose = FALSE), by = "sample") %>%
    left_join(eem_coble_peaks(ek, verbose = FALSE), by = "sample")
  write.csv(idx, file.path(out, "indices.csv"), row.names = FALSE)
  cat("corrections done:", length(raw), "samples\n")
}

run_parafac <- function(inp, out, rank, nstart, maxit, cores) {
  cor <- read_eems(file.path(inp, "eemkit_corrected.csv"))
  t0 <- Sys.time()
  pf <- eem_parafac(cor, comps = rank, normalise = TRUE, nstart = nstart, maxit = maxit,
                    ctol = 1e-8, cores = cores, output = "best")
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  m <- norm2A(pf[[1]])
  cc <- eempf_corcondia(m, cor)
  fmax <- sweep(m$A, 2, apply(m$B, 2, max) * apply(m$C, 2, max), `*`)
  write.csv(data.frame(em = cor[[1]]$em, m$B), file.path(out, "pf_em_loadings.csv"),
            row.names = FALSE)
  write.csv(data.frame(ex = cor[[1]]$ex, m$C), file.path(out, "pf_ex_loadings.csv"),
            row.names = FALSE)
  write.csv(data.frame(sample = eem_names(cor), fmax), file.path(out, "pf_fmax.csv"),
            row.names = FALSE)
  info <- data.frame(rank = rank, nstart = nstart, maxit = maxit, ctol = 1e-8,
                     rsq = pf[[1]]$Rsq, cflag = pf[[1]]$cflag, iter = pf[[1]]$iter,
                     converged_fraction = pf[[1]]$converged, corcondia = cc, seconds = secs,
                     stardom = as.character(packageVersion("staRdom")),
                     multiway = as.character(packageVersion("multiway")))
  write.csv(info, file.path(out, "pf_info.csv"), row.names = FALSE)
  print(info)
}

args <- commandArgs(trailingOnly = TRUE)
dir.create(args[3], showWarnings = FALSE, recursive = TRUE)
if (args[1] == "corrections") {
  run_corrections(args[2], args[3])
} else if (args[1] == "parafac") {
  run_parafac(args[2], args[3], as.integer(args[4]), as.integer(args[5]),
              as.integer(args[6]), as.integer(args[7]))
} else {
  stop("first argument must be 'corrections' or 'parafac'")
}
