"""Phase 0 inventory of the Ombrone fDOM/CDOM dataset.

Read-only scan of Cary Eclipse EEM CSVs and Cary 60 absorbance CSVs.
Nothing in the data folder is written, renamed or moved.

Outputs (in --out):
    eem_files.csv       one row per fluorescence CSV, with parsed metadata and QC
    abs_files.csv       one row per absorbance CSV, with parsed metadata and QC
    ignored_files.csv   every non-CSV file found (binary .FDSW/.BSW, odd extensions)
    pairing_draft.csv   draft EEM <-> absorbance <-> blank pairing, to be checked by hand
    inventory_report.md human-readable summary

Usage:
    python scripts/inventory.py --data-root "<...>/Progetto PARAFAC" --out inventory
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

FLUO_SUBDIR = Path("fDOM_ombrone/fDOM_ombrone")
ABS_SUBDIR = Path("CDOM ombrone/CDOM ombrone")

# The user stated all campaigns are from 2024; folder names without a year
# (e.g. "FARMA DICEMBRE") are assigned to this year and flagged in the report.
DEFAULT_YEAR = 2024

IT_MONTHS = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6,
    "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}
IT_MONTH_ABBR = {
    "gen": 1, "feb": 2, "mar": 3, "apr": 4, "mag": 5, "giu": 6,
    "lug": 7, "ago": 8, "set": 9, "ott": 10, "nov": 11, "dic": 12,
}

EEM_LABEL_RE = re.compile(r"^(?P<name>.*)_EX_(?P<ex>\d+(?:\.\d+)?)$")
ABS_LABEL_RE = re.compile(r"^(?P<name>.*)_(?P<cycle>\d+)$")


# --------------------------------------------------------------------------- #
# Generic CSV helpers
# --------------------------------------------------------------------------- #

def read_lines(path: Path) -> tuple[list[str], str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc).splitlines(), enc
        except UnicodeDecodeError:
            continue
    raise RuntimeError("unreachable")


def split_sections(lines: list[str]) -> tuple[str, str, list[str], list[str]]:
    """Return (label row, unit row, data rows, log rows)."""
    data = []
    i = 2
    while i < len(lines) and lines[i].strip() != "":
        data.append(lines[i])
        i += 1
    return lines[0], lines[1] if len(lines) > 1 else "", data, lines[i + 1:]


def header_labels(line: str) -> list[str]:
    return [t.strip() for t in line.split(",") if t.strip()]


def parse_numeric(rows: list[str], ncols: int) -> tuple[np.ndarray, int]:
    """Parse comma-separated rows into a float array; return (array, n_bad_cells)."""
    arr = np.full((len(rows), ncols), np.nan)
    bad = 0
    for r, row in enumerate(rows):
        toks = row.split(",")
        for c in range(min(ncols, len(toks))):
            t = toks[c].strip()
            if not t:
                continue
            try:
                arr[r, c] = float(t)
            except ValueError:
                bad += 1
    return arr, bad


def parse_log(log: list[str]) -> list[dict]:
    """Split a Cary log section into one dict per scan.

    A scan starts at a 'Collection Time:' line; the preceding non-empty line is
    its label. Keys are read only from the 'Parameter List' section, where key
    and value are separated by two or more spaces.
    """
    scans: list[dict] = []
    current: dict | None = None
    in_params = False
    prev = ""
    for line in log:
        s = line.rstrip()
        if s.startswith("Collection Time:"):
            current = {"_label": prev.strip().rstrip(","),
                       "_collection_time": s.split(":", 1)[1].strip()}
            scans.append(current)
            in_params = False
        elif current is not None:
            if s.startswith("Parameter List"):
                in_params = True
            elif s.startswith("Method Log"):
                in_params = False
            elif s.startswith("Method Name"):
                current["_method_name"] = s.split(":", 1)[1].strip()
            elif in_params:
                parts = re.split(r"\s{2,}", s.strip())
                if len(parts) >= 2:
                    current[" ".join(parts[:-1])] = parts[-1].strip()
        if s.strip():
            prev = s
    return scans


def parse_cary_time(text: str | None) -> datetime | None:
    if not text:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def summarise_param(scans: list[dict], key: str) -> tuple[str, bool]:
    """Return ('v1' or 'v1|v2', inconsistent?) for a parameter across scans."""
    vals = [sc.get(key, "") for sc in scans]
    uniq = list(dict.fromkeys(vals))
    return "|".join(uniq), len(uniq) > 1


def step_stats(x: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(np.diff(x[~np.isnan(x)]))
    if d.size == 0:
        return (np.nan, np.nan, np.nan)
    return float(np.median(d)), float(d.min()), float(d.max())


# --------------------------------------------------------------------------- #
# Sample-name interpretation (project-specific, Phase 0 only)
# --------------------------------------------------------------------------- #

def parse_sample_name(name: str) -> dict:
    """Interpret a sample name as river/site/replicate/variant.

    Returns kind = 'blank' | 'sample' | 'unparsed'. Nothing is discarded:
    the trailing suffix (month token for Orcia, ME/MO/N for Merse/Farma) is kept.
    """
    low = re.sub(r"\s+", " ", name.strip().lower())
    m = re.fullmatch(r"(blank|bianco) ?([a-z]*)", low)
    if m:
        return dict(kind="blank", river="", site="", rep="", variant="", suffix=m.group(2))
    m = re.fullmatch(r"orcia ?(1bis|1b|\d)(\d) ?([a-z]*)", low)
    if m:
        site = "1bis" if m.group(1) in ("1b", "1bis") else m.group(1)
        return dict(kind="sample", river="Orcia", site=site, rep=m.group(2),
                    variant="", suffix=m.group(3))
    for river in ("farma", "merse"):
        m = re.fullmatch(rf"{river} ?(\d)(\d) ?([a-z]*)", low)
        if m:
            return dict(kind="sample", river=river.capitalize(), site=m.group(1),
                        rep=m.group(2), variant=m.group(3).upper(), suffix="")
    river = next((r.capitalize() for r in ("orcia", "farma", "merse") if low.startswith(r)), "")
    return dict(kind="unparsed", river=river, site="", rep="", variant="", suffix="")


def folder_date(folder: str) -> datetime | None:
    for fmt in ("%Y%m%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(folder, fmt)
        except ValueError:
            continue
    return None


def folder_month(folder: str) -> int | None:
    d = folder_date(folder)
    if d:
        return d.month
    for word in folder.lower().split():
        if word in IT_MONTHS:
            return IT_MONTHS[word]
    return None


# --------------------------------------------------------------------------- #
# Per-file parsers
# --------------------------------------------------------------------------- #

EEM_PARAMS = {
    "instrument": "Instrument",
    "serial": "Instrument Serial Number",
    "data_mode": "Data mode",
    "scan_mode": "Scan mode",
    "em_start": "Start (nm)",
    "em_stop": "Stop (nm)",
    "ex_slit": "Ex. Slit (nm)",
    "em_slit": "Em. Slit (nm)",
    "scan_rate": "Scan rate (nm/min)",
    "data_interval": "Data interval (nm)",
    "averaging_time": "Averaging Time (s)",
    "ex_filter": "Excitation filter",
    "em_filter": "Emission filter",
    "pmt": "PMT voltage (V)",
    "corrected": "Corrected spectra",
}

ABS_PARAMS = {
    "instrument": "Instrument",
    "start": "Start (nm)",
    "stop": "Stop (nm)",
    "y_mode": "Y Mode",
    "scan_rate": "UV-Vis Scan Rate (nm/min)",
    "data_interval": "UV-Vis Data Interval (nm)",
    "averaging_time": "UV-Vis Ave. Time (sec)",
    "beam_mode": "Beam Mode",
    "baseline_correction": "Baseline Correction",
    "cycle_count": "Cycle Count",
}


def inspect_eem(path: Path) -> dict:
    lines, enc = read_lines(path)
    label_row, unit_row, data_rows, log = split_sections(lines)
    labels = header_labels(label_row)
    rec: dict = {"encoding": enc, "n_labels": len(labels), "issues": []}

    names, exs = [], []
    for lab in labels:
        m = EEM_LABEL_RE.match(lab)
        if m:
            names.append(m.group("name"))
            exs.append(float(m.group("ex")))
        else:
            rec["issues"].append(f"label not '<name>_EX_<nm>': {lab!r}")
    uniq_names = list(dict.fromkeys(names))
    rec["name_in_file"] = "|".join(uniq_names)
    if len(uniq_names) > 1:
        rec["issues"].append("several sample names in header")
    if not unit_row.startswith("Wavelength (nm),Intensity"):
        rec["issues"].append("unexpected unit row")

    ex = np.array(exs)
    rec["n_ex"] = len(ex)
    rec["ex_min"] = ex.min() if ex.size else np.nan
    rec["ex_max"] = ex.max() if ex.size else np.nan
    rec["ex_step"] = step_stats(ex)[0] if ex.size > 1 else np.nan
    if ex.size and len(set(exs)) != len(exs):
        rec["issues"].append("duplicate excitation wavelengths")

    arr, bad = parse_numeric(data_rows, 2 * len(labels))
    wl, inten = arr[:, 0::2], arr[:, 1::2]
    em = wl[:, 0]
    rec["n_em"] = len(em)
    rec["em_min"] = np.nanmin(em) if em.size else np.nan
    rec["em_max"] = np.nanmax(em) if em.size else np.nan
    rec["em_step_median"], rec["em_step_min"], rec["em_step_max"] = step_stats(em)
    same = np.allclose(wl, em[:, None], equal_nan=True, atol=1e-6) if wl.size else False
    rec["em_identical_across_ex"] = bool(same)
    if not same:
        rec["issues"].append("emission axis differs between excitations")
    rec["n_bad_cells"] = bad
    rec["n_missing_intensity"] = int(np.isnan(inten).sum())
    finite = inten[np.isfinite(inten)]
    rec["frac_negative"] = float((finite < 0).mean()) if finite.size else np.nan
    rec["max_intensity"] = float(finite.max()) if finite.size else np.nan
    if finite.size:
        i, j = np.unravel_index(np.nanargmax(inten), inten.shape)
        rec["max_at_ex_em"] = f"{ex[j]:.0f}/{em[i]:.1f}"

    rec["data_hash"] = hashlib.sha1(np.round(inten, 5).tobytes()).hexdigest()

    # In 3-D mode every log block reports 'Ex. Wavelength' = 3-D start (200 nm),
    # so the excitation of each scan is taken from the header labels only.
    scans = parse_log(log)
    rec["n_scans_logged"] = len(scans)
    if len(scans) != len(labels):
        rec["issues"].append(f"{len(scans)} log blocks for {len(labels)} scans")
    for col, key in EEM_PARAMS.items():
        val, inconsistent = summarise_param(scans, key)
        rec[col] = val
        if inconsistent:
            rec["issues"].append(f"'{key}' varies between scans")
    times = [t for t in (parse_cary_time(sc["_collection_time"]) for sc in scans) if t]
    rec["first_scan"] = min(times) if times else None
    rec["last_scan"] = max(times) if times else None
    rec["method_name"] = "|".join(dict.fromkeys(sc.get("_method_name", "") for sc in scans))
    return rec


def inspect_abs(path: Path) -> dict:
    lines, enc = read_lines(path)
    label_row, unit_row, data_rows, log = split_sections(lines)
    labels = header_labels(label_row)
    rec: dict = {"encoding": enc, "issues": []}

    names, cycles = [], []
    for lab in labels:
        m = ABS_LABEL_RE.match(lab)
        if m:
            names.append(m.group("name"))
            cycles.append(int(m.group("cycle")))
        else:
            rec["issues"].append(f"label not '<name>_<cycle>': {lab!r}")
    uniq_names = list(dict.fromkeys(names))
    rec["name_in_file"] = "|".join(uniq_names)
    if len(uniq_names) > 1:
        rec["issues"].append("several sample names in header")
    rec["n_cycles"] = len(labels)
    if len(labels) != 3:
        rec["issues"].append(f"{len(labels)} cycles instead of 3")
    if not unit_row.startswith("Wavelength (nm),Abs"):
        rec["issues"].append("unexpected unit row")

    arr, bad = parse_numeric(data_rows, 2 * len(labels))
    wl, absorb = arr[:, 0::2], arr[:, 1::2]
    w = wl[:, 0]
    rec["n_wl"] = len(w)
    rec["wl_min"], rec["wl_max"] = np.nanmin(w), np.nanmax(w)
    rec["wl_descending"] = bool(np.all(np.diff(w) < 0))
    rec["wl_step_median"], rec["wl_step_min"], rec["wl_step_max"] = step_stats(w)
    rec["wl_off_integer_max"] = float(np.nanmax(np.abs(w - np.round(w))))
    same = np.allclose(wl, w[:, None], equal_nan=True, atol=1e-6)
    rec["wl_identical_across_cycles"] = bool(same)
    if not same:
        rec["issues"].append("wavelength axis differs between cycles")
    rec["n_bad_cells"] = bad
    rec["n_missing_abs"] = int(np.isnan(absorb).sum())
    rec["data_hash"] = hashlib.sha1(np.round(absorb, 6).tobytes()).hexdigest()

    order = np.argsort(w)
    def at(nm: float) -> np.ndarray:
        return np.array([np.interp(nm, w[order], absorb[order, k]) for k in range(absorb.shape[1])])
    a254 = at(254.0)
    rec["A254_mean"] = float(a254.mean())
    rec["A254_sd_cycles"] = float(a254.std(ddof=1)) if a254.size > 1 else np.nan
    rec["A200_mean"] = float(at(200.0).mean())
    rec["A700_mean"] = float(at(700.0).mean())
    band = (w >= 240) & (w <= 700)
    rec["max_cycle_sd_240_700"] = float(np.nanmax(absorb[band].std(axis=1, ddof=1))) if band.any() else np.nan

    scans = parse_log(log)
    rec["n_scans_logged"] = len(scans)
    for col, key in ABS_PARAMS.items():
        val, inconsistent = summarise_param(scans, key)
        rec[col] = val
        if inconsistent:
            rec["issues"].append(f"'{key}' varies between cycles")
    times = [t for t in (parse_cary_time(sc["_collection_time"]) for sc in scans) if t]
    rec["first_scan"] = min(times) if times else None
    rec["last_scan"] = max(times) if times else None
    rec["method_name"] = "|".join(dict.fromkeys(sc.get("_method_name", "") for sc in scans))
    return rec


# --------------------------------------------------------------------------- #
# Walking the tree
# --------------------------------------------------------------------------- #

def walk(root: Path, sub: Path, technique: str, inspector) -> tuple[pd.DataFrame, list[dict]]:
    rows, ignored = [], []
    base = root / sub
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(base).parts
        operator = parts[0] if len(parts) > 1 else ""
        folder = parts[1] if len(parts) > 2 else ""
        if path.suffix.lower() != ".csv":
            ignored.append(dict(technique=technique, file=rel, operator=operator, folder=folder,
                                extension=path.suffix or "(none)", size_bytes=path.stat().st_size))
            continue
        rec = dict(technique=technique, file=rel, operator=operator, folder=folder,
                   file_stem=path.stem, size_bytes=path.stat().st_size)
        try:
            rec.update(inspector(path))
        except Exception as exc:  # report, never stop the inventory
            rec["issues"] = [f"PARSE ERROR: {exc!r}"]
        if len(parts) != 3:
            rec["issues"].append("unexpected folder depth")
        if rec.get("name_in_file") and rec["name_in_file"] != path.stem:
            rec["issues"].append(f"header name {rec['name_in_file']!r} != file name")
        info = parse_sample_name(path.stem)
        rec.update({k if k != "kind" else "sample_kind": v for k, v in info.items()})
        rows.append(rec)

    by_hash: dict = {}
    for rec in rows:
        if rec.get("data_hash"):
            by_hash.setdefault(rec["data_hash"], []).append(rec)
    for rec in rows:
        others = [r["file"] for r in by_hash.get(rec.get("data_hash"), []) if r is not rec]
        rec["duplicate_of"] = "|".join(others)
        if others:
            rec["issues"].append("identical data to " + ", ".join(others))
    df = pd.DataFrame(rows)
    return df, ignored


def assign_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    """Assign river and campaign (YYYY-MM) to every file of a folder.

    Month comes from the sample-name suffix when the folder has one (Orcia),
    otherwise from the folder name (Italian month or date).
    """
    df = df.copy()
    df["river_folder"] = ""
    df["campaign"] = ""
    df["campaign_source"] = ""
    df["folder_date"] = ""
    for (_, folder), idx in df.groupby(["operator", "folder"]).groups.items():
        sub = df.loc[idx]
        rivers = Counter(r for r in sub["river"] if r)
        river = rivers.most_common(1)[0][0] if rivers else ""
        tokens = Counter(s for s in sub["suffix"] if s in IT_MONTH_ABBR)
        fdate = folder_date(folder)
        year = fdate.year if fdate else DEFAULT_YEAR
        if tokens:
            month = IT_MONTH_ABBR[tokens.most_common(1)[0][0]]
            source = "sample-name suffix"
        else:
            month = folder_month(folder)
            source = "folder date" if fdate else "folder month name"
        if not fdate:
            source += f" (year {year} assumed)"
        df.loc[idx, "river_folder"] = river
        df.loc[idx, "campaign"] = f"{year}-{month:02d}" if month else ""
        df.loc[idx, "campaign_source"] = source
        df.loc[idx, "folder_date"] = fdate.strftime("%Y-%m-%d") if fdate else ""
    return df


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def same_key(a: str, b: str) -> bool:
    ka, kb = parse_sample_name(a), parse_sample_name(b)
    fields = ("kind", "river", "site", "rep", "variant")
    return ka["kind"] != "unparsed" and all(ka[f] == kb[f] for f in fields)


def day(t) -> str:
    return t.strftime("%Y-%m-%d") if isinstance(t, datetime) and not pd.isna(t) else ""


def build_pairing(eem: pd.DataFrame, ab: pd.DataFrame) -> pd.DataFrame:
    out = []
    campaigns = sorted(set(zip(eem["river_folder"], eem["campaign"])) |
                       set(zip(ab["river_folder"], ab["campaign"])))
    for river, camp in campaigns:
        e = eem[(eem.river_folder == river) & (eem.campaign == camp)]
        a = ab[(ab.river_folder == river) & (ab.campaign == camp)]
        e_blank = e[e.sample_kind == "blank"]
        a_blank = a[a.sample_kind == "blank"]
        e_s = e[e.sample_kind != "blank"]
        a_s = a[a.sample_kind != "blank"]

        # flags: need a human decision; info: harmless differences, kept for traceability
        camp_flags, camp_info = [], []
        if len(e_blank) != 1:
            camp_flags.append(f"{len(e_blank)} EEM blanks in folder")
        if len(a_blank) != 1:
            camp_flags.append(f"{len(a_blank)} absorbance blanks in folder")
        e_dates = set(e["folder_date"]) - {""}
        a_dates = set(a["folder_date"]) - {""}
        if e_dates and a_dates and e_dates != a_dates:
            camp_info.append(f"folder dates differ: EEM {sorted(e_dates)} vs abs {sorted(a_dates)}")

        def key(r):
            if r.sample_kind == "sample":
                return (r.site, r.rep, r.variant)
            return ("?", r.file_stem.lower().replace(" ", ""), "")

        e_map: dict = {}
        for r in e_s.itertuples():
            e_map.setdefault(key(r), []).append(r)
        a_map: dict = {}
        for r in a_s.itertuples():
            a_map.setdefault(key(r), []).append(r)

        for k in sorted(set(e_map) | set(a_map), key=lambda t: tuple(map(str, t))):
            er = e_map.get(k, [])
            ar = a_map.get(k, [])
            flags, info = list(camp_flags), list(camp_info)
            if len(er) > 1 or len(ar) > 1:
                flags.append("duplicate key")
            er0 = er[0] if er else None
            ar0 = ar[0] if ar else None
            for tech, r in (("EEM", er0), ("abs", ar0)):
                if r is None:
                    continue
                if r.sample_kind == "unparsed":
                    flags.append(f"name {r.file_stem!r} not interpretable")
                if r.suffix and r.suffix not in IT_MONTH_ABBR:
                    flags.append(f"name suffix {r.suffix!r} is not a month token ({r.file_stem!r})")
                if r.name_in_file and r.name_in_file != r.file_stem:
                    msg = f"{tech} file {r.file_stem!r} has header name {r.name_in_file!r}"
                    (info if same_key(r.name_in_file, r.file_stem) else flags).append(msg)
                if r.duplicate_of:
                    flags.append(f"{tech} data identical to {r.duplicate_of}")
            if er0 is not None and ar0 is not None:
                if er0.file_stem != ar0.file_stem:
                    info.append(f"file names differ: {er0.file_stem!r} vs {ar0.file_stem!r}")
                if day(er0.first_scan) != day(ar0.first_scan):
                    flags.append(f"measured on different days: EEM {day(er0.first_scan)}, "
                                 f"abs {day(ar0.first_scan)}")

            if er0 is None:
                status = "missing_eem"
            elif ar0 is None:
                status = "missing_abs"
            elif len(e_blank) != 1 or len(a_blank) != 1:
                status = "missing_blank"
            elif flags:
                status = "check"
            else:
                status = "ok"

            parsed = (er0 if er0 is not None else ar0).sample_kind == "sample"
            out.append(dict(
                river=river,
                campaign=camp,
                site=k[0] if parsed else "",
                replicate=k[1] if parsed else "",
                variant=k[2],
                eem_file=er0.file if er0 is not None else "",
                eem_blank_file=e_blank.iloc[0].file if len(e_blank) == 1 and er0 is not None else "",
                abs_file=ar0.file if ar0 is not None else "",
                abs_blank_file=a_blank.iloc[0].file if len(a_blank) == 1 and ar0 is not None else "",
                eem_measured=day(er0.first_scan) if er0 is not None else "",
                abs_measured=day(ar0.first_scan) if ar0 is not None else "",
                status=status,
                include="yes" if status in ("ok", "check") else "no",
                flags="; ".join(dict.fromkeys(flags)),
                info="; ".join(dict.fromkeys(info)),
                notes="",
            ))
    df = pd.DataFrame(out)
    df.insert(0, "pair_id", [f"P{i:03d}" for i in range(1, len(df) + 1)])
    return df


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def value_counts_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        for v, n in df[c].astype(str).value_counts().items():
            rows.append({"parameter": c, "value": v, "n_files": n})
    return pd.DataFrame(rows)


def write_report(path: Path, data_root: Path, eem: pd.DataFrame, ab: pd.DataFrame,
                 ignored: pd.DataFrame, pairing: pd.DataFrame) -> None:
    L: list[str] = []
    L.append("# Phase 0 – Dataset inventory\n")
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{data_root}` (read-only).\n")

    L.append("## 1. Files found\n")
    L.append(f"- Fluorescence CSV (Cary Eclipse): **{len(eem)}** "
             f"({(eem.sample_kind == 'blank').sum()} blanks)")
    L.append(f"- Absorbance CSV (Cary 60): **{len(ab)}** "
             f"({(ab.sample_kind == 'blank').sum()} blanks)")
    ext = ignored.groupby(["technique", "extension"]).size().reset_index(name="n_files")
    L.append(f"- Ignored non-CSV files: **{len(ignored)}**\n")
    L.append(md_table(ext))

    L.append("## 2. Campaigns\n")
    def per_campaign(df: pd.DataFrame, label: str) -> pd.DataFrame:
        g = df.groupby(["river_folder", "campaign"])
        return pd.DataFrame({
            f"{label}_folder": g["folder"].agg(lambda s: "|".join(sorted(set(s)))),
            f"{label}_samples": g["sample_kind"].agg(lambda s: int((s != "blank").sum())),
            f"{label}_blanks": g["sample_kind"].agg(lambda s: int((s == "blank").sum())),
            f"{label}_measured": g["first_scan"].agg(
                lambda s: "|".join(sorted({t.strftime("%Y-%m-%d") for t in s.dropna()}))),
        })
    camp = per_campaign(eem, "eem").join(per_campaign(ab, "abs"), how="outer").reset_index()
    camp = camp.rename(columns={"river_folder": "river"})
    L.append(md_table(camp))
    src = eem.drop_duplicates(["operator", "folder"])[["river_folder", "folder", "campaign", "campaign_source"]]
    L.append("\nHow each fluorescence folder was assigned to a campaign:\n")
    L.append(md_table(src.rename(columns={"river_folder": "river"})))

    L.append("## 3. Cary Eclipse acquisition parameters\n")
    L.append("Counts are per file; a value like `a|b` means the parameter changed between scans of one file.\n")
    L.append(md_table(value_counts_table(eem, [
        "instrument", "serial", "data_mode", "scan_mode", "ex_slit", "em_slit", "pmt",
        "corrected", "data_interval", "scan_rate", "averaging_time", "ex_filter", "em_filter",
        "em_start", "em_stop", "n_ex", "ex_min", "ex_max", "ex_step"])))
    L.append("\nEmission axis as read from the data rows:\n")
    L.append(md_table(value_counts_table(eem, [
        "n_em", "em_min", "em_max", "em_step_median", "em_identical_across_ex"])))
    L.append(f"\nEmission step range over all files: "
             f"{eem.em_step_min.min():.2f}–{eem.em_step_max.max():.2f} nm.\n")
    L.append(f"\nMaximum raw intensity per file: median {eem.max_intensity.median():.1f}, "
             f"max {eem.max_intensity.max():.1f} a.u. Top 10:\n")
    L.append(md_table(eem.nlargest(10, "max_intensity")[["file", "max_intensity", "max_at_ex_em"]]))

    L.append("## 4. Cary 60 acquisition parameters\n")
    L.append(md_table(value_counts_table(ab, [
        "instrument", "beam_mode", "baseline_correction", "cycle_count", "n_cycles",
        "data_interval", "scan_rate", "averaging_time", "start", "stop", "n_wl",
        "wl_descending", "wl_identical_across_cycles"])))
    L.append(f"\nWavelengths are not exact integers: max deviation from the nearest "
             f"integer {ab.wl_off_integer_max.max():.3f} nm; step "
             f"{ab.wl_step_min.min():.3f}–{ab.wl_step_max.max():.3f} nm.\n")
    L.append("\nBlank absorbance (raw, before subtraction):\n")
    L.append(md_table(ab[ab.sample_kind == "blank"][["file", "A200_mean", "A254_mean", "A700_mean"]].round(4)))
    L.append(f"\nSamples, raw A254 (before blank subtraction): median {ab.A254_mean.median():.3f}, "
             f"range {ab.A254_mean.min():.3f}–{ab.A254_mean.max():.3f}. "
             f"Max SD between cycles (240–700 nm): {ab.max_cycle_sd_240_700.max():.4f}.\n")

    L.append("## 5. Pairing (draft)\n")
    L.append("Full table: `pairing_draft.csv`. Status counts:\n")
    L.append(md_table(pairing.status.value_counts().rename_axis("status").reset_index(name="n")))
    L.append("\nRows that are not `ok`:\n")
    L.append(md_table(pairing[pairing.status != "ok"][
        ["pair_id", "river", "campaign", "site", "replicate", "variant", "status",
         "eem_file", "abs_file", "flags"]]))

    L.append("## 6. File-level issues\n")
    iss = []
    for df in (eem, ab):
        for r in df.itertuples():
            for msg in r.issues:
                iss.append({"file": r.file, "issue": msg})
    L.append(md_table(pd.DataFrame(iss)))

    path.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", default=Path("inventory"), type=Path)
    args = ap.parse_args()

    root = args.data_root.resolve()
    out = args.out.resolve()
    if out == root or root in out.parents:
        raise SystemExit("Refusing to write inside the data folder.")
    out.mkdir(parents=True, exist_ok=True)

    eem, ign_e = walk(root, FLUO_SUBDIR, "fluorescence", inspect_eem)
    ab, ign_a = walk(root, ABS_SUBDIR, "absorbance", inspect_abs)
    eem = assign_campaigns(eem)
    ab = assign_campaigns(ab)
    ignored = pd.DataFrame(ign_e + ign_a)
    pairing = build_pairing(eem, ab)

    for df, name in ((eem, "eem_files.csv"), (ab, "abs_files.csv")):
        df.assign(issues=df["issues"].map("; ".join)).to_csv(out / name, index=False)
    ignored.to_csv(out / "ignored_files.csv", index=False)
    pairing.to_csv(out / "pairing_draft.csv", index=False)
    write_report(out / "inventory_report.md", root, eem, ab, ignored, pairing)

    print(f"EEM files: {len(eem)}  absorbance files: {len(ab)}  ignored: {len(ignored)}")
    print(pairing.status.value_counts().to_string())
    print(f"Outputs in {out}")


if __name__ == "__main__":
    main()
