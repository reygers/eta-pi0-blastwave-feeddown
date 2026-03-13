import csv
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class DataBlock:
    centrality: str
    columns: List[str]
    rows: List[List[float]]


def load_hepdata_csv(path: str) -> List[DataBlock]:
    blocks: List[DataBlock] = []
    current_cent = None
    current_cols = None
    current_rows: List[List[float]] = []

    def _canonical_centrality_label(raw: str) -> str:
        s = raw.strip()
        if s == "-" or s.lower() == "unknown":
            return s
        # HEPData centrality strings can look like "0.0-5.0", where '-'
        # is a range separator (not a negative sign).
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if len(nums) >= 2:
            lo = int(round(float(nums[0])))
            hi = int(round(float(nums[1])))
            return f"{lo} TO {hi} pct"
        return s

    def flush():
        nonlocal current_cent, current_cols, current_rows
        if current_cent and current_rows:
            blocks.append(DataBlock(current_cent, current_cols, current_rows))
        current_cent = None
        current_cols = None
        current_rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("#:"):
                if first.startswith("#: CENTRALITY"):
                    flush()
                    # format: #: CENTRALITY,,,0 TO 5 pct
                    cent = row[-1].strip() if row[-1].strip() else row[-2].strip()
                    current_cent = _canonical_centrality_label(cent)
                continue
            if first.startswith("PT ") or first.startswith("PT") or "p_{T}" in first or "p_T" in first:
                current_cols = [c.strip() for c in row]
                continue
            # data line
            try:
                vals = [float(x) for x in row if x.strip() != ""]
            except ValueError:
                continue
            if current_cent is None:
                # table without explicit centrality block
                current_cent = "unknown"
            current_rows.append(vals)
    flush()
    return blocks


def block_by_centrality(blocks: List[DataBlock]) -> Dict[str, DataBlock]:
    return {b.centrality: b for b in blocks}


def extract_xy(block: DataBlock):
    # columns: PT, PT LOW, PT HIGH, yield, stat+, stat-, sys+, sys-
    pt = []
    y = []
    stat = []
    sys = []
    for r in block.rows:
        if len(r) < 8:
            continue
        pt.append(r[0])
        y.append(r[3])
        stat.append(abs(r[4]))
        sys.append(abs(r[6]))
    return pt, y, stat, sys


def extract_spectrum(block: DataBlock):
    # columns: PT, PT LOW, PT HIGH, yield, stat+, stat-, sys+, sys-
    pt = []
    pt_low = []
    pt_high = []
    y = []
    stat = []
    sys = []
    for r in block.rows:
        if len(r) < 8:
            continue
        pt.append(r[0])
        pt_low.append(r[1])
        pt_high.append(r[2])
        y.append(r[3])
        stat.append(abs(r[4]))
        sys.append(abs(r[6]))
    return {
        "pt": np.asarray(pt),
        "pt_low": np.asarray(pt_low),
        "pt_high": np.asarray(pt_high),
        "y": np.asarray(y),
        "stat": np.asarray(stat),
        "sys": np.asarray(sys),
    }


def combine_spectra(spectra: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    # Assumes identical pT bins.
    pt = spectra[0]["pt"]
    pt_low = spectra[0]["pt_low"]
    pt_high = spectra[0]["pt_high"]
    y = np.array([s["y"] for s in spectra])
    stat = np.array([s["stat"] for s in spectra])
    sys = np.array([s["sys"] for s in spectra])
    n = y.shape[0]
    return {
        "pt": pt,
        "pt_low": pt_low,
        "pt_high": pt_high,
        "y": y.mean(axis=0),
        "stat": np.sqrt((stat ** 2).sum(axis=0)) / n,
        "sys": sys.mean(axis=0),
    }


def _load_species_block_map(species: str) -> Dict[str, DataBlock]:
    return _load_species_block_map_by_energy(species, "2.76")


def _load_species_block_map_by_energy(species: str, energy: str) -> Dict[str, DataBlock]:
    table_map = {
        "2.76": {
            "pion": ("data/HEPData-ins1377750-v1-csv/Table1.csv", "data/HEPData-ins1377750-v1-csv/Table4.csv"),
            "kaon": ("data/HEPData-ins1377750-v1-csv/Table2.csv", "data/HEPData-ins1377750-v1-csv/Table5.csv"),
            "proton": ("data/HEPData-ins1377750-v1-csv/Table3.csv", "data/HEPData-ins1377750-v1-csv/Table6.csv"),
        },
        "5.02": {
            "pion": ("data/HEPData-ins1759506-v1-csv/Table1.csv",),
            "kaon": ("data/HEPData-ins1759506-v1-csv/Table3.csv",),
            "proton": ("data/HEPData-ins1759506-v1-csv/Table5.csv",),
        },
    }
    if energy not in table_map:
        raise ValueError(f"Unsupported energy '{energy}'.")
    if species not in table_map[energy]:
        raise ValueError(f"Unsupported species '{species}'.")
    blocks = {}
    for t in table_map[energy][species]:
        blocks.update(block_by_centrality(load_hepdata_csv(t)))
    return blocks


def load_combined_charged_hadron_spectrum(species: str, centrality: str, energy: str = "2.76") -> Dict[str, np.ndarray]:
    # centrality options depend on energy and downstream analysis setup.
    block_map = _load_species_block_map_by_energy(species, energy)

    combo_map = {
        "2.76": {
            "0-10": ["0 TO 5 pct", "5 TO 10 pct"],
            "20-50": ["10 TO 20 pct", "20 TO 30 pct", "30 TO 40 pct", "40 TO 50 pct"],
        },
        "5.02": {
            "0-10": ["0 TO 5 pct", "5 TO 10 pct"],
            "0-20": ["0 TO 5 pct", "5 TO 10 pct", "10 TO 20 pct"],
            "20-40": ["20 TO 30 pct", "30 TO 40 pct"],
            "30-50": ["30 TO 40 pct", "40 TO 50 pct"],
            "40-60": ["40 TO 50 pct", "50 TO 60 pct"],
            "60-80": ["60 TO 70 pct", "70 TO 80 pct"],
        },
    }
    if energy not in combo_map:
        raise ValueError(f"Unsupported energy '{energy}'.")
    if centrality not in combo_map[energy]:
        raise ValueError(
            f"Unsupported combined centrality '{centrality}' for {energy} TeV. "
            f"Supported: {sorted(combo_map[energy].keys())}"
        )

    combo = [block_map[c] for c in combo_map[energy][centrality]]

    spectra = [extract_spectrum(b) for b in combo]
    return combine_spectra(spectra)


def load_pp_charged_hadron_spectrum(species: str, energy: str = "2.76") -> Dict[str, np.ndarray]:
    # pp reference spectra by energy.
    # 2.76 TeV: HEPData-ins1276299-v1-csv
    # 5.02 TeV: HEPData-ins1759506-v1-csv
    table_map = {
        "2.76": {
            "pion": "data/HEPData-ins1276299-v1-csv/Table1.csv",
            "kaon": "data/HEPData-ins1276299-v1-csv/Table2.csv",
            "proton": "data/HEPData-ins1276299-v1-csv/Table3.csv",
        },
        "5.02": {
            "pion": "data/HEPData-ins1759506-v1-csv/Table2.csv",
            "kaon": "data/HEPData-ins1759506-v1-csv/Table4.csv",
            "proton": "data/HEPData-ins1759506-v1-csv/Table6.csv",
        },
    }
    if energy not in table_map:
        raise ValueError(f"Unsupported pp energy '{energy}'. Use one of: {list(table_map.keys())}")
    if species not in table_map[energy]:
        raise ValueError(f"Unsupported species '{species}'.")
    blocks = load_hepdata_csv(table_map[energy][species])
    by_cent = block_by_centrality(blocks)
    # In some datasets pp is under CENTRALITY '-' and in others there is no centrality block.
    if "-" in by_cent:
        return extract_spectrum(by_cent["-"])
    if "unknown" in by_cent:
        return extract_spectrum(by_cent["unknown"])
    # fallback: first block
    if blocks:
        return extract_spectrum(blocks[0])

    # final fallback for plain single-table CSVs without CENTRALITY/headers handled above
    rows = []
    with open(table_map[energy][species], "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            first = row[0].strip()
            if first.startswith("#"):
                continue
            vals = []
            ok = True
            for c in row[:8]:
                c = c.strip()
                if c == "":
                    ok = False
                    break
                try:
                    vals.append(float(c))
                except ValueError:
                    ok = False
                    break
            if ok and len(vals) >= 8:
                rows.append(vals[:8])
    if not rows:
        raise RuntimeError(f"Could not find pp block in {table_map[energy][species]}")

    block = DataBlock(centrality="pp", columns=[], rows=rows)
    return extract_spectrum(block)
