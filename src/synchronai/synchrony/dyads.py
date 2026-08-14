"""Dyad discovery + hardware-alignment verification.

Implements the helpers specified in docs/fnirs_alignment_audit.md (Phase 3):
enumerate sample-aligned parent-child dyads from the QC cache, and verify the
NIRScout multi-subject invariant (shared capture => identical Date/Time/
FileName across both members' .hdr files).

Key facts encoded here (audit 2026-08-12):
- Dyad members are sample-aligned ONLY within one acquisition: pair within
  the same parent folder and the same repeat-run suffix. A child from run 1
  is NOT aligned with a parent from run 2.
- CARE: 4-digit id = adult, 5-digit = child. The 4-digit prefix is a FAMILY
  key, not a dyad key — 34 families have 2-3 enrolled children, so dyad
  identity must use the specific child id.
- R56: {family}-C / {family}-P (DB-DOS only; Flanker is single-subject).
- R01: {family}_C / {family}_P, optional trailing _N run suffix.
- Subject1/Subject2 placeholder dirs cannot be role-labeled; skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from synchronai.utils.logging import get_logger

_CARE_DIR = re.compile(r"^(?P<sub>\d{4,5})_V(?P<tp>\d+)_fNIRS(?:_(?P<run>\d+))?$")
_R56_DIR = re.compile(r"^(?P<fam>\d+)-(?P<role>[CP])_fNIRS")
_R01_DIR = re.compile(r"^(?P<fam>\d+)_(?P<role>[CP])(?:_(?P<run>\d+))?$")


@dataclass(frozen=True)
class Dyad:
    child_path: str
    adult_path: str
    study: str  # CARE | R56 | R01
    site: str  # WashU | PSU | WUSTL
    timepoint: str  # V0/V1/V2 | visit1 | T1/T3/T5
    family_id: str  # split-grouping key (siblings share it)
    child_id: str
    adult_id: str
    run_suffix: str
    child_tier: str = ""
    adult_tier: str = ""

    @property
    def session_key(self) -> str:
        """One dyad-session; repeat runs of the same session share this."""
        return f"{self.study}:{self.child_id}:{self.timepoint}:{self.site}"

    @property
    def dyad_id(self) -> str:
        suffix = f"_run{self.run_suffix}" if self.run_suffix else ""
        return f"{self.study}_{self.child_id}_{self.timepoint}_{self.site}{suffix}"


def classify_fnirs_role(fnirs_path: str) -> dict | None:
    """Study/role/family/timepoint from a recording path, or None if the
    path cannot be confidently role-labeled (placeholders, Flanker, pilots).
    """
    p = str(fnirs_path)
    parts = Path(p).parts
    name = Path(p).name

    if "/CARE/" in p:
        m = _CARE_DIR.match(name)
        if not m:
            return None
        sub, tp, run = m.group("sub"), m.group("tp"), m.group("run") or ""
        role = "child" if len(sub) == 5 else "adult"
        return {
            "study": "CARE", "site": "WashU", "timepoint": f"V{tp}",
            "family_id": sub[:4], "subject_id": sub, "role": role, "run": run,
        }

    if "/R56/" in p:
        if "Flanker" in p or "/test/" in p:
            return None
        m = _R56_DIR.match(name)
        if not m:
            return None
        fam, role_ch = m.group("fam"), m.group("role")
        return {
            "study": "R56", "site": "WashU", "timepoint": "visit1",
            "family_id": fam,
            "subject_id": f"{fam}-{'C' if role_ch == 'C' else 'P'}",
            "role": "child" if role_ch == "C" else "adult", "run": "",
        }

    if "/R01/" in p:
        m = _R01_DIR.match(name)
        if not m:
            return None
        fam, role_ch, run = m.group("fam"), m.group("role"), m.group("run") or ""
        site = "PSU" if "PSU_data" in parts else "WUSTL" if "WUSTL_data" in parts else "unknown"
        tp = next((x for x in parts if x in ("T1", "T3", "T5")), "unknown")
        return {
            "study": "R01", "site": site, "timepoint": tp,
            "family_id": fam, "subject_id": f"{fam}_{role_ch}",
            "role": "child" if role_ch == "C" else "adult", "run": run,
        }

    return None


def discover_dyads(
    qc_csv: str | Path,
    include_tiers: tuple[str, ...] = ("gold", "standard", "salvageable"),
) -> tuple[list[Dyad], dict]:
    """Enumerate dyads from the QC cache.

    Pairs one child + one adult sharing (parent folder, run suffix). Returns
    (dyads, report) where report counts skipped/unpaired recordings.
    """
    logger = get_logger(__name__)
    df = pd.read_csv(qc_csv)
    report = {"unclassifiable": 0, "tier_excluded": 0, "unpaired": 0, "ambiguous_groups": 0}

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in df.itertuples(index=False):
        info = classify_fnirs_role(row.fnirs_path)
        if info is None:
            report["unclassifiable"] += 1
            continue
        if row.quality_tier not in include_tiers:
            report["tier_excluded"] += 1
            continue
        info["path"] = row.fnirs_path
        info["tier"] = row.quality_tier
        key = (str(Path(row.fnirs_path).parent), info["run"])
        groups.setdefault(key, []).append(info)

    dyads: list[Dyad] = []
    for (_, run), members in groups.items():
        children = [m for m in members if m["role"] == "child"]
        adults = [m for m in members if m["role"] == "adult"]
        if len(children) == 1 and len(adults) == 1:
            c, a = children[0], adults[0]
            dyads.append(Dyad(
                child_path=c["path"], adult_path=a["path"],
                study=c["study"], site=c["site"], timepoint=c["timepoint"],
                family_id=c["family_id"], child_id=c["subject_id"],
                adult_id=a["subject_id"], run_suffix=run,
                child_tier=c["tier"], adult_tier=a["tier"],
            ))
        elif len(children) > 1 or len(adults) > 1:
            report["ambiguous_groups"] += 1
        else:
            report["unpaired"] += len(members)

    logger.info(
        "Dyad discovery: %d dyads (%s); skipped: %s",
        len(dyads),
        ", ".join(f"{s}={sum(1 for d in dyads if d.study == s)}"
                  for s in ("CARE", "R56", "R01")),
        report,
    )
    return dyads, report


_HDR_FIELDS = ("Date", "Time", "FileName")


def read_hdr_fields(nirx_dir: str | Path) -> dict[str, str]:
    """Date/Time/FileName from the single .hdr in a NIRx directory."""
    hdrs = sorted(Path(nirx_dir).glob("*.hdr"))
    if len(hdrs) != 1:
        raise FileNotFoundError(f"Expected exactly one .hdr in {nirx_dir}, found {len(hdrs)}")
    out: dict[str, str] = {}
    for line in hdrs[0].read_text(errors="replace").splitlines():
        for f in _HDR_FIELDS:
            if line.startswith(f + "="):
                out[f] = line.split("=", 1)[1].strip().strip('"')
    return out


def verify_dyad_hdr(dyad: Dyad) -> tuple[bool, str]:
    """NIRScout multi-subject invariant: both members' .hdr agree on
    Date/Time/FileName. Returns (ok, detail)."""
    try:
        c = read_hdr_fields(dyad.child_path)
        a = read_hdr_fields(dyad.adult_path)
    except (FileNotFoundError, OSError) as e:
        return False, f"hdr-read-failed: {e}"
    for f in _HDR_FIELDS:
        if c.get(f) != a.get(f):
            return False, f"{f} mismatch: child={c.get(f)!r} adult={a.get(f)!r}"
    return True, "ok"
