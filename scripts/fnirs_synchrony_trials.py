"""Build the Phase-B task-time alignment layer: per-session validation report,
trial/block table in fNIRS time, and the CARE trial-level label file.

Walks the fNIRS recordings of CARE / R56 / R01 (WUSTL + PSU), pairs each
session with its PsychoPy sidecar CSV where one exists, estimates the
sidecar->fNIRS clock offset from the hardware markers (see
synchronai.synchrony.markers for the timing model), and writes:

    {out}/session_validation.csv   per-session alignment grades -- THE GATE:
                                   Phase C trains only on passed sessions
    {out}/trial_table.csv          per (recording, block, trial) windows,
                                   105 s truncation rule applied
    {out}/labels_trials_care.csv   trial table joined with the CARE global
                                   block codes (with --care-labels)
    {out}/care_global_codes.csv    coder-level global scores (with --care-labels)

Dyad members share byte-identical markers (NIRScout multi-subject mode), so
alignment is computed once per session from one member's .hdr and applies to
both. R01's regenerated .evt files are expected to disagree with the .hdr
(recoded + resorted downstream); the evt_status column records this, the .hdr
stays authoritative.

Usage:
    python scripts/fnirs_synchrony_trials.py --study care --limit 5 --verbose
    python scripts/fnirs_synchrony_trials.py --session CARE/50021/V0 --verbose
    python scripts/fnirs_synchrony_trials.py --care-labels
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from synchronai.synchrony import care_codes  # noqa: E402
from synchronai.synchrony.markers import (  # noqa: E402
    SessionAlignment,
    align_session,
    build_trial_rows,
    pick_sidecar_csv,
)

LOCAL_ROOT = "/Volumes/perlmansusan/Active/moochie"
CLUSTER_ROOT = "/storage1/fs1/perlmansusan/Active/moochie"
DATA_ROOT = next(
    (r for r in (LOCAL_ROOT, CLUSTER_ROOT) if os.path.isdir(f"{r}/study_data")),
    LOCAL_ROOT,
)
STUDY_DATA = f"{DATA_ROOT}/study_data"
CARE_GLOBAL_CODES = f"{STUDY_DATA}/CARE/synchrony_coding/global participant codes"


def _nirx_dirs_with_hdr(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(d for d in parent.iterdir()
                  if d.is_dir() and any(d.glob("*.hdr")))


def _pick_recording(cands: list[Path], prefer: str) -> Path | None:
    """One recording per session; dyad members carry identical markers."""
    if not cands:
        return None
    preferred = [d for d in cands if d.name.startswith(prefer)]
    return (preferred or cands)[0]


def discover_sessions(study: str) -> list[tuple[str, str, Path | None, Path | None]]:
    """(study, session_key, nirx_dir, sidecar_csv) per fNIRS session."""
    out = []
    if study == "care":
        nirs_root = Path(f"{STUDY_DATA}/CARE/NIRS_data")
        for child_dir in sorted(nirs_root.iterdir()) if nirs_root.is_dir() else []:
            if not re.fullmatch(r"\d{5}", child_dir.name):
                continue
            for visit_dir in sorted(child_dir.glob("V[0-9]")):
                rec = _pick_recording(_nirx_dirs_with_hdr(visit_dir), child_dir.name)
                if rec is None:
                    continue
                task_dir = Path(f"{STUDY_DATA}/CARE/task_data/DB_DOS/"
                                f"{child_dir.name}/{visit_dir.name}")
                out.append(("CARE", f"CARE/{child_dir.name}/{visit_dir.name}",
                            rec, pick_sidecar_csv(task_dir) if task_dir.is_dir() else None))
    elif study == "r56":
        nirs_root = Path(f"{STUDY_DATA}/P-CAT/R56/NIRS_data")
        for fam_dir in sorted(nirs_root.iterdir()) if nirs_root.is_dir() else []:
            dbdos = fam_dir / f"{fam_dir.name}_DB-DOS"
            rec = _pick_recording(_nirx_dirs_with_hdr(dbdos), f"{fam_dir.name}-C")
            if rec is None:
                continue
            task_dir = Path(f"{STUDY_DATA}/P-CAT/R56/task_data/"
                            f"{fam_dir.name}/{fam_dir.name}_DB-DOS")
            out.append(("R56", f"R56/{fam_dir.name}",
                        rec, pick_sidecar_csv(task_dir) if task_dir.is_dir() else None))
    elif study == "r01":
        for site in ("WUSTL", "PSU"):
            site_root = Path(f"{STUDY_DATA}/P-CAT/R01/data/PSU_share/{site}_data")
            if not site_root.is_dir():
                continue
            for tp_dir in sorted(site_root.glob("T[0-9]")):
                nirs_root = tp_dir / "nirs_data" / "dbdos"
                if not nirs_root.is_dir():
                    continue
                for fam_dir in sorted(nirs_root.iterdir()):
                    if not fam_dir.is_dir():
                        continue
                    rec = _pick_recording(_nirx_dirs_with_hdr(fam_dir),
                                          f"{fam_dir.name}_C")
                    if rec is None:
                        continue
                    task_dir = tp_dir / "task_data" / "dbdos" / fam_dir.name
                    out.append((f"R01-{site}",
                                f"R01-{site}/{tp_dir.name}/{fam_dir.name}",
                                rec,
                                pick_sidecar_csv(task_dir) if task_dir.is_dir() else None))
    else:
        raise ValueError(f"unknown study {study}")
    return out


def session_from_key(key: str) -> tuple[str, str, Path | None, Path | None]:
    """Build one session's paths directly from its key -- avoids walking the
    whole study tree over NFS for --session runs."""
    parts = key.split("/")
    if parts[0] == "CARE":
        child, visit = parts[1], parts[2]
        rec = _pick_recording(
            _nirx_dirs_with_hdr(Path(f"{STUDY_DATA}/CARE/NIRS_data/{child}/{visit}")),
            child)
        task = Path(f"{STUDY_DATA}/CARE/task_data/DB_DOS/{child}/{visit}")
    elif parts[0] == "R56":
        fam = parts[1]
        rec = _pick_recording(
            _nirx_dirs_with_hdr(Path(f"{STUDY_DATA}/P-CAT/R56/NIRS_data/{fam}/{fam}_DB-DOS")),
            f"{fam}-C")
        task = Path(f"{STUDY_DATA}/P-CAT/R56/task_data/{fam}/{fam}_DB-DOS")
    elif parts[0].startswith("R01-"):
        site, tp, fam = parts[0][4:], parts[1], parts[2]
        base = Path(f"{STUDY_DATA}/P-CAT/R01/data/PSU_share/{site}_data/{tp}")
        rec = _pick_recording(_nirx_dirs_with_hdr(base / "nirs_data" / "dbdos" / fam),
                              f"{fam}_C")
        task = base / "task_data" / "dbdos" / fam
    else:
        raise SystemExit(f"cannot parse session key {key}")
    if rec is None:
        raise SystemExit(f"no fNIRS recording found for {key}")
    return parts[0], key, rec, pick_sidecar_csv(task) if task.is_dir() else None


def print_verbose(al: SessionAlignment, est_matches) -> None:
    print(f"\n--- {al.session_key} [{al.offset_source}] "
          f"{'PASS' if al.passed else 'FAIL'}: {al.reason}")
    print(f"    markers={al.n_markers} (codes {al.marker_codes})  "
          f"sidecar_events={al.n_sidecar_events}  trials={al.n_sidecar_trials}")
    if al.offset_s is not None:
        print(f"    offset={al.offset_s:+.3f}s  matched={al.n_matched}  "
              f"resid med/max={al.resid_med_s}/{al.resid_max_s}s  "
              f"wallclock_delta={al.wall_vs_marker_s}")
    if est_matches:
        for mk, name, t, r in est_matches:
            print(f"      code{mk.code} @{mk.time_s:9.2f}s  <->  {name:24s} "
                  f"@{t:9.2f}s  resid={r:+.3f}s")
    print(f"    evt: {al.evt_status}")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  (no rows for {path})")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def build_care_label_join(trial_rows: list[dict], out_dir: Path,
                          limit: int = 0) -> None:
    """CARE global codes -> coder-level CSV + trial-table join."""
    print(f"\n=== CARE global codes from {CARE_GLOBAL_CODES}")
    coder_rows, problems = care_codes.parse_global_codes(CARE_GLOBAL_CODES, limit=limit)
    print(f"  parsed {len(coder_rows)} coder-trial scores; {len(problems)} problem files")
    for p in problems[:15]:
        print(f"    ! {p}")
    if len(problems) > 15:
        print(f"    ... and {len(problems) - 15} more")
    write_csv(out_dir / "care_global_codes.csv", coder_rows)

    consensus = care_codes.consensus_scores(coder_rows)
    by_key = {(c["subject_id"], c["visit"], c["block"], c["trial"]): c
              for c in consensus}
    joined, care_trials = [], 0
    matched_keys = set()
    for tr in trial_rows:
        if tr["study"] != "CARE":
            continue
        care_trials += 1
        _, child, visit = tr["session_key"].split("/")
        key = (child, visit, tr["block"], tr["trial"])
        c = by_key.get(key)
        if c is None:
            continue
        matched_keys.add(key)
        joined.append({**{k: v for k, v in tr.items() if k != "label_text"},
                       "subject_id": child, "family_id": child[:4],
                       "visit": visit,
                       "score_mean": round(c["score_mean"], 3),
                       "n_coders": c["n_coders"],
                       "score_spread": c["score_spread"]})
    unjoined_codes = len(by_key) - len(matched_keys)
    print(f"  CARE trials in table: {care_trials}; labeled: {len(joined)}; "
          f"coded trials without an aligned fNIRS trial: {unjoined_codes}")
    n_passed = sum(1 for j in joined if j["session_passed"])
    print(f"  labeled trials on PASSED sessions (Phase-C usable): {n_passed}")
    write_csv(out_dir / "labels_trials_care.csv", joined)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["care", "r56", "r01", "all"], default="all")
    ap.add_argument("--session", default="",
                    help="only this session_key (e.g. CARE/50021/V0)")
    ap.add_argument("--limit", type=int, default=0,
                    help="dry-run: first N sessions per study")
    ap.add_argument("--out-dir", default="data/synchrony")
    ap.add_argument("--care-labels", action="store_true",
                    help="also parse CARE global codes and write the trial-level "
                         "label join")
    ap.add_argument("--labels-only", action="store_true",
                    help="skip the session walk; rebuild the CARE label join from "
                         "the existing trial_table.csv (re-run as coding lands)")
    ap.add_argument("--verbose", action="store_true",
                    help="print per-session matched marker<->event pairs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.labels_only:
        table = out_dir / "trial_table.csv"
        if not table.exists():
            raise SystemExit(f"{table} not found; run the full walk first")
        with open(table, newline="") as fh:
            trial_rows = [{**r, "block": int(r["block"]), "trial": int(r["trial"]),
                           "session_passed": r["session_passed"] == "True"}
                          for r in csv.DictReader(fh)]
        build_care_label_join(trial_rows, out_dir, limit=0)
        return

    sessions: list[tuple[str, str, Path | None, Path | None]] = []
    if args.session:
        sessions = [session_from_key(args.session)]
    else:
        studies = ["care", "r56", "r01"] if args.study == "all" else [args.study]
        for s in studies:
            found = discover_sessions(s)
            print(f"discovered {len(found)} {s} sessions with fNIRS recordings")
            if args.limit:
                found = found[: args.limit]
            sessions.extend(found)

    val_rows: list[dict] = []
    trial_rows: list[dict] = []
    tally = collections.Counter()
    for study, key, nirx, sidecar_csv in sessions:
        al, sidecar, markers = align_session(study, key, nirx, sidecar_csv)
        # keep the matches for verbose printing by re-deriving them cheaply
        if args.verbose:
            from synchronai.synchrony.markers import estimate_offset
            est = (estimate_offset(markers, sidecar.events)
                   if (markers and sidecar and sidecar.events) else None)
            print_verbose(al, est.matches if est else [])
        val_rows.append(asdict(al))
        trial_rows.extend(build_trial_rows(al, sidecar, markers))
        tally[(study, al.offset_source, al.passed)] += 1

    print("\n=== per-session validation summary")
    for (study, source, passed), n in sorted(tally.items()):
        print(f"  {study:10s} {source:15s} {'PASS' if passed else 'fail':4s}  {n}")
    n_pass = sum(1 for r in val_rows if r["passed"])
    print(f"  total sessions: {len(val_rows)}  passed: {n_pass}  "
          f"trial rows: {len(trial_rows)}")
    write_csv(out_dir / "session_validation.csv", val_rows)
    write_csv(out_dir / "trial_table.csv",
              [{k: v for k, v in r.items() if k != "label_text"} for r in trial_rows])

    if args.care_labels:
        build_care_label_join(trial_rows, out_dir, limit=0)


if __name__ == "__main__":
    main()
