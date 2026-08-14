"""Task-time alignment: NIRx markers + PsychoPy sidecars -> fNIRS trial times.

Phase B1 of docs/results/plans/nirsgen_synchrony_utility_plan.md. Maps DB-DOS
block/trial structure into fNIRS sample time and grades every session before
anything downstream trains on it.

Timing model (verified on CARE 50021/V0, the ground-truth session):
- The .hdr [Markers] block is the authority for hardware triggers. Rows are
  ``time<TAB>code<TAB>frame`` inside an ``Events="# ... #"`` sentinel block;
  frame = floor(time * SamplingRate); time is seconds since acquisition start.
- .evt files carry the same events as ``frame<TAB>8 trigger bits (LSB first)``.
  In R01 the plain .evt was REGENERATED downstream: LF line endings, rows not
  frame-sorted, code 3 recoded to 4, code 4 recoded to bit 8 (=128), extra
  bit-8 rows inserted ~78 frames before onsets, some rows dropped. The
  *_old.evt matches the .hdr. We therefore parse both, report disagreement,
  and never let a .evt override the .hdr.
- Marker codes 1/2/3 fire at the block-1/2/3 trial cue onsets, which are the
  ``intro_txt{,2,3}.started`` columns of the PsychoPy sidecar CSV. Markers can
  be incomplete (block 2 entirely absent on 50021/V0) and extra codes (4, and
  paired end-markers in R56) are undocumented, so the offset between the two
  clocks is estimated robustly: modal value of all marker-vs-sidecar-event
  time differences, then refined as the median over matched pairs. Trigger
  semantics are never assumed beyond "some markers coincide with some logged
  events at one constant offset".
- PsychoPy t=0 is script launch, not task start (operators waited up to 77
  minutes on the "press space" screen), so raw sidecar seconds are never used
  without the estimated offset. Wall-clock (hdr Date/Time vs the sidecar
  ``date`` column) gives a coarse cross-check / last-resort offset: minute
  precision in CARE/R56, millisecond in R01, unknown inter-machine skew.

Block truncation rule (cross-study, user-specified 2026-08-12): every trial
window is cue+5s .. cue+5s+105s, because R56 (and CARE block 2) ran 105 s
plays while other blocks ran 120 s.
"""

from __future__ import annotations

import csv
import datetime
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FS = 7.8125
TRIAL_CUE_LEAD_S = 5.0  # "in 5 seconds" audio/text cue precedes play
TRIAL_WINDOW_S = 105.0  # first-105s rule; R56 block 2 ran only 105 s
BLOCK_ACTIVITY = {1: "arts", 2: "puzzles", 3: "magnetiles"}

# Sidecar columns whose .started value is the trial cue onset, per block.
_CUE_COLS = {1: "intro_txt.started", 2: "intro_txt2.started", 3: "intro_txt3.started"}
_LOOP_PREFIX = {1: "trials", 2: "trials_2", 3: "trials_3"}


@dataclass(frozen=True)
class MarkerEvent:
    time_s: float  # seconds since fNIRS acquisition start
    code: int
    frame: int


@dataclass
class SidecarTrial:
    block: int
    trial: int  # 0-based within block (PsychoPy thisTrialN)
    cue_onset: float  # sidecar clock, seconds
    label_text: str = ""


@dataclass
class Sidecar:
    path: str
    trials: list[SidecarTrial] = field(default_factory=list)
    events: list[tuple[str, float]] = field(default_factory=list)  # (column, t)
    start_wall: datetime.datetime | None = None
    id_fields: dict = field(default_factory=dict)


@dataclass
class OffsetEstimate:
    """offset = fnirs_time - sidecar_time; sidecar_t + offset -> fnirs_t."""

    offset_s: float | None = None
    n_matched: int = 0
    matches: list[tuple[MarkerEvent, str, float, float]] = field(default_factory=list)
    # each match: (marker, sidecar column, sidecar time, residual after offset)

    @property
    def residuals(self) -> list[float]:
        return [m[3] for m in self.matches]


# --------------------------------------------------------------------------- #
# NIRx marker sources
# --------------------------------------------------------------------------- #
def _single_file(nirx_dir: Path, suffix: str) -> Path | None:
    hits = sorted(p for p in nirx_dir.glob(f"*{suffix}")
                  if not p.name.startswith("._"))
    return hits[0] if hits else None


def parse_hdr_markers(nirx_dir: str | Path) -> tuple[list[MarkerEvent], float, dict]:
    """Markers, sampling rate, and GeneralInfo fields from the .hdr.

    The Events value is a quoted multi-line block ``Events="#`` ... ``#"``;
    the same #-sentinel form is used by Gains/S-D-Mask etc., so parsing is
    scoped to the line following the [Markers] section header. Files are CRLF
    with no trailing newline.
    """
    nirx_dir = Path(nirx_dir)
    hdr = _single_file(nirx_dir, ".hdr")
    if hdr is None:
        raise FileNotFoundError(f"no .hdr in {nirx_dir}")
    text = hdr.read_text(errors="replace")
    info: dict[str, str] = {}
    for m in re.finditer(r"^(FileName|Date|Time|Device|NIRStar|Subject)=(.*)$",
                         text, re.MULTILINE):
        info[m.group(1)] = m.group(2).strip().strip('"')
    fs_m = re.search(r"^SamplingRate=([\d.]+)", text, re.MULTILINE)
    fs = float(fs_m.group(1)) if fs_m else DEFAULT_FS

    events: list[MarkerEvent] = []
    block = re.search(r"\[Markers\]\s*Events=\"#\s*(.*?)#\"", text, re.DOTALL)
    if block:
        for line in block.group(1).splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 3:
                try:
                    events.append(MarkerEvent(float(parts[0]), int(parts[1]),
                                              int(parts[2])))
                except ValueError:
                    continue
    events.sort(key=lambda e: e.frame)
    return events, fs, info


def parse_evt(path: str | Path, fs: float) -> list[MarkerEvent]:
    """.evt rows -> MarkerEvents. Row = frame + 8 trigger bits (LSB first);
    code = sum(bit_i * 2^(i-1)). Handles CRLF and LF; sorts by frame because
    R01's regenerated .evt files are not frame-ordered."""
    events = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.strip().split("\t")
        if len(parts) != 9:
            continue
        try:
            frame = int(parts[0])
            code = sum(int(b) << i for i, b in enumerate(parts[1:9]))
        except ValueError:
            continue
        events.append(MarkerEvent(frame / fs, code, frame))
    events.sort(key=lambda e: e.frame)
    return events


def compare_marker_sources(nirx_dir: str | Path) -> dict:
    """hdr vs .evt vs _old.evt agreement report for one recording.

    Agreement means identical (frame, code) sequences. R01 plain .evt is
    expected to disagree (regenerated file); the hdr stays authoritative.
    """
    nirx_dir = Path(nirx_dir)
    hdr_events, fs, _ = parse_hdr_markers(nirx_dir)
    hdr_set = [(e.frame, e.code) for e in hdr_events]
    out = {"n_hdr": len(hdr_events)}
    for tag, pattern in (("evt", "*.evt"), ("old_evt", "*_old.evt")):
        paths = sorted(p for p in nirx_dir.glob(pattern)
                       if not p.name.startswith("._"))
        if tag == "evt":  # *.evt also matches *_old.evt
            paths = [p for p in paths if not p.name.endswith("_old.evt")]
        if not paths:
            out[tag] = "missing"
            continue
        evt = [(e.frame, e.code) for e in parse_evt(paths[0], fs)]
        if evt == hdr_set:
            out[tag] = "match"
        else:
            common = len(set(evt) & set(hdr_set))
            out[tag] = f"DISAGREE(n={len(evt)},common={common})"
    return out


# --------------------------------------------------------------------------- #
# PsychoPy sidecars
# --------------------------------------------------------------------------- #
def _parse_sidecar_date(s: str) -> datetime.datetime | None:
    for fmt in ("%Y_%b_%d_%H%M",            # CARE/R56: 2021_Mar_27_0825
                "%Y-%m-%d_%Hh%M.%S.%f"):    # R01: 2023-03-23_15h01.55.450
        try:
            return datetime.datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def parse_hdr_start_wall(info: dict) -> datetime.datetime | None:
    """GeneralInfo Date="Sat, Mar 27, 2021" + Time="09:42:20.058" -> datetime."""
    try:
        return datetime.datetime.strptime(
            f"{info.get('Date', '')} {info.get('Time', '')}",
            "%a, %b %d, %Y %H:%M:%S.%f")
    except ValueError:
        return None


def parse_psychopy_csv(path: str | Path) -> Sidecar:
    """Wide-format DB-DOS export -> trials + all timestamped events.

    Column sets vary by PsychoPy generation and by which routines actually ran
    (aborted runs write fewer columns), so everything is looked up by name.
    Values may be 'None'; text fields contain quoted commas (R01).
    """
    sc = Sidecar(path=str(path))
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for col, val in row.items():
            if col is None or val in (None, "", "None"):
                continue
            if col.endswith((".started", ".stopped")):
                try:
                    sc.events.append((col, float(val)))
                except ValueError:
                    continue
        for blk, prefix in _LOOP_PREFIX.items():
            tn = row.get(f"{prefix}.thisTrialN", "")
            if tn in (None, "", "None"):
                continue
            cue = row.get(_CUE_COLS[blk], "")
            try:
                trial = int(float(tn))
                cue_t = float(cue)
            except (ValueError, TypeError):
                continue
            sc.trials.append(SidecarTrial(block=blk, trial=trial, cue_onset=cue_t,
                                          label_text=row.get("start_txt", "") or ""))
    for row in rows:  # id/date columns repeat on most rows; take first non-empty
        if sc.start_wall is None and row.get("date"):
            sc.start_wall = _parse_sidecar_date(row["date"])
        for k in ("participant", "visit", "family_id", "timepoint",
                  "expName", "psychopyVersion"):
            if row.get(k) and k not in sc.id_fields:
                sc.id_fields[k] = row[k]
    sc.trials.sort(key=lambda t: (t.block, t.trial))
    return sc


def pick_sidecar_csv(session_dir: str | Path) -> Path | None:
    """Largest CSV wins: aborted stubs are near-empty, and the one real
    re-run case (R01 T3/11004) has the longer run first, so 'last file'
    heuristics are wrong. AppleDouble files excluded."""
    cands = [p for p in Path(session_dir).glob("*.csv")
             if not p.name.startswith(("._", "~$"))]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_size)


# --------------------------------------------------------------------------- #
# Offset estimation
# --------------------------------------------------------------------------- #
def _dedupe_events(events: list[tuple[str, float]],
                   window: float = 0.25) -> list[tuple[str, float]]:
    """Collapse near-simultaneous sidecar events to one representative.

    PsychoPy logs several columns for the same instant (countdown.stopped,
    time_up.started, timeup_txt.started, bellring.started land within
    milliseconds at every trial end). Left unmerged they cast 3-4 votes for
    their alignment and can out-vote the true single-column cue onsets --
    exactly what mis-aligned CARE 50021/V0 by 4 s before this dedup existed.
    """
    out: list[tuple[str, float]] = []
    for name, t in sorted(events, key=lambda e: e[1]):
        if out and t - out[-1][1] <= window:
            continue
        out.append((name, t))
    return out


def estimate_offset(markers: list[MarkerEvent], events: list[tuple[str, float]],
                    tol: float = 0.5) -> OffsetEstimate:
    """Robust constant offset between the fNIRS and sidecar clocks.

    Events are deduped, then all pairwise differences (marker.time - event.t)
    vote in 2*tol bins. The top few candidate offsets are each refined
    (median over matched pairs) and scored; the winner is the one matching
    the most markers, ties broken by median |residual|. No trigger semantics
    assumed: undocumented codes simply go unmatched. Verified at +-0.02 s on
    CARE 50021/V0 (codes 1/3 vs intro_txt cue onsets).
    """
    est = OffsetEstimate()
    if not markers or not events:
        return est
    ded = _dedupe_events(events)
    width = 2 * tol
    bins: dict[int, int] = {}
    for mk in markers:
        for _, t in ded:
            b = int(round((mk.time_s - t) / width))
            bins[b] = bins.get(b, 0) + 1

    def bin_score(b: int) -> int:
        return bins[b] + bins.get(b - 1, 0) + bins.get(b + 1, 0)

    candidates: list[int] = []
    for b in sorted(bins, key=bin_score, reverse=True):
        if any(abs(b - c) <= 2 for c in candidates):
            continue
        candidates.append(b)
        if len(candidates) == 5:
            break

    def match(off: float) -> list[tuple[MarkerEvent, str, float, float]]:
        out = []
        for mk in markers:
            name, t, r = None, None, None
            for ev_name, ev_t in ded:
                res = mk.time_s - (ev_t + off)
                if abs(res) <= tol and (r is None or abs(res) < abs(r)):
                    name, t, r = ev_name, ev_t, res
            if name is not None:
                out.append((mk, name, t, r))
        return out

    best: tuple[int, float] | None = None  # (-n_matched, med_abs_resid)
    for cand in candidates:
        matched = match(cand * width)
        if not matched:
            continue
        diffs = sorted(m[0].time_s - m[2] for m in matched)
        offset = diffs[len(diffs) // 2]
        matched = match(offset)
        resid = sorted(abs(m[3]) for m in matched)
        key = (-len(matched), resid[len(resid) // 2])
        if best is None or key < best:
            best = key
            est.matches = matched
            est.offset_s = offset
            est.n_matched = len(matched)
    return est


# --------------------------------------------------------------------------- #
# Session alignment + trial table
# --------------------------------------------------------------------------- #
@dataclass
class SessionAlignment:
    study: str
    session_key: str  # e.g. CARE/50021/V0, R56/1102, R01-WUSTL/T1/11001
    nirx_dir: str = ""
    sidecar_path: str = ""
    fs: float = DEFAULT_FS
    n_markers: int = 0
    marker_codes: str = ""
    n_sidecar_events: int = 0
    n_sidecar_trials: int = 0
    n_matched: int = 0
    offset_s: float | None = None
    resid_med_s: float | None = None
    resid_max_s: float | None = None
    wallclock_offset_s: float | None = None
    wall_vs_marker_s: float | None = None
    evt_status: str = ""
    offset_source: str = ""  # markers | markers-native | wallclock | none
    passed: bool = False
    reason: str = ""


def align_session(study: str, session_key: str, nirx_dir: str | Path | None,
                  sidecar_csv: str | Path | None) -> tuple[SessionAlignment, Sidecar | None, list[MarkerEvent]]:
    """Estimate the sidecar->fNIRS offset for one session and grade it.

    Grades:
    - markers        : offset from matched marker/sidecar events. Passes when
                       >=6 matches and MEDIAN |residual| <=0.25 s. The median
                       (not max) is deliberate: later-generation sessions log
                       events with up to ~0.5 s jitter and a single marker can
                       grab a nearby unrelated event, but a correct offset
                       shows a tight median (<=6 ms in 90% of sessions) while
                       a wrong one shows ~0.4 s across the board.
    - markers-native : no sidecar, but a complete regular marker pattern
                       (codes 1/2/3, 4 trials each). Timing is native fNIRS
                       time so no offset is needed; passes with that caveat.
    - wallclock      : sidecar only, offset from hdr wall clock vs sidecar
                       date column. Minute precision in CARE/R56 -> never
                       passes; kept for triage.
    - none           : nothing usable; fails.
    """
    al = SessionAlignment(study=study, session_key=session_key)
    markers: list[MarkerEvent] = []
    sidecar: Sidecar | None = None
    info: dict = {}

    if nirx_dir is not None:
        al.nirx_dir = str(nirx_dir)
        try:
            markers, al.fs, info = parse_hdr_markers(nirx_dir)
            al.evt_status = str(compare_marker_sources(nirx_dir))
        except (FileNotFoundError, OSError) as e:
            al.reason = f"hdr-read-failed: {e}"
        al.n_markers = len(markers)
        codes = sorted({e.code for e in markers})
        al.marker_codes = "|".join(str(c) for c in codes)

    if sidecar_csv is not None and Path(sidecar_csv).exists():
        al.sidecar_path = str(sidecar_csv)
        try:
            sidecar = parse_psychopy_csv(sidecar_csv)
            al.n_sidecar_events = len(sidecar.events)
            al.n_sidecar_trials = len(sidecar.trials)
        except (OSError, csv.Error) as e:
            al.reason = f"sidecar-read-failed: {e}"
            sidecar = None

    # wall-clock cross-check / fallback
    nirs_wall = parse_hdr_start_wall(info) if info else None
    if nirs_wall and sidecar and sidecar.start_wall:
        al.wallclock_offset_s = (sidecar.start_wall - nirs_wall).total_seconds()

    if markers and sidecar and sidecar.events:
        est = estimate_offset(markers, sidecar.events)
        if est.n_matched:
            al.n_matched = est.n_matched
            al.offset_s = est.offset_s
            res = sorted(abs(r) for r in est.residuals)
            al.resid_med_s = res[len(res) // 2]
            al.resid_max_s = res[-1]
            al.offset_source = "markers"
            if al.wallclock_offset_s is not None:
                al.wall_vs_marker_s = al.wallclock_offset_s - est.offset_s
            if est.n_matched >= 6 and al.resid_med_s <= 0.25:
                al.passed = True
                al.reason = "ok"
            else:
                al.reason = (f"weak-match: n={est.n_matched} "
                             f"med_resid={al.resid_med_s:.3f}s "
                             f"max_resid={al.resid_max_s:.3f}s")
            return al, sidecar, markers

    if markers and sidecar is None:
        per_code = {c: [e for e in markers if e.code == c] for c in (1, 2, 3)}
        if all(len(v) == 4 for v in per_code.values()):
            al.offset_source = "markers-native"
            al.passed = True
            al.reason = "no sidecar; regular 3x4 marker pattern, native timing"
        else:
            al.offset_source = "markers-native"
            al.reason = ("no sidecar and irregular marker pattern: " +
                         ",".join(f"code{c}x{len(v)}" for c, v in per_code.items()))
        return al, None, markers

    if sidecar and al.wallclock_offset_s is not None:
        al.offset_s = -al.wallclock_offset_s
        al.offset_source = "wallclock"
        al.reason = "no usable markers; wall-clock offset only (minute-grade)"
        return al, sidecar, markers

    al.offset_source = "none"
    if not al.reason:
        al.reason = "no markers and no sidecar"
    return al, sidecar, markers


def build_trial_rows(al: SessionAlignment, sidecar: Sidecar | None,
                     markers: list[MarkerEvent]) -> list[dict]:
    """Per (block, trial) rows with fNIRS-time windows.

    trial_start = cue + 5 s (the "in 5 seconds" lead); the analysis window is
    the first TRIAL_WINDOW_S (105 s) of every trial, the cross-study rule that
    absorbs R56's (and CARE block 2's) shortened plays.
    """
    rows: list[dict] = []

    def emit(block: int, trial: int, cue_fnirs: float, label: str) -> None:
        start = cue_fnirs + TRIAL_CUE_LEAD_S
        end = start + TRIAL_WINDOW_S
        rows.append({
            "study": al.study, "session_key": al.session_key,
            "block": block, "activity": BLOCK_ACTIVITY[block], "trial": trial,
            "cue_onset_fnirs_s": round(cue_fnirs, 3),
            "start_fnirs_s": round(start, 3), "end_fnirs_s": round(end, 3),
            "start_sample": int(round(start * al.fs)),
            "end_sample": int(round(end * al.fs)),
            "offset_source": al.offset_source, "session_passed": al.passed,
            "label_text": label,
        })

    if sidecar is not None and al.offset_s is not None:
        for t in sidecar.trials:
            emit(t.block, t.trial, t.cue_onset + al.offset_s, t.label_text)
    elif sidecar is None and al.offset_source == "markers-native":
        for code in (1, 2, 3):
            for i, mk in enumerate(e for e in markers if e.code == code):
                emit(code, i, mk.time_s, "")
    return rows
