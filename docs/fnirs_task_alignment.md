# fNIRS task-time alignment layer (synchrony Phase B)

How DB-DOS block/trial structure is mapped into fNIRS sample time, and how
sessions are graded before any training consumes them. Code:
`src/synchronai/synchrony/markers.py`, `care_codes.py`;
drivers: `scripts/fnirs_synchrony_trials.py`, `scripts/build_care_repaired_labels.py`.

## Source-of-truth rules

- **`.hdr [Markers]` is the marker authority.** Rows are
  `time<TAB>code<TAB>frame` inside an `Events="# ... #"` sentinel block;
  `frame = floor(time * SamplingRate)`; CRLF, no trailing newline.
- **`.evt` is derived** (`frame` + 8 trigger bits LSB-first). In R01 the plain
  `.evt` was regenerated downstream and disagrees with the `.hdr` (codes
  recoded, rows unsorted, extra pre-onset rows). `*_old.evt` matches the
  `.hdr`. The parser reads both and records agreement per recording
  (`evt_status`); it never lets a `.evt` override the `.hdr`.
- **PsychoPy sidecar CSVs**: t=0 is script launch, not task start (operators
  waited up to ~77 min on the press-space screen) — raw `.started` seconds are
  never used without an estimated offset. Column sets differ by PsychoPy
  generation and by which routines actually ran; parse by column name.
  Filename IDs are unreliable; the directory path is the identity authority.
  Multiple CSVs in one session dir: largest file wins.
- **Dyad members share byte-identical markers** (NIRScout multi-subject mode),
  so alignment is computed once per session and applies to both members.

## Offset estimation (no trigger semantics assumed)

Trigger codes are undocumented and their meaning changed across task
generations (2020-era markers fire at cue-text onsets; later generations fire
at a silent `x_sound` event 10 s after cue and at trial ends). The estimator
therefore assumes only "some markers coincide with some logged events at one
constant offset per recording":

1. dedupe near-simultaneous sidecar events (PsychoPy logs several columns
   within milliseconds at trial end; unmerged they can out-vote the true
   alignment),
2. all pairwise marker-minus-event differences vote in 1 s bins,
3. the top candidate offsets are each refined (median over matched pairs) and
   the winner is the one matching the most markers, ties broken by median
   absolute residual.

**Session gate:** pass = at least 6 matched markers AND median |residual|
<= 0.25 s. Median, not max: a correct offset shows a millisecond-grade median
while single markers can grab a nearby unrelated event; a wrong offset shows
~0.4 s residuals across the board. Sessions with no sidecar pass only with a
complete regular marker pattern (codes 1/2/3 x 4 trials, "markers-native" —
marker time IS fNIRS time). Wall-clock (hdr Date/Time vs sidecar `date`
column) is minute-grade and never passes; it is kept as a cross-check column.

## Trial windows

`trial_start = cue_onset + 5 s` (the "in 5 seconds" lead); the analysis window
is the **first 105 s** of every trial in every study (R56 — and CARE block 2 —
ran ~105 s plays; other blocks 120 s). Windows are emitted in fNIRS seconds
and samples at the hdr sampling rate (7.8125 Hz everywhere).

## Outputs (git-ignored, under `data/synchrony/`)

- `session_validation.csv` — one row per session; **downstream training must
  filter on `passed`**.
- `trial_table.csv` — per (recording, block, trial):
  `study,session_key,block,activity,trial,cue_onset_fnirs_s,start_fnirs_s,
  end_fnirs_s,start_sample,end_sample,offset_source,session_passed`.
- `care_global_codes.csv` / `labels_trials_care.csv` — CARE block-code ingest
  and its join onto the trial table (adds
  `subject_id,family_id,visit,score_mean,n_coders,score_spread`).
- `data/labels_care_repaired.csv` — CARE per-second labels re-decoded with the
  correct Excel time semantics (see below), same schema as `labels.csv`.

Re-run only the label join as new coding lands:
`python scripts/fnirs_synchrony_trials.py --labels-only`.

## CARE coding-sheet time decoding

Per-second sheets store Time three ways, mixed within files: date-serials
incrementing 60/86400 per row under an `h:mm` format (display "14:44" is the
coder's MM:SS; decode `round(serial*1440)` s), literal `"MM:SS"` strings, and
— in a minority of files — serials incrementing 1/86400 per row where the
stored time-of-day IS the elapsed MM:SS (decode `round(serial*86400)`).
`build_care_repaired_labels.py` picks the serial interpretation per file using
the prior that per-second coding advances by 1 s per row, and flags any file
that fails the continuity check. Global block-code files are fixed-position
templates (B4:J8, one file per block, TRIAL 0-3 rows, half-point scores,
`_N` suffix = reliability coder); identity comes from the directory, activity
from the BLOCK cell with a filename fallback.

## Known open items

- A small set of sessions fails with a distinctive signature: about half the
  markers matching at ~0.4 s median residual — consistent with two systematic
  event-lag families rather than noise. A manual look could recover them.
- Sidecar-less R01 sessions with irregular marker patterns currently fail the
  gate. If R01 behavioral coding lands, a pattern-fitting pass (fitting the
  ~120 s trial spacing while tolerating extra/missing markers) would likely
  recover most of them.
- Per-block video<->task offsets (plan item B4) are not built: the per-second
  labels are in video time; joining them to fNIRS time needs block anchors in
  both clocks.
- Trial 0 of each block is the intro/get-ready trial in both the task logs and
  the global coding sheets; the join keys them 1:1 by index.
