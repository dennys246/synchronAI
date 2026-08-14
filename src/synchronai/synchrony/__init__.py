"""Dyadic fNIRS synchrony: wavelet transform coherence + dyad discovery."""

from synchronai.synchrony.wtc import (
    morlet_cwt,
    wavelet_coherence,
    band_mean_wtc,
    coi_mask,
    drop_duplicate_channels,
)
from synchronai.synchrony.dyads import (
    Dyad,
    classify_fnirs_role,
    discover_dyads,
    read_hdr_fields,
    verify_dyad_hdr,
)
from synchronai.synchrony.markers import (
    MarkerEvent,
    SessionAlignment,
    align_session,
    build_trial_rows,
    compare_marker_sources,
    estimate_offset,
    parse_evt,
    parse_hdr_markers,
    parse_psychopy_csv,
)
