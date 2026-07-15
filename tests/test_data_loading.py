"""
Tests for data loading modules.

Covers:
- synchronai.data.video.dataset (VideoWindowSpec, VideoDatasetConfig, split_by_video,
  VideoWindowDataset class_weights / temporal jitter logic)
- synchronai.data.multimodal.dataset_mm (MultiModalDatasetConfig, MultiModalDataset,
  create_multimodal_splits)

All tests avoid real video/audio I/O.  VideoReaderPool and read_window_frames are
mocked wherever a VideoWindowDataset is instantiated.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from synchronai.data.video.dataset import (
    VideoDatasetConfig,
    VideoWindowDataset,
    VideoWindowSpec,
    split_by_video,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    video_path: str = "/videos/v1.mp4",
    second: int = 0,
    label: int = 0,
    video_fps: float = 30.0,
    sample_fps: float = 12.0,
    window_seconds: float = 2.0,
    frame_size: int = 640,
    subject_id: Optional[str] = None,
    session: Optional[str] = None,
) -> VideoWindowSpec:
    """Create a VideoWindowSpec with sensible defaults."""
    return VideoWindowSpec(
        video_path=video_path,
        second=second,
        label=label,
        video_fps=video_fps,
        sample_fps=sample_fps,
        window_seconds=window_seconds,
        frame_size=frame_size,
        subject_id=subject_id,
        session=session,
    )


def _make_config(labels_file: str = "/tmp/labels.csv", **overrides) -> VideoDatasetConfig:
    """Create a VideoDatasetConfig with sensible defaults."""
    defaults = dict(
        labels_file=labels_file,
        sample_fps=12.0,
        window_seconds=2.0,
        frame_size=640,
        augment=False,
    )
    defaults.update(overrides)
    return VideoDatasetConfig(**defaults)


def _build_specs_for_split(
    n_videos: int = 5,
    windows_per_video: int = 10,
    subject_ids: Optional[list[Optional[str]]] = None,
) -> list[VideoWindowSpec]:
    """Build a list of specs spanning *n_videos* videos.

    Each video has *windows_per_video* consecutive seconds.
    If *subject_ids* is given it must have length *n_videos*; each entry is
    the subject_id assigned to that video (None means no subject).
    """
    specs = []
    for v in range(n_videos):
        vid_path = f"/videos/video_{v}.mp4"
        sid = subject_ids[v] if subject_ids else None
        for s in range(windows_per_video):
            specs.append(
                _make_spec(
                    video_path=vid_path,
                    second=s,
                    label=s % 2,
                    subject_id=sid,
                )
            )
    return specs


# ===========================================================================
# VideoWindowSpec tests
# ===========================================================================


class TestVideoWindowSpec:
    """Tests for the VideoWindowSpec frozen dataclass."""

    def test_n_frames_basic(self):
        """n_frames should equal int(sample_fps * window_seconds)."""
        spec = _make_spec(sample_fps=12.0, window_seconds=2.0)
        assert spec.n_frames == 24

    def test_n_frames_fractional(self):
        """n_frames truncates to int for non-integer products."""
        spec = _make_spec(sample_fps=10.0, window_seconds=1.5)
        # 10 * 1.5 = 15.0
        assert spec.n_frames == 15

    def test_n_frames_fractional_truncation(self):
        """n_frames uses int() truncation, not rounding."""
        spec = _make_spec(sample_fps=7.0, window_seconds=3.0)
        # 7 * 3 = 21
        assert spec.n_frames == 21

        spec2 = _make_spec(sample_fps=5.0, window_seconds=0.3)
        # 5 * 0.3 = 1.5 -> int() = 1
        assert spec2.n_frames == 1

    def test_n_frames_one_second(self):
        """One-second window at 12 fps should yield 12 frames."""
        spec = _make_spec(sample_fps=12.0, window_seconds=1.0)
        assert spec.n_frames == 12

    def test_frozen(self):
        """VideoWindowSpec should be immutable."""
        spec = _make_spec()
        with pytest.raises(AttributeError):
            spec.label = 99  # type: ignore[misc]

    def test_optional_fields_default_none(self):
        """subject_id and session should default to None."""
        spec = _make_spec()
        assert spec.subject_id is None
        assert spec.session is None

    def test_optional_fields_set(self):
        """subject_id and session should store the given values."""
        spec = _make_spec(subject_id="S001", session="V0")
        assert spec.subject_id == "S001"
        assert spec.session == "V0"


# ===========================================================================
# VideoDatasetConfig tests
# ===========================================================================


class TestVideoDatasetConfig:
    """Tests for VideoDatasetConfig default values."""

    def test_defaults(self):
        """All default values should match the expected factory settings."""
        cfg = VideoDatasetConfig(labels_file="/tmp/labels.csv")
        assert cfg.sample_fps == 12.0
        assert cfg.window_seconds == 2.0
        assert cfg.frame_size == 640
        assert cfg.augment is False
        assert cfg.horizontal_flip_prob == 0.5
        assert cfg.color_jitter is False
        assert cfg.temporal_jitter_frames == 0
        assert cfg.random_erase_prob == 0.3
        assert cfg.random_erase_scale == (0.02, 0.2)
        assert cfg.gaussian_noise_std == 0.02
        assert cfg.mixup_alpha == 0.2
        assert cfg.reader_pool_size == 8
        assert cfg.video_backend == "auto"

    def test_override(self):
        """Overriding individual fields should work."""
        cfg = VideoDatasetConfig(
            labels_file="/data/labels.csv",
            sample_fps=6.0,
            augment=True,
            temporal_jitter_frames=3,
        )
        assert cfg.sample_fps == 6.0
        assert cfg.augment is True
        assert cfg.temporal_jitter_frames == 3
        # Unchanged defaults should remain
        assert cfg.frame_size == 640


# ===========================================================================
# split_by_video tests
# ===========================================================================


class TestSplitByVideo:
    """Tests for group-based train/val splitting."""

    def test_both_splits_non_empty(self):
        """Both train and val should be non-empty for a reasonable val_split."""
        specs = _build_specs_for_split(n_videos=5, windows_per_video=10)
        train, val = split_by_video(specs, val_split=0.2, seed=42)
        assert len(train) > 0
        assert len(val) > 0
        assert len(train) + len(val) == len(specs)

    def test_groups_stay_together_video_path(self):
        """All specs from the same video_path must end up in the same split."""
        specs = _build_specs_for_split(n_videos=6, windows_per_video=8)
        train, val = split_by_video(specs, val_split=0.3, group_by="video_path", seed=7)

        train_videos = {s.video_path for s in train}
        val_videos = {s.video_path for s in val}
        # No video should appear in both splits
        assert train_videos.isdisjoint(val_videos), (
            f"Leaking videos: {train_videos & val_videos}"
        )

    def test_groups_stay_together_subject_id(self):
        """All specs for the same subject_id must end up in the same split."""
        subject_ids = ["S1", "S1", "S2", "S2", "S3"]
        specs = _build_specs_for_split(
            n_videos=5, windows_per_video=6, subject_ids=subject_ids
        )
        train, val = split_by_video(
            specs, val_split=0.3, group_by="subject_id", seed=99
        )

        train_subjects = {s.subject_id for s in train}
        val_subjects = {s.subject_id for s in val}
        assert train_subjects.isdisjoint(val_subjects), (
            f"Leaking subjects: {train_subjects & val_subjects}"
        )

    def test_subject_id_fallback_to_video_path(self):
        """When group_by='subject_id' but subject_id is None, fall back to video_path.

        This verifies that specs with None subject_id are still grouped (by
        video_path) rather than all being dumped into the training set.
        """
        # All subject_ids are None
        specs = _build_specs_for_split(
            n_videos=5,
            windows_per_video=8,
            subject_ids=[None, None, None, None, None],
        )
        train, val = split_by_video(
            specs, val_split=0.2, group_by="subject_id", seed=42
        )

        # Should still produce non-empty splits
        assert len(train) > 0
        assert len(val) > 0

        # Groups (video_path here) must not leak
        train_paths = {s.video_path for s in train}
        val_paths = {s.video_path for s in val}
        assert train_paths.isdisjoint(val_paths)

    def test_subject_id_mixed_none_and_set(self):
        """Specs with subject_id=None should fall back to video_path grouping
        while specs with subject_id set are grouped by subject_id.
        """
        # Video 0 and 1 have subject_id; video 2 and 3 do not
        subject_ids = ["S1", "S1", None, None]
        specs = _build_specs_for_split(
            n_videos=4, windows_per_video=5, subject_ids=subject_ids
        )
        train, val = split_by_video(
            specs, val_split=0.3, group_by="subject_id", seed=11
        )

        assert len(train) + len(val) == len(specs)
        # The None-subject specs should be grouped by video_path
        for path in ["/videos/video_2.mp4", "/videos/video_3.mp4"]:
            split_assignment = {
                "train" if s in train else "val"
                for s in specs
                if s.video_path == path
            }
            assert len(split_assignment) == 1, (
                f"Specs from {path} (no subject) appear in both splits"
            )

    def test_deterministic_same_seed(self):
        """Same seed should produce identical splits."""
        specs = _build_specs_for_split(n_videos=8, windows_per_video=5)
        train1, val1 = split_by_video(specs, val_split=0.25, seed=123)
        train2, val2 = split_by_video(specs, val_split=0.25, seed=123)
        assert train1 == train2
        assert val1 == val2

    def test_different_seeds_produce_different_splits(self):
        """Different seeds should (with high probability) produce different splits."""
        specs = _build_specs_for_split(n_videos=10, windows_per_video=5)
        train1, _ = split_by_video(specs, val_split=0.3, seed=1)
        train2, _ = split_by_video(specs, val_split=0.3, seed=2)

        train_paths_1 = {s.video_path for s in train1}
        train_paths_2 = {s.video_path for s in train2}
        # With 10 groups and 30% val, different seeds should almost certainly
        # pick a different set of val groups.
        assert train_paths_1 != train_paths_2

    def test_val_split_at_least_one_group(self):
        """Even with a very small val_split the function should assign >= 1 group to val."""
        specs = _build_specs_for_split(n_videos=10, windows_per_video=3)
        _, val = split_by_video(specs, val_split=0.01, seed=42)
        assert len(val) > 0

    def test_all_specs_accounted_for(self):
        """Every input spec must appear in exactly one of the two output lists."""
        specs = _build_specs_for_split(n_videos=4, windows_per_video=7)
        train, val = split_by_video(specs, val_split=0.25, seed=0)
        assert sorted(train + val, key=lambda s: (s.video_path, s.second)) == sorted(
            specs, key=lambda s: (s.video_path, s.second)
        )


# ===========================================================================
# VideoWindowDataset.class_weights tests
# ===========================================================================


class TestClassWeights:
    """Tests for the class_weights property computed at init time."""

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_balanced_binary(self, mock_pool_cls):
        """Balanced binary labels should yield weight 1.0 for both classes."""
        mock_pool_cls.return_value = MagicMock()
        specs = [_make_spec(label=i % 2, second=i) for i in range(20)]
        config = _make_config()
        ds = VideoWindowDataset(specs, config)

        assert ds.class_weights[0] == pytest.approx(1.0)
        assert ds.class_weights[1] == pytest.approx(1.0)

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_imbalanced_binary(self, mock_pool_cls):
        """Imbalanced labels: minority class should have higher weight."""
        mock_pool_cls.return_value = MagicMock()
        # 80 negatives, 20 positives
        specs = [_make_spec(label=0, second=i) for i in range(80)] + [
            _make_spec(label=1, second=i) for i in range(20)
        ]
        config = _make_config()
        ds = VideoWindowDataset(specs, config)

        # weight(0) = 100 / (2 * 80) = 0.625
        # weight(1) = 100 / (2 * 20) = 2.5
        assert ds.class_weights[0] == pytest.approx(0.625)
        assert ds.class_weights[1] == pytest.approx(2.5)

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_single_class(self, mock_pool_cls):
        """With a single class the weight should be 1.0."""
        mock_pool_cls.return_value = MagicMock()
        specs = [_make_spec(label=0, second=i) for i in range(10)]
        config = _make_config()
        ds = VideoWindowDataset(specs, config)

        assert ds.class_weights[0] == pytest.approx(1.0)

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_multiclass(self, mock_pool_cls):
        """Verify class weights with 3 classes."""
        mock_pool_cls.return_value = MagicMock()
        # label 0: 10 samples, label 1: 30 samples, label 2: 60 samples
        specs = (
            [_make_spec(label=0, second=i) for i in range(10)]
            + [_make_spec(label=1, second=i) for i in range(30)]
            + [_make_spec(label=2, second=i) for i in range(60)]
        )
        config = _make_config()
        ds = VideoWindowDataset(specs, config)

        total = 100
        n_classes = 3
        assert ds.class_weights[0] == pytest.approx(total / (n_classes * 10))
        assert ds.class_weights[1] == pytest.approx(total / (n_classes * 30))
        assert ds.class_weights[2] == pytest.approx(total / (n_classes * 60))


# ===========================================================================
# Temporal jitter tests
# ===========================================================================


class TestTemporalJitter:
    """Verify the temporal-jitter offset logic permits negative offsets."""

    @patch("synchronai.data.video.dataset.read_window_frames")
    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_negative_offset_possible(self, mock_pool_cls, mock_read):
        """With second > 0 the temporal offset can be negative (shift earlier)."""
        mock_pool_cls.return_value = MagicMock()
        # Return dummy frames: (n_frames, C, H, W) float32
        n_frames = 24
        dummy = np.zeros((n_frames, 3, 64, 64), dtype=np.float32)
        mock_read.return_value = dummy

        spec = _make_spec(second=5, label=0, sample_fps=12.0, window_seconds=2.0)
        config = _make_config(
            augment=True,
            temporal_jitter_frames=6,
        )
        ds = VideoWindowDataset([spec], config, augment=True)

        # Call __getitem__ many times and collect the 'second' arg passed to
        # read_window_frames.  The effective start time = second + offset.
        # Because second=5 and max_offset = 6/12 = 0.5, the start can be in
        # [4.5, 5.5], so offsets < 0 are valid.
        observed_starts = []
        for _ in range(200):
            _ = ds[0]
            call_kwargs = mock_read.call_args
            # read_window_frames is called with keyword 'second'
            start = call_kwargs.kwargs.get("second", call_kwargs[1].get("second"))
            if start is None:
                # Positional: second is the 2nd arg (index 1)
                start = call_kwargs[0][1]
            observed_starts.append(start)

        min_start = min(observed_starts)
        max_start = max(observed_starts)
        # At least one call should have start < 5.0 (negative offset)
        assert min_start < 5.0, (
            f"Expected some negative offsets (start < 5.0) but min was {min_start}"
        )
        # And at least one > 5.0 (positive offset)
        assert max_start > 5.0, (
            f"Expected some positive offsets (start > 5.0) but max was {max_start}"
        )

    @patch("synchronai.data.video.dataset.read_window_frames")
    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_offset_clamped_at_second_zero(self, mock_pool_cls, mock_read):
        """When second=0 the offset cannot go below 0 (clamped by min_offset)."""
        mock_pool_cls.return_value = MagicMock()
        n_frames = 24
        dummy = np.zeros((n_frames, 3, 64, 64), dtype=np.float32)
        mock_read.return_value = dummy

        spec = _make_spec(second=0, label=0, sample_fps=12.0, window_seconds=2.0)
        config = _make_config(
            augment=True,
            temporal_jitter_frames=6,
        )
        ds = VideoWindowDataset([spec], config, augment=True)

        for _ in range(200):
            _ = ds[0]
            call_kwargs = mock_read.call_args
            start = call_kwargs.kwargs.get("second", call_kwargs[1].get("second"))
            if start is None:
                start = call_kwargs[0][1]
            # Start should never be negative
            assert start >= 0.0, f"Start time went negative: {start}"

    @patch("synchronai.data.video.dataset.read_window_frames")
    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_no_jitter_when_disabled(self, mock_pool_cls, mock_read):
        """When augment=False or temporal_jitter_frames=0, offset should always be 0."""
        mock_pool_cls.return_value = MagicMock()
        n_frames = 24
        dummy = np.zeros((n_frames, 3, 64, 64), dtype=np.float32)
        mock_read.return_value = dummy

        spec = _make_spec(second=5, label=0, sample_fps=12.0, window_seconds=2.0)
        config = _make_config(augment=False, temporal_jitter_frames=6)
        ds = VideoWindowDataset([spec], config, augment=False)

        for _ in range(50):
            _ = ds[0]
            call_kwargs = mock_read.call_args
            start = call_kwargs.kwargs.get("second", call_kwargs[1].get("second"))
            if start is None:
                start = call_kwargs[0][1]
            assert start == pytest.approx(5.0), (
                f"Expected no jitter (start=5.0) but got {start}"
            )


# ===========================================================================
# pos_weight tests
# ===========================================================================


class TestPosWeight:
    """Tests for VideoWindowDataset.get_pos_weight()."""

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_balanced(self, mock_pool_cls):
        """Balanced labels should yield pos_weight ~1.0."""
        mock_pool_cls.return_value = MagicMock()
        specs = [_make_spec(label=i % 2, second=i) for i in range(100)]
        ds = VideoWindowDataset(specs, _make_config())
        assert ds.get_pos_weight() == pytest.approx(1.0)

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_imbalanced(self, mock_pool_cls):
        """pos_weight = neg_count / pos_count."""
        mock_pool_cls.return_value = MagicMock()
        specs = [_make_spec(label=0, second=i) for i in range(90)] + [
            _make_spec(label=1, second=i) for i in range(10)
        ]
        ds = VideoWindowDataset(specs, _make_config())
        # 90/10 = 9.0
        assert ds.get_pos_weight() == pytest.approx(9.0)

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_clamped_high(self, mock_pool_cls):
        """pos_weight should be clamped to the upper bound."""
        mock_pool_cls.return_value = MagicMock()
        specs = [_make_spec(label=0, second=i) for i in range(999)] + [
            _make_spec(label=1, second=0)
        ]
        ds = VideoWindowDataset(specs, _make_config())
        # 999/1 = 999 but default clamp is (0.1, 10.0) -> 10.0
        assert ds.get_pos_weight() == pytest.approx(10.0)

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_no_positives(self, mock_pool_cls):
        """With no positive samples pos_weight should fall back to 1.0."""
        mock_pool_cls.return_value = MagicMock()
        specs = [_make_spec(label=0, second=i) for i in range(50)]
        ds = VideoWindowDataset(specs, _make_config())
        assert ds.get_pos_weight() == pytest.approx(1.0)


# ===========================================================================
# MultiModalDatasetConfig tests
# ===========================================================================


class TestMultiModalDatasetConfig:
    """Tests for the multi-modal config dataclass defaults."""

    def test_defaults(self):
        """Verify default field values for MultiModalDatasetConfig."""
        from synchronai.data.multimodal.dataset_mm import MultiModalDatasetConfig

        cfg = MultiModalDatasetConfig(labels_file="/tmp/mm_labels.csv")
        assert cfg.sample_fps == 12.0
        assert cfg.window_seconds == 2.0
        assert cfg.frame_size == 640
        assert cfg.video_augment is False
        assert cfg.audio_augment is False
        assert cfg.sample_rate == 16000
        assert cfg.audio_chunk_duration == 1.0
        assert cfg.verify_alignment is True
        assert cfg.cache_audio is False

    def test_override(self):
        """Overriding individual fields should work."""
        from synchronai.data.multimodal.dataset_mm import MultiModalDatasetConfig

        cfg = MultiModalDatasetConfig(
            labels_file="/data/labels.csv",
            sample_fps=6.0,
            video_augment=True,
            sample_rate=22050,
        )
        assert cfg.sample_fps == 6.0
        assert cfg.video_augment is True
        assert cfg.sample_rate == 22050


# ===========================================================================
# create_multimodal_splits tests
# ===========================================================================


class TestCreateMultimodalSplits:
    """Tests for create_multimodal_splits config immutability.

    Full integration is avoided because it requires video/audio file access.
    Instead we test the key contract: dataclasses.replace() does not mutate
    the caller's config objects.
    """

    def test_replace_does_not_mutate_original(self):
        """dataclasses.replace() -- which create_multimodal_splits uses --
        should create a copy, leaving the original unchanged.
        """
        original = _make_config(augment=False)
        modified = replace(original, augment=True, labels_file="/new/path.csv")

        assert original.augment is False
        assert original.labels_file == "/tmp/labels.csv"
        assert modified.augment is True
        assert modified.labels_file == "/new/path.csv"

    def test_replace_preserves_other_fields(self):
        """Fields not overridden should carry over from the source config."""
        original = _make_config(
            sample_fps=6.0, frame_size=320, temporal_jitter_frames=4
        )
        modified = replace(original, augment=True)

        assert modified.sample_fps == original.sample_fps
        assert modified.frame_size == original.frame_size
        assert modified.temporal_jitter_frames == original.temporal_jitter_frames


# ===========================================================================
# Edge-case / regression tests
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    def test_split_single_video(self):
        """With only one video, val should contain that single group."""
        specs = _build_specs_for_split(n_videos=1, windows_per_video=10)
        train, val = split_by_video(specs, val_split=0.5, seed=42)
        # n_val = max(1, int(1 * 0.5)) = 1, so the single group goes to val
        assert len(val) == 10
        assert len(train) == 0

    def test_split_two_videos(self):
        """With two videos and 50% split, exactly one video goes to each split."""
        specs = _build_specs_for_split(n_videos=2, windows_per_video=5)
        train, val = split_by_video(specs, val_split=0.5, seed=42)
        assert len(train) == 5
        assert len(val) == 5

    @patch("synchronai.data.video.dataset.VideoReaderPool")
    def test_dataset_len_matches_specs(self, mock_pool_cls):
        """__len__ should match the number of specs provided."""
        mock_pool_cls.return_value = MagicMock()
        specs = _build_specs_for_split(n_videos=3, windows_per_video=7)
        ds = VideoWindowDataset(specs, _make_config())
        assert len(ds) == 21

    def test_n_frames_zero_window(self):
        """Zero-length window should yield 0 frames."""
        spec = _make_spec(sample_fps=12.0, window_seconds=0.0)
        assert spec.n_frames == 0

    def test_spec_equality(self):
        """Two specs with the same fields should be equal (frozen dataclass)."""
        a = _make_spec(video_path="/a.mp4", second=3, label=1, subject_id="X")
        b = _make_spec(video_path="/a.mp4", second=3, label=1, subject_id="X")
        assert a == b

    def test_spec_inequality(self):
        """Specs that differ in any field should not be equal."""
        a = _make_spec(second=1)
        b = _make_spec(second=2)
        assert a != b
