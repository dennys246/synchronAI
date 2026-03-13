"""
CLI entrypoint for synchronAI.

Currently implemented:
- fNIRS diffusion training (U-Net DDPM baseline)
- fNIRS diffusion generation (120s hemoglobin windows)
- Video classification training (YOLO backbone with temporal aggregation)
- Video classification inference
- Audio classification inference (Whisper encoder-based)
- Raw data preprocessing (xlsx labels to CSV)
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import os
import signal
import sys
from pathlib import Path
from typing import List, Optional

from synchronai.utils.logging import get_logger, setup_logging
from synchronai.utils.trace import trace


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="synchronai", description="synchronAI CLI")

    # Modality selection
    parser.add_argument("--modality", choices=["fnirs", "video", "audio"], default=None)
    parser.add_argument("--fnirs", action="store_true", help="Shortcut for --modality fnirs.")
    parser.add_argument("--video", action="store_true", help="Shortcut for --modality video.")
    parser.add_argument("--audio", action="store_true", help="Shortcut for --modality audio.")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument("--mode", choices=["train", "generate", "predict", "preprocess", "validate"])
    mode_group.add_argument(
        "--train",
        choices=["diffusion", "classifier", "audio-classifier"],
        metavar="MODEL",
        help="Shortcut for --mode train with a model name.",
    )
    mode_group.add_argument(
        "--generate",
        choices=["diffusion"],
        metavar="MODEL",
        help="Shortcut for --mode generate with a model name.",
    )
    mode_group.add_argument(
        "--predict",
        choices=["classifier"],
        metavar="MODEL",
        help="Shortcut for --mode predict with a model name.",
    )
    mode_group.add_argument(
        "--heatmap",
        action="store_true",
        help="Generate heatmap visualizations from video synchrony classifier.",
    )
    mode_group.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess raw data (xlsx labels to CSV).",
    )
    mode_group.add_argument(
        "--validate",
        action="store_true",
        help="Validate dataset before training.",
    )

    # Common arguments
    parser.add_argument("--save-dir", default="runs/experiment", help="Output directory for configs/checkpoints.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (INFO, DEBUG, ...).")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print trace checkpoints to stderr for segfault debugging.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable extra logging for debugging.")
    parser.add_argument(
        "--disable-numba",
        action="store_true",
        help="Disable Numba JIT (workaround for llvmlite segfaults).",
    )

    # Device
    parser.add_argument("--device", choices=["auto", "cpu", "gpu", "cuda"], default="auto")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # ========================
    # Preprocessing arguments
    # ========================
    parser.add_argument("--label-dir", help="Directory containing label xlsx files.")
    parser.add_argument("--video-dir", help="Directory containing video files.")
    parser.add_argument("--output-csv", default="data/labels.csv", help="Output path for labels CSV.")
    parser.add_argument(
        "--conflict-strategy",
        choices=["last", "first", "error"],
        default="last",
        help="How to handle conflicting labels.",
    )
    parser.add_argument(
        "--label-encoding",
        default="a:0,s:1",
        help="Label encoding as key:value pairs (e.g., 'a:0,s:1').",
    )

    # ========================
    # Video training arguments
    # ========================
    parser.add_argument("--labels-file", help="Path to labels.csv for training.")
    parser.add_argument("--video-path", help="Path to video file for inference.")

    # Model configuration
    parser.add_argument("--backbone", default="yolo11n", help="YOLO backbone model.")
    parser.add_argument("--backbone-task", choices=["detect", "pose"], default="detect")
    parser.add_argument(
        "--temporal-aggregation",
        choices=["mean", "max", "attention", "lstm"],
        default="attention",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training configuration
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--use-amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--num-workers", type=int, default=4)

    # Data configuration
    parser.add_argument("--sample-fps", type=float, default=12.0, help="Target FPS for frame sampling.")
    parser.add_argument("--frame-size", type=int, default=640, help="Frame size for YOLO preprocessing.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction.")
    parser.add_argument(
        "--group-by",
        choices=["video_path", "subject_id"],
        default="subject_id",
        help="Column to group by for train/val split.",
    )

    # Fine-tuning
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs for head-only training.")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Learning rate for backbone in stage 2.")

    # Heatmap generation during training
    parser.add_argument(
        "--heatmap-batch-interval",
        type=int,
        default=10,
        help="Generate heatmaps every N batches during training (0 = disabled).",
    )
    parser.add_argument(
        "--batch-plot-interval",
        type=int,
        default=10,
        help="Update batch_progress.png and history.json every N batches (0 = disabled).",
    )
    parser.add_argument(
        "--heatmap-video",
        help="Path to sample video for heatmap generation during training.",
    )
    parser.add_argument(
        "--heatmap-use-gradcam",
        action="store_true",
        default=True,
        help="Generate Grad-CAM spatial heatmaps during training (default: True).",
    )
    parser.add_argument(
        "--no-heatmap-gradcam",
        action="store_true",
        help="Disable Grad-CAM spatial heatmaps during training.",
    )
    parser.add_argument(
        "--heatmap-gradcam-aggregate",
        choices=["max", "mean", "weighted"],
        default="max",
        help="How to aggregate Grad-CAM across frames during training.",
    )

    # Inference
    parser.add_argument("--weights-path", help="Path to model weights for inference.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")

    # Heatmap generation
    parser.add_argument(
        "--visualizations",
        nargs="+",
        default=["all"],
        help="Visualizations to generate: timeline, grid, distribution, segments, data, video, gradcam, thumbnails, gradcam_thumbnails, all, all_with_video, all_with_gradcam",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.25,
        help="Transparency of video overlay (0-1).",
    )
    parser.add_argument(
        "--colormap",
        default="RdYlGn",
        help="Matplotlib colormap for heatmaps.",
    )
    parser.add_argument(
        "--use-gradcam",
        action="store_true",
        help="Use Grad-CAM spatial heatmaps for video/thumbnail overlays.",
    )
    parser.add_argument(
        "--gradcam-aggregate",
        choices=["max", "mean", "weighted"],
        default="max",
        help="How to aggregate Grad-CAM across frames in a window.",
    )

    # ========================
    # fNIRS arguments (existing)
    # ========================
    parser.add_argument("--xla", action="store_true", help="Enable XLA JIT in TensorFlow.")
    parser.add_argument("--data-dir", help="BIDS root or a single fNIRS recording path.")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--sfreq-hz", type=float, default=None, help="Optionally resample to a fixed Hz.")
    parser.add_argument("--segments-per-recording", type=int, default=4)
    parser.add_argument("--max-recordings", type=int, default=0)
    parser.add_argument("--recordings-per-batch", type=int, default=4)
    parser.add_argument("--deconvolution", action="store_true")
    parser.add_argument("--signal-type", default="hemodynamic", choices=["hemodynamic", "neural"])
    parser.add_argument("--diffusion-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--unet-base-width", type=int, default=32)
    parser.add_argument("--unet-depth", type=int, default=3)
    parser.add_argument("--unet-time-embed-dim", type=int, default=128)
    parser.add_argument("--unet-dropout", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of recordings held out for validation (0 to disable).")
    parser.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine_restarts"],
                        help="LR schedule: constant or cosine decay with warm restarts.")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--log-every", type=int, default=25)

    # ========================
    # Audio arguments
    # ========================
    parser.add_argument("--audio-path", help="Path to audio or video file for audio inference.")
    parser.add_argument("--audio-dir", help="Directory of audio/video files for batch inference.")
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size for audio feature extraction.",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json", "both"],
        default="both",
        help="Output format for audio predictions.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have output (for batch processing).",
    )
    parser.add_argument(
        "--vocalization-threshold",
        type=float,
        default=0.5,
        help="Threshold for vocalization detection.",
    )
    parser.add_argument(
        "--audio-labels-file",
        help="Path to audio labels CSV for training.",
    )
    parser.add_argument(
        "--resume-from",
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement).",
    )

    return parser.parse_args(argv)


def _normalize_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    # Determine modality
    modality_flags = [args.fnirs, args.video, getattr(args, "audio", False)]
    if sum(modality_flags) > 1:
        raise ValueError("Cannot specify multiple modality flags (--fnirs, --video, --audio).")
    if args.fnirs:
        args.modality = "fnirs"
    elif args.video:
        args.modality = "video"
    elif getattr(args, "audio", False):
        args.modality = "audio"
    elif args.preprocess:
        args.modality = "preprocessing"
        args.mode = "preprocess"
    elif args.validate:
        args.modality = "video"
        args.mode = "validate"
    elif not args.modality:
        # Default based on other args
        if getattr(args, "audio_path", None) or getattr(args, "audio_dir", None):
            args.modality = "audio"
        elif args.label_dir or args.labels_file or args.video_path:
            args.modality = "video"
        else:
            args.modality = "fnirs"

    # Determine mode
    if args.train:
        args.mode = "train"
        args.architecture = args.train
        # Set modality based on architecture
        if args.train == "audio-classifier":
            args.modality = "audio"
    elif args.generate:
        args.mode = "generate"
        args.architecture = args.generate
    elif args.predict:
        args.mode = "predict"
        args.architecture = args.predict
    elif args.heatmap:
        args.mode = "heatmap"
        args.architecture = "classifier"
        args.modality = "video"
    elif args.preprocess:
        args.mode = "preprocess"
        args.architecture = None
    elif args.validate:
        args.mode = "validate"
        args.architecture = None
    elif args.mode:
        if args.modality == "audio":
            args.architecture = "audio-classifier"
        elif args.modality == "video":
            args.architecture = "classifier"
        else:
            args.architecture = "diffusion"
    elif args.modality == "audio":
        # Default to predict mode for audio
        args.mode = "predict"
        args.architecture = "audio-classifier"
    else:
        raise ValueError("Specify --mode or use --train/--generate/--predict/--preprocess/--heatmap shortcuts.")

    # Parse label encoding
    if args.label_encoding:
        encoding = {}
        for pair in args.label_encoding.split(","):
            key, value = pair.split(":")
            encoding[key.strip()] = int(value.strip())
        args.label_encoding_dict = encoding
    else:
        args.label_encoding_dict = {"a": 0, "s": 1}

    # Convert 0 to None for unlimited recordings
    if hasattr(args, "max_recordings") and args.max_recordings == 0:
        args.max_recordings = None

    return args


def _run_preprocessing(args: argparse.Namespace) -> None:
    """Run raw data preprocessing."""
    from synchronai.data.preprocessing.raw_to_csv import (
        RawDataConfig,
        preprocess_raw_to_csv,
        print_preprocessing_report,
    )

    if not args.label_dir:
        raise ValueError("--label-dir is required for preprocessing")
    if not args.video_dir:
        raise ValueError("--video-dir is required for preprocessing")

    config = RawDataConfig(
        label_dir=Path(args.label_dir),
        video_dir=Path(args.video_dir),
        output_csv=Path(args.output_csv),
        label_encoding=args.label_encoding_dict,
        conflict_strategy=args.conflict_strategy,
    )

    logger = get_logger(__name__)
    logger.info("Starting preprocessing...")
    logger.info(f"  Label directory: {config.label_dir}")
    logger.info(f"  Video directory: {config.video_dir}")
    logger.info(f"  Output CSV: {config.output_csv}")

    df, report = preprocess_raw_to_csv(config)
    print_preprocessing_report(report)


def _run_video_training(args: argparse.Namespace) -> None:
    """Run video classifier training."""
    from synchronai.models.cv.video_classifier import VideoClassifierConfig
    from synchronai.training.video.train import TrainingConfig, train_video_classifier
    from synchronai.data.video.dataset import VideoDatasetConfig

    if not args.labels_file:
        raise ValueError("--labels-file is required for training")

    model_config = VideoClassifierConfig(
        window_seconds=1.0,
        sample_fps=args.sample_fps,
        frame_height=args.frame_size,
        frame_width=args.frame_size,
        backbone=args.backbone,
        backbone_task=args.backbone_task,
        temporal_aggregation=args.temporal_aggregation,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        labels_file=args.labels_file,
        val_split=args.val_split,
        group_by=args.group_by,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_amp=args.use_amp,
        seed=args.seed,
        num_workers=args.num_workers,
        stage1_epochs=args.stage1_epochs,
        backbone_lr=args.backbone_lr,
        heatmap_batch_interval=args.heatmap_batch_interval,
        heatmap_video_path=args.heatmap_video,
        heatmap_use_gradcam=getattr(args, "heatmap_use_gradcam", True) and not getattr(args, "no_heatmap_gradcam", False),
        heatmap_gradcam_aggregate=getattr(args, "heatmap_gradcam_aggregate", "max"),
        batch_plot_interval=args.batch_plot_interval,
    )

    data_config = VideoDatasetConfig(
        labels_file=args.labels_file,
        sample_fps=args.sample_fps,
        frame_size=args.frame_size,
    )

    logger = get_logger(__name__)
    logger.info("Starting video classifier training...")
    logger.info(f"  Labels file: {args.labels_file}")
    logger.info(f"  Save directory: {args.save_dir}")
    logger.info(f"  Backbone: {args.backbone} ({args.backbone_task})")
    logger.info(f"  Temporal aggregation: {args.temporal_aggregation}")

    resume_from = getattr(args, "resume_from", None)
    if resume_from:
        logger.info(f"  Resuming from: {resume_from}")

    model, history = train_video_classifier(
        labels_file=args.labels_file,
        save_dir=args.save_dir,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        resume_from=resume_from,
    )

    logger.info(f"Training complete. Best AUC: {history.best_val_auc:.4f}")
    logger.info(f"Training plot saved to: {args.save_dir}/training_plot.png")


def _run_video_inference(args: argparse.Namespace) -> None:
    """Run video classifier inference."""
    from synchronai.inference.video.predict import predict_video

    if not args.video_path:
        raise ValueError("--video-path is required for inference")
    if not args.weights_path:
        raise ValueError("--weights-path is required for inference")

    logger = get_logger(__name__)
    logger.info("Running video classification inference...")
    logger.info(f"  Video: {args.video_path}")
    logger.info(f"  Weights: {args.weights_path}")

    results = predict_video(
        video_path=args.video_path,
        weights_path=args.weights_path,
        threshold=args.threshold,
        device=args.device if args.device != "auto" else None,
    )

    logger.info(f"Prediction: {results['prediction']} (probability: {results['probability']:.4f})")
    return results


def _run_video_heatmap(args: argparse.Namespace) -> None:
    """Generate heatmap visualizations from video synchrony classifier."""
    from synchronai.inference.video.heatmap import generate_video_heatmap
    from synchronai.utils.heatmap import HeatmapConfig

    if not args.video_path:
        raise ValueError("--video-path is required for heatmap generation")
    if not args.weights_path:
        raise ValueError("--weights-path is required for heatmap generation")

    logger = get_logger(__name__)
    logger.info("Generating video synchrony heatmaps...")
    logger.info(f"  Video: {args.video_path}")
    logger.info(f"  Weights: {args.weights_path}")
    logger.info(f"  Output directory: {args.save_dir}")
    logger.info(f"  Visualizations: {args.visualizations}")

    use_gradcam = getattr(args, "use_gradcam", False)
    gradcam_aggregate = getattr(args, "gradcam_aggregate", "max")

    if use_gradcam:
        logger.info("  Grad-CAM spatial heatmaps: ENABLED")
        logger.info(f"  Grad-CAM aggregation: {gradcam_aggregate}")

    # Create heatmap config
    heatmap_config = HeatmapConfig(
        colormap=args.colormap,
        threshold=args.threshold,
    )

    result = generate_video_heatmap(
        video_path=args.video_path,
        weights_path=args.weights_path,
        output_dir=args.save_dir,
        threshold=args.threshold,
        device=args.device if args.device != "auto" else None,
        heatmap_config=heatmap_config,
        visualizations=args.visualizations,
        overlay_alpha=args.overlay_alpha,
        use_gradcam=use_gradcam,
        gradcam_aggregate=gradcam_aggregate,
    )

    logger.info("Heatmap generation complete!")
    logger.info(f"  Synchrony ratio: {result.prediction_result.synchrony_ratio:.1%}")
    logger.info(f"  Overall probability: {result.prediction_result.overall_probability:.3f}")
    logger.info("  Generated files:")
    for viz_type, path in result.output_files.items():
        logger.info(f"    - {viz_type}: {path}")

    return result


def _run_fnirs_training(args: argparse.Namespace) -> None:
    """Run fNIRS diffusion training."""
    from synchronai.training.diffusion.train import train_fnirs_diffusion

    if not args.data_dir:
        raise ValueError("--data-dir is required for --mode train")

    train_fnirs_diffusion(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        duration_seconds=args.duration_seconds,
        target_sfreq_hz=args.sfreq_hz,
        segments_per_recording=args.segments_per_recording,
        max_recordings=args.max_recordings,
        recordings_per_batch=args.recordings_per_batch,
        diffusion_timesteps=args.diffusion_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        unet_base_width=args.unet_base_width,
        unet_depth=args.unet_depth,
        unet_time_embed_dim=args.unet_time_embed_dim,
        unet_dropout=args.unet_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        deconvolution=args.deconvolution,
        signal_type=args.signal_type,
        val_fraction=args.val_fraction,
        lr_schedule=args.lr_schedule,
    )


def _run_fnirs_generation(args: argparse.Namespace) -> None:
    """Run fNIRS diffusion generation."""
    from synchronai.inference.fnirs.generate import generate_fnirs_diffusion

    generate_fnirs_diffusion(
        save_dir=args.save_dir,
        n_samples=args.n_samples,
        config_path=args.config_path,
        weights_path=args.weights_path,
        seed=args.seed,
        log_every=args.log_every,
        out_path=args.out_path,
    )


def _run_audio_inference(args: argparse.Namespace) -> None:
    """Run audio classifier inference."""
    from synchronai.inference.audio.predict import (
        predict_audio,
        predict_audio_batch,
        export_predictions_csv,
        export_predictions_json,
    )
    from pathlib import Path

    logger = get_logger(__name__)

    if not args.weights_path:
        raise ValueError("--weights-path is required for audio inference")

    device = args.device if args.device != "auto" else None

    # Single file inference
    if args.audio_path:
        logger.info("Running audio classification inference...")
        logger.info(f"  Audio/Video: {args.audio_path}")
        logger.info(f"  Weights: {args.weights_path}")

        result = predict_audio(
            input_path=args.audio_path,
            weights_path=args.weights_path,
            device=device,
            vocalization_threshold=args.vocalization_threshold,
        )

        # Export results
        output_dir = Path(args.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_stem = Path(args.audio_path).stem
        if args.output_format in ["csv", "both"]:
            export_predictions_csv(result, output_dir / f"{input_stem}_audio.csv")
        if args.output_format in ["json", "both"]:
            export_predictions_json(result, output_dir / f"{input_stem}_audio.json")

        logger.info(f"Vocalization ratio: {result.vocalization_ratio:.1%}")
        logger.info(f"Speech ratio: {result.speech_ratio:.1%}")
        logger.info(f"Dominant event: {result.dominant_event}")
        return result

    # Batch inference
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # Find all audio/video files (recursive search)
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac"]
        input_paths = []
        for ext in extensions:
            input_paths.extend(audio_dir.glob(f"**/*{ext}"))
            input_paths.extend(audio_dir.glob(f"**/*{ext.upper()}"))

        if not input_paths:
            raise ValueError(f"No audio/video files found in {audio_dir}")

        logger.info(f"Running batch audio classification on {len(input_paths)} files...")
        logger.info(f"  Input directory: {audio_dir}")
        logger.info(f"  Output directory: {args.save_dir}")

        results = predict_audio_batch(
            input_paths=input_paths,
            weights_path=args.weights_path,
            output_dir=args.save_dir,
            device=device,
            skip_existing=args.skip_existing,
            output_format=args.output_format,
        )

        logger.info(f"Batch processing complete: {len(results)} files processed")
        return results

    else:
        raise ValueError("--audio-path or --audio-dir is required for audio inference")


def _run_audio_training(args: argparse.Namespace) -> None:
    """Run audio classifier training."""
    from synchronai.training.audio.train import (
        AudioTrainingConfig,
        train_audio_classifier,
    )
    from synchronai.models.audio.audio_classifier import AudioClassifierConfig

    logger = get_logger(__name__)

    # Determine labels file
    labels_file = getattr(args, "audio_labels_file", None) or args.labels_file
    if not labels_file:
        raise ValueError("--audio-labels-file or --labels-file is required for audio training")

    # Model config
    model_config = AudioClassifierConfig(
        whisper_model_size=args.whisper_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    # Training config
    training_config = AudioTrainingConfig(
        labels_file=labels_file,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_amp=args.use_amp,
        num_workers=args.num_workers,
        seed=args.seed,
        early_stopping_patience=getattr(args, "early_stopping_patience", 15),
    )

    logger.info("Starting audio classifier training...")
    logger.info(f"  Labels file: {labels_file}")
    logger.info(f"  Save directory: {args.save_dir}")
    logger.info(f"  Whisper model: {args.whisper_model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")

    model, history = train_audio_classifier(
        labels_file=labels_file,
        save_dir=args.save_dir,
        model_config=model_config,
        training_config=training_config,
        resume_from=getattr(args, "resume_from", None),
    )

    logger.info(f"Training complete!")
    logger.info(f"  Best validation loss: {history.best_val_loss:.4f}")
    logger.info(f"  Best validation accuracy: {history.best_val_acc:.2%}")
    logger.info(f"  Best epoch: {history.best_epoch + 1}")
    logger.info(f"  Model saved to: {args.save_dir}/best.pt")


def _configure_runtime(args: argparse.Namespace) -> None:
    """Configure runtime environment."""
    trace(f"Runtime config: device={args.device}")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        trace("Set CUDA_VISIBLE_DEVICES=-1")

    # Import torch early to configure CUDA
    if args.modality == "video":
        try:
            import torch
            if args.device == "auto" and torch.cuda.is_available():
                trace(f"CUDA available: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass


def _setup_enhanced_crash_detection(trace_enabled: bool) -> None:
    """Set up crash detection facilities."""
    if not trace_enabled:
        return

    trace("Setting up enhanced crash detection")

    if not faulthandler.is_enabled():
        faulthandler.enable(all_threads=True)
        trace("Enabled faulthandler for all threads")

    try:
        if hasattr(signal, "SIGUSR1"):
            faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
            trace("Registered faulthandler for SIGUSR1")
    except (ValueError, OSError) as e:
        trace(f"Could not register SIGUSR1 handler: {e}")

    import platform
    trace(f"Python version: {sys.version}")
    trace(f"Platform: {platform.platform()}")


def main(argv: Optional[List[str]] = None) -> None:
    args = _normalize_cli_args(_parse_args(argv))

    trace_env = os.environ.get("SYNCHRONAI_TRACE", "").lower()
    trace_enabled = args.trace or args.verbose or trace_env in {"1", "true", "yes", "on"}
    if trace_enabled:
        os.environ.setdefault("SYNCHRONAI_TRACE", "1")

    _setup_enhanced_crash_detection(trace_enabled)
    trace("Parsed CLI arguments")

    if args.disable_numba:
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        os.environ.setdefault("MNE_USE_NUMBA", "0")
        trace("Disabled Numba JIT")

    log_level = args.log_level
    tf_cpp_min_log_level = "2"
    if args.verbose:
        log_level = "DEBUG"
        tf_cpp_min_log_level = "0"
        os.environ.setdefault("SYNCHRONAI_VERBOSE", "1")

    setup_logging(log_level, tf_cpp_min_log_level=tf_cpp_min_log_level)
    logger = get_logger(__name__)

    os.makedirs(args.save_dir, exist_ok=True)

    _configure_runtime(args)

    # Route to appropriate handler
    if args.mode == "preprocess":
        _run_preprocessing(args)
        return

    if args.modality == "video":
        if args.mode == "train":
            _run_video_training(args)
        elif args.mode == "predict":
            _run_video_inference(args)
        elif args.mode == "heatmap":
            _run_video_heatmap(args)
        elif args.mode == "validate":
            from synchronai.data.video.validation import (
                print_validation_report,
                validate_dataset,
            )

            if not args.labels_file:
                raise ValueError("--labels-file is required for validation")

            logger.info(f"Validating dataset: {args.labels_file}")
            result = validate_dataset(args.labels_file)
            print_validation_report(result)

            if not result.valid:
                sys.exit(1)
        else:
            raise ValueError(f"Unsupported mode for video: {args.mode}")
        return

    if args.modality == "audio":
        if args.mode == "train":
            _run_audio_training(args)
        elif args.mode == "predict":
            _run_audio_inference(args)
        else:
            raise ValueError(f"Unsupported mode for audio: {args.mode}. Use --mode train or --mode predict.")
        return

    if args.modality == "fnirs":
        # Import TensorFlow for fNIRS
        if hasattr(args, "xla") and args.xla:
            trace("Enabling TensorFlow XLA JIT")
            import tensorflow as tf
            tf.config.optimizer.set_jit(True)

        if args.mode == "train":
            _run_fnirs_training(args)
        elif args.mode == "generate":
            _run_fnirs_generation(args)
        else:
            raise ValueError(f"Unsupported mode for fNIRS: {args.mode}")
        return

    raise ValueError(f"Unsupported modality: {args.modality}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n[FATAL ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("\n" + "=" * 70, file=sys.stderr)
        print("CRASH REPORT", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"Error Type: {type(e).__name__}", file=sys.stderr)
        print(f"Error Message: {e}", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(1)
