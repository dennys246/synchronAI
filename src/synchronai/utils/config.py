import json, atexit, copy, os
from glob import glob
from pathlib import Path
from typing import Optional

DEFAULT_SAVE_ROOT = Path("keras") / "snowgan"


def _normalize_save_dir(path: Optional[str]) -> str:
    """
    Normalize save_dir inputs so downstream string concatenation keeps working.
    Always returns a POSIX-style path ending with '/' (many callers expect it).
    """
    if not path:
        path = DEFAULT_SAVE_ROOT.as_posix()
    save_dir = Path(path).as_posix()
    if not save_dir.endswith("/"):
        save_dir = f"{save_dir}/"
    return save_dir


def _normalize_checkpoint(save_dir: str, checkpoint: Optional[str], default_filename: str) -> str:
    """
    Resolve a checkpoint path so that it respects the configured save_dir while
    still honouring explicit absolute paths or already-resolved relatives.
    """
    save_dir_path = Path(save_dir)
    if not checkpoint:
        return (save_dir_path / default_filename).as_posix()

    checkpoint_path = Path(checkpoint)

    # Leave explicit absolute paths untouched.
    if checkpoint_path.is_absolute():
        return checkpoint_path.as_posix()

    # If the checkpoint already exists relative to the current working dir, keep it.
    if checkpoint_path.exists():
        return checkpoint_path.as_posix()

    # If checkpoint already points inside save_dir, keep relative layout.
    try:
        relative = checkpoint_path.relative_to(save_dir_path)
        return (save_dir_path / relative).as_posix()
    except ValueError:
        pass

    # Detect legacy defaults (keras/snowgan/...) and relocate them beneath save_dir.
    try:
        legacy_relative = checkpoint_path.relative_to(DEFAULT_SAVE_ROOT)
        return (save_dir_path / legacy_relative).as_posix()
    except ValueError:
        pass  

    # Respect user-provided relative prefixes like "./" or "../" by leaving them alone.
    first_segment = checkpoint_path.parts[0] if checkpoint_path.parts else ""
    if first_segment in (".", ".."):
        return checkpoint_path.as_posix()

    # Otherwise treat the checkpoint as relative to save_dir.
    return (save_dir_path / checkpoint_path).as_posix()

config_template = {
            "save_dir": "keras/fnirs_diffusion/",
            "checkpoint": "keras/snowgan/diffusion.keras",
            "dataset": "CARE",
            "datatype": "fnirs",
            "architecture": "diffusion",
            "resolution": [1024, 1024],
            "channels": 3,
            "depth": 1,
            "images": None,
            "trained_pool": None,
            "validation_pool": None,
            "test_pool": None,
            "model_history": None,
            "n_samples": 10,
            "epochs": 10,
            "current_epoch": 0,
            "batch_size": 2,
            "training_steps": 2,
            "learning_rate": 1e-5,
            "beta_1": 0.5,
            "beta_2": 0.9,
            "negative_slope": 0.25,
            "lambda_gp": 10.0,
            "latent_dim": 256,
            "convolution_depth": 5,
            "filter_counts": [64, 128, 256, 512, 1024],
            "kernel_size": [3, 3],
            "kernel_stride": [2, 2],
            "batch_norm": False,
            "final_activation": "tanh",
            "zero_padding": None,
            "padding": "same",
            "optimizer": "adam",
            "loss": None,
            "train_ind": 0,
            "trained_data": [],
            "seen_profiles": [],
            "rebuild": False,
            "fade": False,
            "fade_steps": 50000,
            "fade_step": 0,
            "cleanup_milestone": 1000
}

class build:
    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        if os.path.exists(config_filepath): # Try and load config if folder passed in
            print(f"Loading config file: {self.config_filepath}")
            config_json = self.load_config(self.config_filepath)
        else:
            print("WARNING: Config not found, building from default template...")
            config_json = copy.deepcopy(config_template)

        # Backwards compatibility for new fields
        config_json.setdefault("seen_profiles", [])
        config_json.setdefault("channels", 3)
        config_json.setdefault("depth", 1)

        self.configure(**config_json) # Build configuration

        atexit.register(self.save_config)

        
    def __repr__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
    
    def save_config(self, config_filepath = None):
        # Save the config filepath if passed in
        if config_filepath: self.config_filepath = config_filepath

        # Ensure destination directory exists
        dest_dir = os.path.dirname(self.config_filepath) or "."
        os.makedirs(dest_dir, exist_ok=True)

        with open(self.config_filepath, 'w') as config_file:
            json.dump(self.dump(), config_file, indent = 4)
             
    def load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config_json = json.load(config_file)
        else:
            config_json = config_template
        return config_json

    def configure(self, save_dir, checkpoint, dataset, datatype, architecture, resolution, images, trained_pool, validation_pool, test_pool, model_history, n_samples, epochs, current_epoch, batch_size, training_steps, learning_rate, beta_1, beta_2, negative_slope, lambda_gp, latent_dim, convolution_depth, filter_counts, kernel_size, kernel_stride, batch_norm, final_activation, zero_padding, padding, optimizer, loss, train_ind, trained_data, rebuild, fade=False, fade_steps=10000, fade_step=0, cleanup_milestone=1000, seen_profiles=None, channels=3, depth=1):
		# Process lists
        if isinstance(filter_counts, str):
            filter_counts = [int(datum) for datum in filter_counts.split(' ')]

        if isinstance(kernel_size, str):
            kernel_size = [int(datum) for datum in kernel_size.split(' ')]

        if isinstance(kernel_stride, str):
            kernel_stride = [int(datum) for datum in kernel_stride.split(' ')]
        
        #-------------------------------- Model Set-Up -------------------------------#
        self.save_dir = _normalize_save_dir(save_dir)
        self.dataset = dataset or "dennys246/rocky_mountain_snowpack"
        self.datatype = datatype or "magnified_profile"
        self.architecture = architecture or "generator"
        self.resolution = resolution or (1024, 1024)
        self.channels = int(channels) if channels is not None else 3
        self.depth = int(depth) if depth is not None else 1
        self.images = images or None 
        self.trained_pool = trained_pool or None
        self.validation_pool = validation_pool or None
        self.test_pool = test_pool or None
        self.model_history = model_history or None
        self.n_samples = int(n_samples) or 10
        self.epochs = int(epochs) or 10
        self.current_epoch = int(current_epoch) or 0
        self.batch_size = int(batch_size) or 4
        self.training_steps = int(training_steps) or 5
        self.learning_rate = float(learning_rate) or 1e-4
        self.beta_1 = float(beta_1) or 0.5
        self.beta_2 = float(beta_2) or 0.9
        self.negative_slope = float(negative_slope) or 0.25
        self.lambda_gp = float(lambda_gp) or None
        self.latent_dim = int(latent_dim) or 100
        self.convolution_depth = int(convolution_depth) or 5
        self.filter_counts = filter_counts or [32, 64, 128, 256, 512]
        self.kernel_size = kernel_size or [5, 5]
        self.kernel_stride = kernel_stride or [2, 2]
        self.batch_norm = batch_norm or False
        self.final_activation = final_activation or "tanh"
        self.zero_padding = zero_padding or None
        self.padding = padding or "same"
        self.optimizer = optimizer or "adam"
        self.loss = loss or None
        self.train_ind = train_ind or 0
        self.trained_data = trained_data or []
        # Track seen profiles as a set in memory for fast membership checks; still serialized as list
        self.seen_profiles = set(seen_profiles or [])
        self.rebuild = rebuild or False
        # Progressive fade configuration and persisted progress
        self.fade = bool(fade)
        self.fade_steps = int(fade_steps) if fade_steps is not None else 10000
        self.fade_step = int(fade_step) if fade_step is not None else 0
        if cleanup_milestone is None:
            cleanup_value = 1000
        else:
            try:
                cleanup_value = int(cleanup_milestone)
            except (TypeError, ValueError):
                cleanup_value = 1000
        self.cleanup_milestone = max(0, cleanup_value)

        default_checkpoint_filename = "generator.keras" if self.architecture == "generator" else "discriminator.keras"
        self.checkpoint = _normalize_checkpoint(self.save_dir, checkpoint, default_checkpoint_filename)

    def dump(self):
        config = {
            "save_dir": self.save_dir,
            "checkpoint": self.checkpoint,
            "dataset": self.dataset,
            "datatype": self.datatype,
            "architecture": self.architecture,
            "resolution": self.resolution,
            "channels": self.channels,
            "depth": self.depth,
            "images": self.images,
            "trained_pool": self.trained_pool,
            "validation_pool": self.validation_pool,
            "test_pool": self.test_pool,
            "model_history": self.model_history,
            "n_samples": self.n_samples,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "batch_size": self.batch_size,
            "training_steps": self.training_steps,
            "learning_rate": self.learning_rate,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "negative_slope": self.negative_slope,
            "lambda_gp": self.lambda_gp,
            "latent_dim": self.latent_dim,
            "convolution_depth": self.convolution_depth,
            "filter_counts": self.filter_counts,
            "kernel_size": self.kernel_size,
            "kernel_stride": self.kernel_stride,
            "batch_norm": self.batch_norm,
            "final_activation":self.final_activation,
            "zero_padding": self.zero_padding,
            "padding": self.padding,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "train_ind": self.train_ind,
            "trained_data": self.trained_data,
            "seen_profiles": list(self.seen_profiles),
            "rebuild": self.rebuild,
            "fade": self.fade,
            "fade_steps": self.fade_steps,
            "fade_step": self.fade_step,
            "cleanup_milestone": self.cleanup_milestone
        }
        return config


def load_gen_config(config_filepath, config = None):
    # Configure the discriminator
    gen_config = build(config_filepath, config)

    if not os.path.exists(config_filepath):
        split = config_filepath.split("/")
        gen_config.save_dir = _normalize_save_dir(gen_config.save_dir or "/".join(split[:-1]))
        gen_config.checkpoint = _normalize_checkpoint(gen_config.save_dir, gen_config.checkpoint or "keras/snowgan/generator.keras", "generator.keras")
        gen_config.architecture = "generator"
    return gen_config 


def configure_gen(config, args):
    config = configure_generic(config, args)

    # Check if using default config
    if config.architecture == "discriminator":
        print(f"Setting Gen Default!")
        config.architecture = "generator"
        config.checkpoint = "keras/snowgan/generator.keras"
        config.training_steps = 3
        config.learning_rate = 1e-4

    if args.gen_checkpoint: config.checkpoint = args.gen_checkpoint
    if args.gen_kernel: config.kernel_size = [int(datum) for datum in args.gen_kernel.split(' ')]
    if args.gen_stride: config.kernel_stride = [int(datum) for datum in args.gen_stride.split(' ')]
    if args.gen_norm: config.batch_norm = args.gen_norm
    if args.gen_lr: config.learning_rate = args.gen_lr
    if args.gen_beta_1: config.beta_1 = args.gen_beta_1
    if args.gen_beta_2: config.beta_2 = args.gen_beta_2
    if args.gen_negative_slope: config.negative_slope = args.gen_negative_slope
    if args.gen_steps: config.training_steps = args.gen_steps
    if args.gen_filters: config.filter_counts = [int(datum) for datum in args.gen_filters.split(' ')]
    config.checkpoint = _normalize_checkpoint(config.save_dir, config.checkpoint, "generator.keras")
    return config
    

def load_disc_config(config_filepath, config = None):
    # Configure the discriminator
    disc_config = build(config_filepath, config)

    if not os.path.exists(config_filepath):
        split = config_filepath.split("/")
        disc_config.save_dir = _normalize_save_dir(disc_config.save_dir or "/".join(split[:-1]))
        disc_config.checkpoint = _normalize_checkpoint(disc_config.save_dir, disc_config.checkpoint or "keras/snowgan/discriminator.keras", "discriminator.keras")
        disc_config.architecture = "discriminator"
    return disc_config 


def configure_disc(config, args):
    config = configure_generic(config, args)

    if args.disc_checkpoint: config.checkpoint = args.disc_checkpoint
    if args.disc_kernel: config.kernel_size = [int(datum) for datum in args.disc_kernel.split(' ')]
    if args.disc_stride: config.kernel_stride = [int(datum) for datum in args.disc_stride.split(' ')]
    if args.disc_lr: config.learning_rate = args.disc_lr
    if args.disc_beta_1: config.beta_1 = args.disc_beta_1
    if args.disc_beta_2: config.beta_2 = args.disc_beta_2
    if args.disc_negative_slope: config.negative_slope = args.disc_negative_slope
    if args.disc_steps:
        config.training_steps = args.disc_steps
    else:
        # Favor a stronger discriminator at 1024x1024
        config.training_steps = config.training_steps or 5
    if args.disc_filters: config.filter_counts = [int(datum) for datum in args.disc_filters.split(' ')]
    if config.learning_rate is None or config.learning_rate == 0:
        config.learning_rate = 1e-4
    if config.lambda_gp is None:
        config.lambda_gp = 10.0
    config.checkpoint = _normalize_checkpoint(config.save_dir, config.checkpoint, "discriminator.keras")
    return config

def configure_generic(config, args):
    if args.save_dir: config.save_dir = _normalize_save_dir(args.save_dir)
    if getattr(args, "dataset_dir", None): config.dataset = args.dataset_dir
    if args.rebuild: config.rebuild = args.rebuild

    if args.resolution: config.resolution = args.resolution
    if args.n_samples: config.n_samples = args.n_samples
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.epochs = args.epochs
    if args.latent_dim: config.latent_dim = args.latent_dim
    # Progressive fade options
    if args.fade: config.fade = args.fade
    if args.fade_steps: config.fade_steps = args.fade_steps
    if getattr(args, "cleanup_milestone", None) is not None:
        config.cleanup_milestone = args.cleanup_milestone
    return config
