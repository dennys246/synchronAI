# fNIRS Feature Storage Format

How pre-extracted fNIRS encoder features are stored on disk, and why.

## Two formats

Feature directories can exist in either of two layouts. The training code
auto-detects which format is present.

### Unpacked (legacy)

```
data/fnirs_perpair_<size>_features/
├── feature_index.csv        # one row per sample
└── features/
    ├── 50001_V0_fNIRS_000_S1_D1_<hash>.pt
    ├── 50001_V0_fNIRS_001_S1_D1_<hash>.pt
    └── ... (hundreds of thousands of tiny .pt files)
```

- One file per sample, each ~15–240 KB depending on model size.
- `__getitem__` does `torch.load(path)` — a fresh `open()` syscall per sample.
- **Unusable for training at scale on GPFS/NFS.** Each epoch requires
  500K+ network round trips; epochs take hours of wall-clock even with
  trivial compute.
- Still useful for resume/crash-recovery during extraction — partial
  progress is visible as files appear on disk.

### Packed (preferred)

```
data/fnirs_perpair_<size>_features/
├── feature_index.csv        # gains a `row_idx` column
├── features_packed.bin      # raw float32 array, shape (N, T, C), row-major
└── features_meta.json       # { "shape": [N, T, C], "dtype": "float32", ... }
```

- Single contiguous binary file. No individual `.pt` files.
- Loaded via `np.memmap(mode="r")` on the training side. The OS pages
  in rows on demand; hot rows stay in the page cache.
- Zero syscall overhead per sample — random access is pure page faults.
- First-epoch reads may page through the whole file; subsequent epochs
  are mostly served from cache if the working set fits in RAM.
- **Epochs go from hours to seconds.** This is the standard format.

## Creating the packed format

### Option A — pack during extraction (new extractions)

```bash
python scripts/extract_fnirs_features.py \
    --encoder-weights "$ENCODER_PT" \
    --data-dirs "$FNIRS_DIRS" \
    --output-dir "$FEATURE_DIR" \
    --qc-cache "$SYNCHRONAI_DIR/data/qc_tiers.csv" \
    --include-tiers "gold,standard,salvageable" \
    --pack-output \
    --delete-unpacked
```

The `--pack-output` flag calls `pack_features()` at the end of extraction.
`--delete-unpacked` removes the individual `.pt` files after successful
pack to save disk.

All three per-pair extraction BSub scripts pass these flags by default:
- `pre_fnirs_perpair_transfer_bsub.sh` (v8+)
- `pre_fnirs_extract_features_bsub.sh` (v4+)
- `pre_fnirs_extract_random_bsub.sh` (v3+)

### Option B — pack an existing unpacked dir

```bash
python scripts/pack_features.py \
    data/fnirs_perpair_<size>_features \
    --delete-unpacked
```

Loads the existing `.pt` files and writes `features_packed.bin` +
`features_meta.json`, appending a `row_idx` column to `feature_index.csv`.
Idempotent — safe to re-run.

## Implementation notes

### Shape uniformity

All features in a packed file must share the same tensor shape. The pack
function inspects the first sample to infer shape and dtype, then validates
every subsequent sample matches exactly. Mismatches raise `ValueError`
immediately rather than silently writing corrupt data.

For per-pair bottleneck extraction with a fixed model config, this
invariant holds automatically: every `(pair, window)` tuple goes through
the same encoder producing identical `(n_frames, feature_dim)` output.

### Worker-fork safety

`FnirsPackedFeatureDataset` lazy-opens the mmap on first `__getitem__`
access rather than in `__init__`. When `DataLoader` forks workers, each
worker opens its own `np.memmap` handle — avoiding shared-handle issues
and making the dataset safely picklable.

### Write path — sequential I/O, not mmap

`pack_features()` writes the binary file via `open(path, "wb")` and
sequential `file.write(arr.tobytes())` calls. It does NOT use
`np.memmap(mode="w+")` for writing.

This matters because the login node enforces a per-user address-space
ulimit. `np.memmap(mode="w+")` would reserve the full file's VA space
upfront, which fails for multi-GB outputs with `OSError: [Errno 12]
Cannot allocate memory`. Sequential writes have no such requirement.

The read path inside the training container (`np.memmap(mode="r")`) is
unaffected — read-mode mmap has different ulimit treatment and runs
inside Docker where limits are looser.

### Memory sizing

Packed file sizes scale with `N × T × C × 4` bytes:

| Model  | bottleneck_dim | ~N × T × C bytes | Total size |
|--------|---------------:|-----------------:|-----------:|
| micro  |             32 | 640K × 59 × 32 × 4 | ~4.6 GB  |
| small  |            128 | 640K × 59 × 128 × 4 | ~18 GB  |
| medium |            256 | 640K × 59 × 256 × 4 | ~37 GB  |
| large  |            512 | 640K × 59 × 512 × 4 | ~74 GB  |

The training jobs request 16 GB RAM. With `np.memmap`, this is the
working-set size, not the total-file size — the OS keeps hot pages
in cache and evicts cold ones. Larger models see more page churn but
no OOM. If epoch times become I/O-bound again for `large`, bump the
job's RAM to 32 GB or shard the file.

## Related files

- `src/synchronai/data/fnirs/feature_dataset.py` — `FnirsFeatureDataset`,
  `FnirsPackedFeatureDataset`, `pack_features()`, `is_feature_dir_packed()`
- `scripts/pack_features.py` — standalone CLI
- `scripts/extract_fnirs_features.py` — `--pack-output` flag
- `scripts/train_fnirs_from_features.py` — auto-detects packed format
