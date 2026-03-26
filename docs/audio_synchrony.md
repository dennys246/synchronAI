# Audio Synchrony: Second-by-Second Audio Classification

## Revision Notes

- **Dual-model architecture**: separate language and audio classifiers.
- **Language Classifier**: Uses Whisper transcripts as input to a text-based classifier.
- **Audio Classifier**: Uses Whisper encoder as a pretrained feature extractor for audio classification.
- Both produce per-second labels aligned with video synchrony timeline.

## Related Changes

- Update fNIRS generative log to `synchronai_fnirs_gen_DATE.log` (in `pre_generative_fnirs_bsub.sh`).

## Overview

This plan defines **two completely separate pipelines** for second-by-second audio analysis:

---

## Pipeline 1: Transcript Synchrony Classifier

**Purpose**: Classify *what is being said* based on transcribed text.

### Input
- Whisper transcription → words assigned to each second
- Single words or short phrases per 1-second window

### Architecture
```
Whisper Decoder Output (text)
       │
       ▼
┌─────────────────────────┐
│  Word-level Features    │
│  - word embedding       │
│  - word type (POS tag)  │
│  - context window (±2s) │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Text Classifier Head   │
│  (small transformer or  │
│   lightweight MLP)      │
└─────────────────────────┘
       │
       ▼
   Per-second labels
```

### Single-Word Classification: Is it possible?

**Yes, but with caveats:**

| Approach | Pros | Cons |
|----------|------|------|
| **Single word only** | Simple, fast | Very limited semantics ("um", "yeah" hard to classify) |
| **Context window (±2-3 sec)** | Richer semantics | Slight latency, overlapping info |
| **Utterance-level** | Best semantics | Variable length, harder to align |

**Recommended**: Use a **context window** of ±2 seconds (5 words total if ~1 word/sec average).
This gives the classifier enough context while maintaining per-second granularity.

### Output Labels

| Field | Type | Description |
|-------|------|-------------|
| `second` | int | Time index |
| `has_speech` | bool | Words present in this second |
| `transcription` | str | Word(s) in this second |
| `word_count` | int | Number of words |
| `word_type` | str | `content`, `filler`, `backchannel`, `question`, `exclamation` |
| `sentiment` | float | Word/phrase sentiment [-1, 1] |
| `is_turn_start` | bool | Start of a new speaking turn |

---

## Pipeline 2: Audio Synchrony Classifier

**Purpose**: Classify *how it sounds* directly from audio (independent of transcription).

### Input
- Raw audio waveform (16kHz mono)
- 1-second audio chunks

### Architecture
```
Raw Audio (16kHz, 1 second)
       │
       ▼
┌─────────────────────────┐
│  Whisper Encoder        │
│  (frozen pretrained)    │
│  → 1500-dim embedding   │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Audio Classifier Head  │
│  - Linear layers        │
│  - Temporal pooling     │
└─────────────────────────┘
       │
       ▼
   Per-second labels
```

### Why Whisper Encoder?
- Pretrained on 680k hours of audio → excellent audio representations
- Captures speech, music, noise, silence patterns
- No transcription needed → works for non-speech sounds (laughter, crying)

### Output Labels

| Field | Type | Description |
|-------|------|-------------|
| `second` | int | Time index |
| `audio_event` | str | `speech`, `laughter`, `crying`, `babbling`, `silence`, `noise`, `music` |
| `event_confidence` | float | Confidence [0-1] |
| `has_vocalization` | bool | Human vocalization present |
| `energy_db` | float | Audio energy in decibels |
| `is_speech` | bool | Speech specifically (not other vocalizations) |

---

## Comparison

| Aspect | Transcript Classifier | Audio Classifier |
|--------|----------------------|------------------|
| **Input** | Text (from Whisper decoder) | Audio (via Whisper encoder) |
| **Pretrained model** | Word embeddings (or small LM) | Whisper encoder (frozen) |
| **Detects** | Linguistic content, sentiment | Sound type, acoustic events |
| **Works without speech** | No | Yes (classifies silence, noise) |
| **Language-dependent** | Yes | No |
| **Training data needed** | Labeled transcripts | Labeled audio segments |

## Architecture

### Processing Flow

```
Video or Audio File
       │
       ▼
┌─────────────────────┐
│  Audio Extraction   │  (ffmpeg CLI)
│   → mono 16kHz WAV  │
└─────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                   Whisper Large Model                   │
│  ┌─────────────────┐         ┌─────────────────────┐   │
│  │ Encoder         │────────►│ Decoder             │   │
│  │ (audio → embed) │         │ (embed → text)      │   │
│  └────────┬────────┘         └──────────┬──────────┘   │
└───────────┼─────────────────────────────┼──────────────┘
            │                             │
            ▼                             ▼
┌───────────────────────┐     ┌───────────────────────┐
│   AUDIO CLASSIFIER    │     │  LANGUAGE CLASSIFIER  │
│                       │     │                       │
│  Whisper encoder      │     │  Transcription text   │
│  embeddings (frozen)  │     │  → Text encoder       │
│  → Classification head│     │  → Classification head│
│                       │     │                       │
│  Output:              │     │  Output:              │
│  - audio_event        │     │  - sentiment          │
│  - vocalization type  │     │  - linguistic_category│
│  - acoustic features  │     │  - intent             │
└───────────┬───────────┘     └───────────┬───────────┘
            │                             │
            └──────────┬──────────────────┘
                       ▼
            ┌─────────────────────┐
            │  Second Alignment   │
            │  + Merge Results    │
            └─────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   Output Results    │  CSV / JSON
            └─────────────────────┘
```

### Data Structures

```python
# ============================================
# Audio Classifier Data Structures
# ============================================

@dataclass
class AudioClassification:
    """Audio classifier output for a single second."""
    second: int
    audio_event: str              # speech, laughter, crying, babbling, silence, noise
    audio_confidence: float       # Confidence [0-1]
    has_vocalization: bool        # Any human sound
    # Acoustic features (from Whisper encoder or librosa)
    acoustic_energy: float        # RMS energy
    acoustic_embedding: np.ndarray | None  # Optional: full embedding vector

@dataclass
class AudioClassifierResult:
    """Complete audio classifier output for a file."""
    source_path: str
    model_name: str               # e.g., "whisper-large-v3-audio-classifier"
    classifications: list[AudioClassification]
    total_seconds: int
    vocalization_seconds: int
    vocalization_ratio: float

# ============================================
# Language Classifier Data Structures
# ============================================

@dataclass
class WordSegment:
    """Individual word with timing."""
    word: str
    start: float
    end: float
    probability: float | None

@dataclass
class LanguageClassification:
    """Language classifier output for a single second."""
    second: int
    has_speech: bool
    transcription: str
    word_count: int
    words: list[WordSegment]
    # Language classifier outputs
    linguistic_sentiment: float | None   # [-1, 1] negative to positive
    linguistic_category: str | None      # question, statement, exclamation, etc.
    linguistic_confidence: float | None

@dataclass
class LanguageClassifierResult:
    """Complete language classifier output for a file."""
    source_path: str
    model_name: str               # e.g., "whisper-large-v3 + distilbert-sentiment"
    language: str | None
    language_probability: float | None
    full_text: str
    classifications: list[LanguageClassification]
    total_seconds: int
    speech_seconds: int
    speech_ratio: float

# ============================================
# Combined Result
# ============================================

@dataclass
class AudioSynchronyResult:
    """Combined output from both classifiers."""
    source_path: str
    duration_seconds: float
    total_seconds: int
    audio_result: AudioClassifierResult
    language_result: LanguageClassifierResult
```

## Implementation Plan

### 0. Shared: Audio Extraction

**File**: `src/synchronai/data/audio/processing.py`

```python
def extract_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """Extract or normalize audio with ffmpeg to 16kHz mono WAV."""

def get_audio_duration(audio_path: str | Path) -> float:
    """Get duration in seconds using ffprobe or wave module."""

def load_audio_chunk(audio_path: str | Path, start_sec: int, duration: float = 1.0) -> np.ndarray:
    """Load a specific 1-second chunk for processing."""
```

---

## Pipeline 1: Transcript Synchrony

### 1.1 Whisper Transcription

**File**: `src/synchronai/inference/audio/transcribe.py`

```python
def transcribe_audio(
    input_path: str | Path,
    model_size: str = "large-v3",
    device: str | None = None,
    language: str | None = None,
) -> TranscriptionResult:
    """Run Whisper transcription with word timestamps."""

def align_words_to_seconds(
    words: list[WordSegment],
    total_seconds: int,
) -> dict[int, list[WordSegment]]:
    """Assign words to seconds using midpoint."""
```

### 1.2 Transcript Classifier Model

**File**: `src/synchronai/models/audio/transcript_classifier.py`

```python
class TranscriptClassifierConfig:
    context_window: int = 2          # ±2 seconds of context
    embedding_dim: int = 256
    hidden_dim: int = 128
    num_classes: int = 5             # word_type categories
    use_pretrained_embeddings: bool = True

class TranscriptClassifier(nn.Module):
    """Classifies transcript text per second.

    Input: Words from current second + context window
    Output: word_type, sentiment, is_turn_start
    """

    def __init__(self, config: TranscriptClassifierConfig):
        # Word embedding layer (or pretrained)
        # Context aggregation (attention or concat)
        # Classification heads

    def forward(self, word_ids: Tensor, context_mask: Tensor) -> dict[str, Tensor]:
        # Returns: word_type_logits, sentiment, turn_start_prob
```

### 1.3 Transcript Inference

**File**: `src/synchronai/inference/audio/transcript_predict.py`

```python
def predict_transcript_synchrony(
    input_path: str | Path,
    weights_path: str | Path,
    whisper_model_size: str = "large-v3",
    context_window: int = 2,
) -> TranscriptSynchronyResult:
    """Full pipeline: transcribe → classify per second."""
```

---

## Pipeline 2: Audio Synchrony

### 2.1 Whisper Encoder Feature Extraction

**File**: `src/synchronai/models/audio/whisper_encoder.py`

```python
class WhisperEncoderFeatures:
    """Extract features from Whisper encoder (frozen)."""

    def __init__(self, model_size: str = "large-v3", device: str | None = None):
        self.model = whisper.load_model(model_size, device=device)
        # Freeze encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract encoder embeddings for 1-second audio chunk.

        Args:
            audio: 16kHz mono audio array (16000 samples for 1 sec)

        Returns:
            Encoder embedding (e.g., 1500-dim for large model)
        """
        # Pad/trim to 30s (Whisper requirement), extract encoder output
        # Pool temporal dimension to get fixed-size embedding
```

### 2.2 Audio Classifier Model

**File**: `src/synchronai/models/audio/audio_classifier.py`

```python
class AudioClassifierConfig:
    whisper_model_size: str = "large-v3"
    encoder_dim: int = 1280           # Whisper large encoder dim
    hidden_dim: int = 256
    num_event_classes: int = 7        # speech, laughter, crying, babbling, silence, noise, music
    freeze_encoder: bool = True

class AudioClassifier(nn.Module):
    """Classifies audio events from Whisper encoder features.

    Input: 1-second audio waveform
    Output: audio_event, confidence, has_vocalization, energy_db
    """

    def __init__(self, config: AudioClassifierConfig):
        self.encoder = WhisperEncoderFeatures(config.whisper_model_size)
        self.classifier = nn.Sequential(
            nn.Linear(config.encoder_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_dim, config.num_event_classes),
        )
        self.vocalization_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, audio: Tensor) -> dict[str, Tensor]:
        features = self.encoder.extract_features(audio)
        # Returns: event_logits, vocalization_prob, energy
```

### 2.3 Audio Inference

**File**: `src/synchronai/inference/audio/audio_predict.py`

```python
def predict_audio_synchrony(
    input_path: str | Path,
    weights_path: str | Path,
) -> AudioSynchronyResult:
    """Classify audio events for each second."""
```

---

## Batch Processing (Both Pipelines)

**File**: `src/synchronai/inference/audio/batch.py`

```python
def process_batch(
    input_paths: list[Path],
    output_dir: Path,
    pipeline: str = "both",          # "transcript", "audio", or "both"
    transcript_weights: str | None = None,
    audio_weights: str | None = None,
    skip_existing: bool = True,
) -> list[AudioSynchronyResult]:
    """Process multiple files through one or both pipelines."""
```

---

## Export Helpers

```python
def export_transcript_csv(result: TranscriptSynchronyResult, path: Path) -> Path:
    """CSV: second, transcription, word_count, word_type, sentiment, is_turn_start"""

def export_audio_csv(result: AudioSynchronyResult, path: Path) -> Path:
    """CSV: second, audio_event, confidence, has_vocalization, energy_db"""

def export_combined_csv(result: CombinedResult, path: Path) -> Path:
    """CSV: Merged outputs from both pipelines."""

def export_json(result: Any, path: Path) -> Path:
    """Full JSON with all details."""
```

## Configuration

**File**: `configs/inference/audio.yaml`

```yaml
audio:
  # Shared settings
  sample_rate: 16000
  mono: true
  whisper_model: "large-v3"

  # Pipeline selection
  pipeline: "both"  # transcript, audio, or both

  # Transcript pipeline settings
  transcript:
    context_window: 2          # ±seconds for context
    language: null             # null = auto-detect
    weights: null              # path to trained classifier

  # Audio pipeline settings
  audio_classifier:
    freeze_encoder: true
    hidden_dim: 256
    weights: null              # path to trained classifier

  # Output settings
  output_format: "both"        # csv, json, both
  skip_existing: true
```

**File**: `configs/model/audio_classifier.yaml`

```yaml
audio_classifier:
  whisper_model_size: "large-v3"
  encoder_dim: 1280            # Whisper large encoder output dim
  hidden_dim: 256
  num_event_classes: 7
  freeze_encoder: true
  dropout: 0.3

transcript_classifier:
  context_window: 2
  embedding_dim: 256
  hidden_dim: 128
  num_classes: 5               # word_type categories
  use_pretrained_embeddings: true
```

## CLI Integration

**Add to** `src/synchronai/main.py`:

### New Modality and Modes

```python
# Modality selection
parser.add_argument("--modality", choices=["fnirs", "video", "audio"], default=None)
parser.add_argument("--audio", action="store_true", help="Shortcut for --modality audio.")

# Audio-specific modes
mode_group.add_argument(
    "--pipeline",
    choices=["transcript", "audio", "both"],
    default="both",
    help="Which audio pipeline to run.",
)
```

### Audio Arguments

```python
audio_group = parser.add_argument_group("Audio Processing")
audio_group.add_argument("--audio-path", help="Single audio/video file.")
audio_group.add_argument("--audio-dir", help="Directory of audio/video files.")
audio_group.add_argument("--transcript-weights", help="Weights for transcript classifier.")
audio_group.add_argument("--audio-weights", help="Weights for audio classifier.")
audio_group.add_argument("--whisper-model", default="large-v3",
                         choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"])
audio_group.add_argument("--context-window", type=int, default=2,
                         help="Context window (±seconds) for transcript classifier.")
audio_group.add_argument("--output-format", choices=["csv", "json", "both"], default="both")
audio_group.add_argument("--skip-existing", action="store_true",
                         help="Skip files that already have output.")
```

### Example Commands

**Run both pipelines:**
```bash
python -m synchronai.main --audio --mode predict \
  --audio-dir /path/to/videos \
  --output-dir data/audio_sync \
  --pipeline both \
  --whisper-model large-v3 \
  --output-format both
```

**Run transcript pipeline only:**
```bash
python -m synchronai.main --audio --mode predict \
  --audio-path video.mp4 \
  --pipeline transcript \
  --transcript-weights runs/transcript_classifier/best.pt
```

**Run audio pipeline only:**
```bash
python -m synchronai.main --audio --mode predict \
  --audio-dir /path/to/videos \
  --pipeline audio \
  --audio-weights runs/audio_classifier/best.pt
```

**Train audio classifier:**
```bash
python -m synchronai.main --audio --train audio-classifier \
  --labels-file data/audio_labels.csv \
  --save-dir runs/audio_classifier \
  --whisper-model large-v3 \
  --epochs 50
```

## BSub Integration (Aligned with Existing Scripts)

### Job Script: `scripts/bsub/audio_synchrony_bsub.sh`

```bash
#!/bin/bash
#BSUB -G compute-perlmansusan
#BSUB -q general
#BSUB -m general
#BSUB -M 99000000
#BSUB -a 'docker(continuumio/anaconda3)'
#BSUB -n 20
#BSUB -R 'select[mem>99GB && tmp>99GB] rusage[mem=99GB, tmp=99GB]'

conda init
source /home/$USER/.bashrc
source $SYNCHRONAI_DIR/ml-env/bin/activate
cd $SYNCHRONAI_DIR

pip install -e ".[audio]"

# Ensure ffmpeg is available
which ffmpeg || conda install -y -c conda-forge ffmpeg

bash $SYNCHRONAI_DIR/scripts/transcribe_audio.sh
```

**Note**: Whisper large runs on CPU by default in this configuration. For GPU acceleration,
add these directives and use a PyTorch image:
```bash
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"
#BSUB -a 'docker(pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime)'
```

### Submission Script: `scripts/bsub/pre_audio_synchrony_bsub.sh`

```bash
#!/bin/sh

export CONDA_ENVS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/perlmansusan/Active/moochie/resources/conda/pkgs/"

export SYNCHRONAI_DIR="/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI/"

export PATH="/opt/conda/bin:$PATH"
export PYTHONPATH='/storage1/fs1/perlmansusan/Active/moochie/github/synchronAI':$PYTHONPATH

export LSF_DOCKER_VOLUMES="/storage1/fs1/perlmansusan/Active:/storage1/fs1/perlmansusan/Active /home/$USER:/home/$USER"
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

export DATE=$(date +'%m-%d')

bsub -J synchronai-audio-$DATE \
     -oo $SYNCHRONAI_DIR/scripts/bsub/logs/synchronai_audio_$DATE.log \
     -g /$USER/preprocessing \
     < $SYNCHRONAI_DIR/scripts/bsub/audio_synchrony_bsub.sh
```

### Processing Script: `scripts/audio_synchrony.sh`

```bash
#!/bin/bash

INPUT_DIRECTORY="/storage1/fs1/perlmansusan/Active/moochie/study_data/..."
OUTPUT_DIRECTORY="data/audio_sync/"

# Run both pipelines
python -m synchronai.main --audio --mode predict \
  --audio-dir "$INPUT_DIRECTORY" \
  --output-dir "$OUTPUT_DIRECTORY" \
  --pipeline both \
  --whisper-model large-v3 \
  --output-format both \
  --skip-existing
```

### Training Scripts

**`scripts/train_transcript_classifier.sh`**:
```bash
#!/bin/bash

python -m synchronai.main --audio --train transcript-classifier \
  --labels-file data/transcript_labels.csv \
  --save-dir runs/transcript_classifier_$(date +%Y%m%d) \
  --whisper-model large-v3 \
  --context-window 2 \
  --epochs 50 \
  --batch-size 32 \
  --use-amp
```

**`scripts/train_audio_classifier.sh`**:
```bash
#!/bin/bash

python -m synchronai.main --audio --train audio-classifier \
  --labels-file data/audio_labels.csv \
  --save-dir runs/audio_classifier_$(date +%Y%m%d) \
  --whisper-model large-v3 \
  --epochs 50 \
  --batch-size 16 \
  --use-amp
```

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
audio = [
    "openai-whisper>=20231117",
    "librosa>=0.10.0",              # Audio feature extraction
    "soundfile>=0.12.0",            # Audio I/O
]
```

System requirement:
- `ffmpeg` (and `ffprobe`) must be installed.

**Note**: Both classifiers use PyTorch (already a dependency for video pipeline).

## Training Data Format

### Audio Classifier Labels (`audio_labels.csv`)

| audio_path | second | audio_event | has_vocalization |
|------------|--------|-------------|------------------|
| /path/video1.mp4 | 0 | speech | true |
| /path/video1.mp4 | 1 | laughter | true |
| /path/video1.mp4 | 2 | silence | false |
| /path/video1.mp4 | 3 | crying | true |
| /path/video1.mp4 | 4 | noise | false |

**Event classes**: `speech`, `laughter`, `crying`, `babbling`, `silence`, `noise`, `music`

### Transcript Classifier Labels (`transcript_labels.csv`)

| audio_path | second | transcription | word_type | sentiment |
|------------|--------|---------------|-----------|-----------|
| /path/video1.mp4 | 0 | "hello" | content | 0.2 |
| /path/video1.mp4 | 1 | "um" | filler | 0.0 |
| /path/video1.mp4 | 2 | "what?" | question | -0.1 |
| /path/video1.mp4 | 3 | "wow" | exclamation | 0.5 |

**Word types**: `content`, `filler`, `backchannel`, `question`, `exclamation`

---

## Output Format

### Transcript Classifier CSV (`<name>_transcript.csv`)

| second | transcription | word_count | word_type | sentiment | is_turn_start |
|--------|---------------|------------|-----------|-----------|---------------|
| 0 | "Hello there" | 2 | content | 0.3 | true |
| 1 | "um" | 1 | filler | 0.0 | false |
| 2 | "" | 0 | — | — | false |
| 3 | "what?" | 1 | question | -0.1 | true |

### Audio Classifier CSV (`<name>_audio.csv`)

| second | audio_event | confidence | has_vocalization | energy_db |
|--------|-------------|------------|------------------|-----------|
| 0 | speech | 0.95 | true | -12.3 |
| 1 | laughter | 0.88 | true | -8.5 |
| 2 | silence | 0.99 | false | -45.2 |
| 3 | speech | 0.91 | true | -15.1 |

### Combined JSON Output (`<name>_audio_sync.json`)

```json
{
  "source_path": "/path/to/video.mp4",
  "duration_seconds": 120.0,
  "total_seconds": 120,

  "transcript_result": {
    "language": "en",
    "speech_seconds": 85,
    "speech_ratio": 0.708,
    "full_text": "Hello there um what...",
    "classifications": [
      {
        "second": 0,
        "transcription": "Hello there",
        "word_count": 2,
        "word_type": "content",
        "sentiment": 0.3,
        "is_turn_start": true
      }
    ]
  },

  "audio_result": {
    "vocalization_seconds": 90,
    "vocalization_ratio": 0.75,
    "classifications": [
      {
        "second": 0,
        "audio_event": "speech",
        "confidence": 0.95,
        "has_vocalization": true,
        "energy_db": -12.3
      }
    ]
  }
}
```

## Future Enhancements

1. **Speaker diarization**: Identify speakers (adult vs child) with pyannote-audio.
2. **Turn-taking analysis**: Detect conversational overlaps, gaps, and turn boundaries.
3. **Audio-visual fusion**: Combine audio + video classifiers for multimodal synchrony.
4. **Emotion recognition**: Fine-grained emotion from both audio tone and text content.
5. **Real-time inference**: Stream processing for live analysis.
6. **Active learning**: Semi-supervised labeling using model predictions.

## File Structure

```
src/synchronai/
├── data/
│   └── audio/
│       ├── __init__.py
│       ├── processing.py          # Audio extraction, chunking
│       └── dataset.py             # PyTorch datasets for training
├── models/
│   └── audio/
│       ├── __init__.py
│       ├── whisper_encoder.py     # Whisper encoder feature extraction
│       ├── audio_classifier.py    # Audio event classifier
│       └── transcript_classifier.py  # Text-based classifier
├── inference/
│   └── audio/
│       ├── __init__.py
│       ├── transcribe.py          # Whisper transcription
│       ├── audio_predict.py       # Audio classifier inference
│       ├── transcript_predict.py  # Transcript classifier inference
│       └── batch.py               # Batch processing
├── training/
│   └── audio/
│       ├── __init__.py
│       ├── train_audio.py         # Audio classifier training
│       └── train_transcript.py    # Transcript classifier training
scripts/
├── audio_synchrony.sh             # Main inference script
├── train_audio_classifier.sh
├── train_transcript_classifier.sh
└── bsub/
    ├── audio_synchrony_bsub.sh
    └── pre_audio_synchrony_bsub.sh
configs/
├── inference/
│   └── audio.yaml
└── model/
    └── audio_classifier.yaml
```
