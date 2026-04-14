# Transcript Synchrony: Second-by-Second Language Classification

## Overview

This document describes the implementation plan for a **transcript-based classifier** that analyzes
*what is being said* based on Whisper transcriptions. This complements the audio classifier
(which analyzes *how it sounds*) to provide a complete audio understanding pipeline.

## Relationship to Audio Classifier

The transcript classifier builds on the audio infrastructure already implemented:

| Component | Audio Classifier (Done) | Transcript Classifier (Planned) |
|-----------|------------------------|--------------------------------|
| **Input** | Raw audio waveform | Whisper transcription text |
| **Feature extraction** | Whisper encoder (frozen) | Word embeddings + context |
| **Output** | Audio events (speech, laughter, etc.) | Linguistic features (sentiment, type) |
| **Works without speech** | Yes | No (requires transcribed text) |

## Architecture

### Processing Flow

```
Audio/Video File
       │
       ▼
┌─────────────────────────┐
│  Whisper Transcription  │  (shared with audio pipeline)
│  → word-level timestamps│
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Word-to-Second Align   │  (midpoint assignment)
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Context Window Builder │  (±2 seconds)
│  → gather surrounding   │
│     words for context   │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│  Transcript Classifier  │
│  - Word embeddings      │
│  - Context aggregation  │
│  - Classification heads │
└─────────────────────────┘
       │
       ▼
   Per-second labels:
   - word_type
   - sentiment
   - is_turn_start
```

### Data Structures

```python
# Reuse from audio pipeline
from synchronai.data.audio.processing import extract_audio, get_audio_duration

# New structures for transcript classification
@dataclass
class TranscriptClassification:
    """Classification result for a single second."""
    second: int
    has_speech: bool
    transcription: str
    word_count: int
    words: list[WordSegment]  # Reuse from audio
    # Classification outputs
    word_type: str            # content, filler, backchannel, question, exclamation
    sentiment: float          # [-1, 1]
    is_turn_start: bool
    confidence: float

@dataclass
class TranscriptClassifierResult:
    """Complete result for a file."""
    source_path: str
    language: str | None
    full_text: str
    classifications: list[TranscriptClassification]
    total_seconds: int
    speech_seconds: int
    speech_ratio: float
```

## Implementation Plan

### 1. Whisper Transcription Module

**File**: `src/synchronai/inference/audio/transcribe.py` (new)

```python
def transcribe_audio(
    input_path: str | Path,
    model_size: str = "large-v3",
    device: str | None = None,
    language: str | None = None,
) -> TranscriptionResult:
    """Run Whisper transcription with word timestamps.

    Returns full transcription with word-level timing.
    Reuses the Whisper model from audio classifier if loaded.
    """

def align_words_to_seconds(
    words: list[WordSegment],
    total_seconds: int,
) -> dict[int, list[WordSegment]]:
    """Assign words to seconds using midpoint.

    Each word is assigned to exactly one second based on its
    temporal midpoint: second = int((start + end) / 2)
    """
```

### 2. Context Window Builder

**File**: `src/synchronai/data/audio/context.py` (new)

```python
def build_context_windows(
    words_by_second: dict[int, list[WordSegment]],
    total_seconds: int,
    context_window: int = 2,
) -> list[ContextWindow]:
    """Build context windows for each second.

    For second t, gather words from [t - context_window, t + context_window].
    This provides richer semantic context for classification.
    """

@dataclass
class ContextWindow:
    """Words in a context window centered on a second."""
    center_second: int
    words: list[WordSegment]
    center_word_indices: list[int]  # Which words are in the center second
```

### 3. Transcript Classifier Model

**File**: `src/synchronai/models/audio/transcript_classifier.py` (new)

```python
@dataclass
class TranscriptClassifierConfig:
    context_window: int = 2
    embedding_dim: int = 256
    hidden_dim: int = 128
    num_word_types: int = 5
    use_pretrained_embeddings: bool = True
    pretrained_embedding_name: str = "glove.6B.100d"  # or sentence-transformers
    dropout: float = 0.3

class TranscriptClassifier(nn.Module):
    """Text-based classifier for transcript analysis.

    Architecture:
    1. Word embedding layer (pretrained or learned)
    2. Context aggregation (attention over context window)
    3. Multiple classification heads:
       - word_type: 5-class classification
       - sentiment: regression [-1, 1]
       - is_turn_start: binary classification
    """

    def __init__(self, config: TranscriptClassifierConfig):
        # Word embeddings
        self.embedding = self._build_embeddings(config)

        # Context attention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=4,
        )

        # Classification heads
        self.word_type_head = nn.Linear(config.hidden_dim, config.num_word_types)
        self.sentiment_head = nn.Linear(config.hidden_dim, 1)
        self.turn_start_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, word_ids, context_mask):
        # Embed words
        embeddings = self.embedding(word_ids)

        # Attend over context
        attended, _ = self.context_attention(embeddings, embeddings, embeddings)

        # Pool center second
        center_features = self._pool_center(attended, context_mask)

        # Classify
        return {
            "word_type_logits": self.word_type_head(center_features),
            "sentiment": torch.tanh(self.sentiment_head(center_features)),
            "turn_start_logit": self.turn_start_head(center_features),
        }
```

### 4. Transcript Inference

**File**: `src/synchronai/inference/audio/transcript_predict.py` (new)

```python
def predict_transcript_synchrony(
    input_path: str | Path,
    weights_path: str | Path,
    whisper_model_size: str = "large-v3",
    context_window: int = 2,
    device: str | None = None,
) -> TranscriptClassifierResult:
    """Full pipeline: transcribe → build context → classify per second."""
```

### 5. Combined Pipeline

**File**: `src/synchronai/inference/audio/combined.py` (new)

```python
def predict_audio_synchrony_combined(
    input_path: str | Path,
    audio_weights_path: str | Path,
    transcript_weights_path: str | Path | None = None,
    device: str | None = None,
) -> CombinedAudioResult:
    """Run both audio and transcript classifiers.

    If transcript_weights_path is None, only run audio classifier.
    This allows gradual adoption of the transcript classifier.
    """

@dataclass
class CombinedAudioResult:
    """Combined output from both classifiers."""
    audio_result: AudioPredictionResult
    transcript_result: TranscriptClassifierResult | None
```

## Training Data Format

### Transcript Labels (`transcript_labels.csv`)

| audio_path | second | transcription | word_type | sentiment | is_turn_start |
|------------|--------|---------------|-----------|-----------|---------------|
| video1.mp4 | 0 | "hello" | content | 0.2 | true |
| video1.mp4 | 1 | "um" | filler | 0.0 | false |
| video1.mp4 | 2 | "what?" | question | -0.1 | true |
| video1.mp4 | 3 | "" | — | — | false |

**Word type classes**:
- `content`: Meaningful words (nouns, verbs, adjectives)
- `filler`: Filler words (um, uh, like)
- `backchannel`: Response tokens (yeah, uh-huh, okay)
- `question`: Question words/phrases
- `exclamation`: Exclamations (wow, oh, yay)

## CLI Integration

Add to `src/synchronai/main.py`:

```python
# New mode for transcript classification
parser.add_argument(
    "--transcript",
    action="store_true",
    help="Run transcript classifier (requires --weights-path for transcript model).",
)
parser.add_argument(
    "--transcript-weights",
    help="Path to transcript classifier weights.",
)
parser.add_argument(
    "--context-window",
    type=int,
    default=2,
    help="Context window size (±seconds) for transcript classifier.",
)
```

**Example usage**:
```bash
# Run transcript classifier only
python -m synchronai.main --audio --mode predict \
    --audio-path video.mp4 \
    --transcript \
    --transcript-weights runs/transcript_classifier/best.pt

# Run both classifiers
python -m synchronai.main --audio --mode predict \
    --audio-path video.mp4 \
    --weights-path runs/audio_classifier/best.pt \
    --transcript \
    --transcript-weights runs/transcript_classifier/best.pt
```

## Dependencies

No additional dependencies required beyond what's already in `[project.optional-dependencies].audio`:
- `openai-whisper` (already included)
- `soundfile` (already included)

Optional for pretrained embeddings:
```toml
[project.optional-dependencies]
transcript = [
    "sentence-transformers>=2.2.0",  # For better text embeddings
]
```

## Output Format

### Transcript CSV (`<name>_transcript.csv`)

| second | transcription | word_count | word_type | sentiment | is_turn_start |
|--------|---------------|------------|-----------|-----------|---------------|
| 0 | "Hello there" | 2 | content | 0.3 | true |
| 1 | "um" | 1 | filler | 0.0 | false |
| 2 | "" | 0 | — | — | false |

### Combined JSON (`<name>_audio_combined.json`)

```json
{
  "source_path": "/path/to/video.mp4",
  "duration_seconds": 120.0,

  "audio_result": {
    "vocalization_seconds": 90,
    "classifications": [...]
  },

  "transcript_result": {
    "language": "en",
    "full_text": "Hello there um...",
    "speech_seconds": 85,
    "classifications": [...]
  }
}
```

## File Structure (After Implementation)

```
src/synchronai/
├── data/
│   └── audio/
│       ├── processing.py       # ✓ Done
│       ├── context.py          # NEW: Context window builder
│       └── dataset.py          # NEW: Training dataset
├── models/
│   └── audio/
│       ├── whisper_encoder.py  # ✓ Done
│       ├── audio_classifier.py # ✓ Done
│       └── transcript_classifier.py  # NEW
├── inference/
│   └── audio/
│       ├── predict.py          # ✓ Done (audio classifier)
│       ├── transcribe.py       # NEW: Whisper transcription
│       ├── transcript_predict.py  # NEW: Transcript inference
│       └── combined.py         # NEW: Combined pipeline
└── training/
    └── audio/
        ├── train_audio.py      # TODO: Audio classifier training
        └── train_transcript.py # NEW: Transcript classifier training
```

## Implementation Order

1. **Phase 1**: Whisper transcription module (`transcribe.py`)
   - Word-level timestamps
   - Second alignment
   - Integration with existing audio pipeline

2. **Phase 2**: Context window builder (`context.py`)
   - Build context windows for each second
   - Handle edge cases (start/end of file)

3. **Phase 3**: Transcript classifier model (`transcript_classifier.py`)
   - Word embeddings (start simple, add pretrained later)
   - Context attention
   - Classification heads

4. **Phase 4**: Inference and CLI integration
   - `transcript_predict.py`
   - `combined.py`
   - CLI arguments

5. **Phase 5**: Training pipeline
   - Dataset class
   - Training loop
   - BSub scripts

## Design Decisions

### Why separate from audio classifier?

1. **Different inputs**: Audio classifier uses raw waveforms; transcript uses text
2. **Different features**: Audio captures acoustic patterns; transcript captures semantics
3. **Independent operation**: Can use one or both depending on needs
4. **Easier debugging**: Issues can be isolated to specific classifier

### Why context window?

Single words per second are often meaningless ("um", "the"). Context provides:
- Richer semantic information
- Better turn-taking detection
- More accurate sentiment analysis

### Why ±2 seconds default?

- Covers typical phrase length (3-5 seconds)
- Balances context richness with locality
- Matches video synchrony window granularity
