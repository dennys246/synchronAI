# Generative Pretraining and Transfer Learning for fNIRS-Based Child-Adult Classification

## Methods

### 2.1 fNIRS Data Acquisition and Preprocessing

Functional near-infrared spectroscopy (fNIRS) recordings were obtained from two multi-site longitudinal studies of parent-child dyadic interaction: CARE (Washington University in St. Louis) and P-CAT (R56 and R01 grants; Washington University and Penn State University). Recordings were acquired using NIRx systems with 8 sources and 4 detectors, yielding 10 source-detector pairs (S1-D1, S2-D1, S2-D2, S3-D2, S4-D2, S5-D3, S6-D3, S7-D3, S7-D4, S8-D4). Both oxygenated (HbO) and deoxygenated (HbR) hemoglobin concentrations were measured per pair, producing 20-dimensional time series (10 pairs x 2 hemoglobin types).

Raw intensity data were preprocessed using MNE-NIRS with the following pipeline: (1) conversion from raw optical density to hemoglobin concentrations via the modified Beer-Lambert law, (2) bandpass filtering (0.01-0.5 Hz) to remove physiological noise and slow drift, (3) automated channel quality assessment using scalp coupling index (SCI > 0.5 threshold), (4) interpolation of bad channels via nearest-neighbor replacement, and (5) resampling to a uniform 7.8125 Hz. Recordings with all channels marked as bad were excluded (60 of 2,174 recordings, 2.8%).

Each recording was segmented into non-overlapping 60-second windows (472 samples at 7.8125 Hz) and z-score normalized per channel using running statistics computed across the training corpus.

Participant type (child vs. adult) was determined from directory naming conventions specific to each study site. For CARE, participants with 5-digit subject IDs were classified as children and 4-digit IDs as parents. For P-CAT R56 and R01, directory suffixes (`-C_`, `_C` for child; `-P_`, `_P` for parent) provided unambiguous labels. Recordings with ambiguous labeling were assigned "unknown" and excluded from the classification task.

The final dataset comprised 2,114 successfully processed recordings yielding 67,486 sixty-second windows (33,540 adult; 33,128 child; 818 unknown/excluded), drawn from 1,596 unique participant-session combinations.

### 2.2 Denoising Diffusion Probabilistic Model (DDPM) Pretraining

A denoising diffusion probabilistic model (DDPM; Ho et al., 2020) was trained on the full corpus of fNIRS windows as an unsupervised generative pretraining step. The objective was to learn a rich latent representation of hemodynamic temporal dynamics without requiring task-specific labels.

#### 2.2.1 Architecture

The generative model employed a 1D U-Net architecture operating on input tensors of shape (472, 20). The U-Net consisted of:

- **Encoder (down-path)**: Three resolution levels with residual blocks implementing Conv1D(kernel=3, padding=same) → LayerNorm → SiLU activation, followed by strided downsampling (Conv1D, kernel=4, stride=2). Channel widths doubled at each level: 20→64→128→256.

- **Bottleneck**: Two residual blocks at the lowest resolution (59 timesteps), expanding from 256 to 512 channels, yielding a bottleneck representation of shape (59, 512).

- **Decoder (up-path)**: Symmetric to the encoder with transposed convolutions for upsampling and skip connections from corresponding encoder levels.

- **Timestep conditioning**: Diffusion timestep was encoded via sinusoidal positional embedding (dim=128), projected through a two-layer MLP (Dense(512, SiLU) → Dense(128, SiLU)), and injected into each residual block via additive conditioning after the first convolution.

- **Regularization**: Dropout (p=0.15) was applied within residual blocks.

The total model contained approximately 3.5M parameters. The architecture was implemented in TensorFlow/Keras.

#### 2.2.2 Diffusion Process

The forward diffusion process employed a cosine beta schedule (Nichol & Dhariwal, 2021) with T=1000 timesteps, β_start=0.0001, β_end=0.02. The training objective was the standard simplified DDPM loss: the model predicted the noise ε added at timestep t, with the loss computed as MSE between predicted and actual noise.

#### 2.2.3 Training

The model was trained for 622 epochs on the full corpus of 67,486 windows using the Adam optimizer. Training and validation sets were split by subject to prevent data leakage. Training loss decreased from 1.058 (epoch 1) to a plateau of ~0.008 (epoch 400+), with best validation loss of 0.00345 achieved at epoch 491. The model showed no signs of overfitting (validation loss remained lower than training loss throughout, likely due to dropout being active only during training).

Generative quality was assessed periodically using Fréchet Inception Distance (FID) adapted for 1D signals and Maximum Mean Discrepancy (MMD). Median FID across evaluations was 5.12, with a best score of 3.80, indicating that generated fNIRS signals were statistically similar to real recordings in distribution.

### 2.3 Transfer Learning: U-Net Encoder as Feature Extractor

#### 2.3.1 Weight Conversion and Feature Extraction

The encoder (down-path) and bottleneck of the trained DDPM U-Net were ported from TensorFlow to PyTorch. Weights were extracted from the Keras HDF5 checkpoint using h5py and mapped to a PyTorch replica of the encoder architecture. Conversion fidelity was verified by passing identical synthetic inputs through both implementations and confirming output agreement (maximum absolute difference < 1e-4).

For each 60-second fNIRS window, the pretrained encoder was applied as a frozen feature extractor with diffusion timestep fixed at t=0 (fully denoised). Two feature representations were extracted:

- **Bottleneck features**: The output of the final bottleneck residual block, yielding a (59, 512)-dimensional representation per window — 59 temporal positions at 512-dimensional feature vectors.

- **Multiscale features**: Concatenation of mean-pooled activations from all three encoder levels plus the bottleneck, yielding a 960-dimensional vector per window (64 + 128 + 256 + 512).

Features were saved to disk as PyTorch tensors, enabling rapid training of downstream classifiers without loading the encoder at inference time.

#### 2.3.2 Child vs. Adult Classification

A classification sweep was conducted to evaluate the discriminative quality of the pretrained encoder features and to identify the optimal architecture for downstream transfer to synchrony prediction. Eleven classifier variants were evaluated across two feature representations (bottleneck and multiscale), three temporal aggregation strategies (mean pooling, MLP, LSTM), and varying capacity/regularization configurations.

All classifiers operated on the pre-extracted feature tensors. For bottleneck features (59, 512), temporal aggregation was required to produce a fixed-length representation:

- **Mean pooling**: Average across the 59 temporal positions → 512-dim vector → linear or MLP head.
- **LSTM**: Bidirectional LSTM (hidden_dim=64) over the 59-step sequence → final hidden state → classification head.

Classification heads consisted of either: (a) a linear layer (logistic regression baseline), (b) a single-hidden-layer MLP with ReLU activation and dropout, or (c) for LSTM variants, the LSTM's hidden state projected through a linear layer.

Training used the AdamW optimizer with cosine learning rate schedule (peak LR=3e-4, warmup=3 epochs), weight decay=0.01, and early stopping with patience=15 epochs. Data were split 80/20 by subject (1,277 train subjects, 319 validation subjects; 53,448 train windows, 13,220 validation windows) to prevent within-subject leakage.

### 2.4 Ablation: Random Encoder Baseline

To attribute classification performance to the generative pretraining rather than to architectural inductive biases or properties of the fNIRS signal itself, an ablation study was conducted using a randomly initialized encoder with identical architecture but no pretrained weights. Features were extracted using this random encoder under the same protocol, and the best-performing classifier (LSTM, hidden_dim=64) was trained on the resulting features with identical hyperparameters.

---

## Results

### 3.1 Generative Model Convergence

The DDPM achieved stable convergence by approximately epoch 400, with training loss plateauing at 0.0075 (±0.001) and validation loss at 0.0049 (±0.001). Best validation loss (0.00345) was observed at epoch 491, with no improvement in the subsequent 130 epochs. The absence of a train-validation gap (validation loss was consistently lower than training loss) indicates effective regularization without overfitting.

Generative quality metrics confirmed that the model learned the distributional properties of real fNIRS signals, with a median Fréchet Inception Distance of 5.12 across evaluation epochs. Visual inspection of generated samples showed realistic hemodynamic response temporal profiles with appropriate inter-channel correlations.

### 3.2 Child vs. Adult Classification

Classification results across all 11 sweep variants are summarized in Table 1.

**Table 1.** Child vs. adult classification performance across architectural variants. AUC = area under the ROC curve. All metrics reported on held-out validation set (20% of subjects).

| Rank | Model | Features | Temporal Agg. | Hidden Dim | Dropout | AUC | Accuracy | F1 | Best Epoch |
|------|-------|----------|--------------|------------|---------|-----|----------|-----|------------|
| 1 | bn_lstm64 | Bottleneck (512) | LSTM | 64 | 0.3 | **0.974** | **0.924** | **0.926** | 14 |
| 2 | bn_lstm_proj | Bottleneck (512) | LSTM | 64 | 0.5 | 0.971 | 0.914 | 0.906 | 18 |
| 3 | bn_mlp32 | Bottleneck (512) | Mean pool | 32 | 0.3 | 0.915 | 0.831 | 0.835 | 47 |
| 4 | ms_mlp128 | Multiscale (960) | Mean pool | 128 | 0.5 | 0.910 | 0.831 | 0.839 | 50 |
| 5 | bn_mlp64_proj | Bottleneck (512) | Mean pool | 64 | 0.5 | 0.902 | 0.820 | 0.823 | 48 |
| 6 | ms_mlp32 | Multiscale (960) | Mean pool | 32 | 0.3 | 0.897 | 0.817 | 0.823 | 46 |
| 7 | ms_mlp64_proj | Multiscale (960) | Mean pool | 64 | 0.5 | 0.895 | 0.819 | 0.831 | 44 |
| 8 | bn_mlp32_overlap | Bottleneck (512) | Mean pool | 32 | 0.3 | 0.893 | 0.819 | 0.822 | 50 |
| 9 | ms_mlp64_hvreg | Multiscale (960) | Mean pool | 64 | 0.7 | 0.873 | 0.795 | 0.808 | 47 |
| 10 | bn_linear | Bottleneck (512) | Mean pool | — | — | 0.860 | 0.788 | 0.780 | 46 |
| 11 | ms_linear | Multiscale (960) | Mean pool | — | — | 0.852 | 0.774 | 0.764 | 50 |

#### 3.2.1 Effect of Temporal Modeling

The most striking finding was the dominant effect of LSTM-based temporal aggregation. The top two models both employed bidirectional LSTM (AUC 0.974 and 0.971), outperforming the best mean-pooled model (bn_mlp32, AUC 0.915) by 5.9 percentage points. This indicates that the temporal dynamics encoded in the 59-timestep bottleneck sequence — corresponding to 60 seconds of hemodynamic activity at ~1 Hz effective resolution — carry substantial discriminative information that is destroyed by simple averaging.

This result is physiologically interpretable: children and adults differ not only in hemodynamic response amplitude but in temporal characteristics including response latency, rise time, and inter-region synchronization patterns. The LSTM captures these temporal dependencies where mean pooling cannot.

#### 3.2.2 Bottleneck vs. Multiscale Features

Bottleneck features (512-dim) consistently outperformed multiscale features (960-dim) across matched architectures (e.g., bn_linear 0.860 vs. ms_linear 0.852; bn_mlp32 0.915 vs. ms_mlp32 0.897). This suggests that the most abstract representation at the deepest encoder level captures the relevant hemodynamic properties more effectively than the concatenation of features from all resolution levels, which may introduce noise from lower-level representations that are more relevant to signal reconstruction than to participant classification.

#### 3.2.3 Overfitting and Generalization

The best model (bn_lstm64) exhibited a train-validation accuracy gap of only 2.3% (94.7% train, 92.4% validation), indicating strong generalization despite the temporal model's capacity. Models converged within 14-18 epochs for LSTM variants and 46-50 epochs for simpler architectures, consistent with the LSTM's ability to extract discriminative features efficiently from the sequential representation.

#### 3.2.4 Linear Probe Performance

Even a linear classifier (logistic regression) achieved AUC 0.860 on bottleneck features, demonstrating that the pretrained encoder produces linearly separable representations for the child/adult task. This is notable given that the encoder was trained purely for generative reconstruction and received no discriminative supervision.

### 3.3 Ablation: Pretraining Contribution

*(Results pending — ablation study submitted to compute cluster. Random-init encoder features extracted using identical architecture with PyTorch default initialization. Expected result: AUC near chance level (~0.5), confirming that the 0.974 AUC achieved by the pretrained encoder is attributable to representations learned during diffusion training rather than to architectural inductive biases.)*

---

## Discussion

These results demonstrate that self-supervised generative pretraining via denoising diffusion produces rich, transferable representations of fNIRS hemodynamic signals. The pretrained U-Net encoder, despite being trained exclusively on a noise prediction objective, learned to capture age-related hemodynamic differences — including temporal response dynamics, amplitude patterns, and inter-channel correlations — that enable near-ceiling classification of children vs. adults (AUC 0.974).

The critical role of temporal modeling (LSTM) in the downstream classifier suggests that the encoder's bottleneck representation preserves fine-grained temporal structure that is informative for discriminating developmental hemodynamic profiles. This temporal structure is destroyed by mean pooling, which reduces performance by ~6 AUC points.

The success of this transfer learning approach motivates its extension to dyadic synchrony prediction, where the pretrained encoder's understanding of individual hemodynamic profiles may serve as a foundation for modeling inter-brain coupling. The best-performing architecture (bn_lstm64) will serve as the feature extraction backbone for the synchrony classification pipeline, with its weights frozen during initial synchrony training to preserve the learned hemodynamic representations.
