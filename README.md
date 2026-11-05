# AcousticTap: Hybrid Fourier Palm Rest Tap Recognition Using Structure-Borne Acoustics

AcousticTap turns the aluminum palm rest of a laptop into programmable macro buttons without adding sensors, contact microphones, or external hardware. By capturing the acoustic transients generated when a finger strikes the chassis, the system classifies tap locations and triggers operating system shortcuts such as Copy and Paste.

In this second generation (V2), we solve the acoustic noise interference problem of baseline machine learning by combining time-frequency feature extraction with physics-based Fourier Transform triangulation.

---

## 1. Problem Statement

Most laptop workflows rely entirely on the keyboard and trackpad. The aluminum areas flanking the trackpad remain idle during normal use, while users repeatedly interrupt typing to reach for two-key combinations like Cmd+C and Cmd+V.

Adding external hardware, such as capacitive stickers, piezo sensors, or auxiliary microphones, increases manufacturing cost, adds physical bulk, and makes software distribution impractical. The goal of this project is to create virtual tap zones on the palm rest using only the hardware already built into standard laptops.

---

## 2. Approach and Hardware Compatibility

### How We Arrived at the Solution
Tapping an aluminum palm rest sends mechanical stress waves directly through the laptop body. Because the internal microphone is physically mounted inside the chassis, it records both airborne sound and structure-borne vibrations.

When you strike the left side of the palm rest, the mechanical wave path to the microphone differs from a strike on the right side. The chassis material, internal battery placement, display hinge, and internal cavities shape the resulting sound. We capture this short acoustic transient, extract frequency and dispersion features, and use a hybrid classifier to identify where the tap occurred.

### Why It Works on Any MacBook Air
Many acoustic gesture recognition projects require stereo microphone arrays, specialized ultrasound emitters, or high-speed data acquisition boards.

Our system runs entirely on standard single-channel microphone hardware, making it compatible with any standard MacBook Air.
- It requires only a single microphone channel (`channels: 1`) sampled at the standard rate of 44.1 kHz.
- It does not rely on stereo phase differences or ultrasound emissions.
- It runs locally on the CPU with inference latency under 6 milliseconds, consuming negligible system resources.
- It adapts automatically to the host operating system, using `Cmd` shortcuts on macOS and `Ctrl` shortcuts on Windows or Linux.

---

## 3. Project Objectives

1. Accurately distinguish between left palm rest taps, right palm rest taps, and ambient background noise.
2. Eliminate false positive triggers caused by typing key clatter, desk thumps, and room conversation.
3. Maintain low real-time latency so triggering a macro feels instantaneous (under 20 ms end-to-end).
4. Prevent accidental activations during typing through active keyboard suppression.
5. Provide a modular pipeline where signal conditioning, feature extraction, and triangulation can be benchmarked independently.

---

## 4. Key Libraries Used

The implementation relies on standard Python scientific computing and signal processing libraries:

- **`sounddevice`**: Manages real-time audio input streams with low buffer latency using PortAudio bindings.
- **`numpy`**: Handles ring buffers, fast vector math, signal windowing, and Fourier transforms.
- **`librosa`**: Extracts audio features including Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, rolloff, flatness, and zero-crossing rates.
- **`scikit-learn`**: Provides classification algorithms (SVM, Random Forest, Extra Trees), Linear Discriminant Analysis, stratified cross-validation, and performance metrics.
- **`joblib`**: Serializes trained models, scalers, and fusion pipelines for fast runtime loading.
- **`pynput`**: Monitors system keyboard events to suppress tap detection while typing, and synthesizes macro keystrokes when a valid tap is detected.
- **`pyyaml`**: Reads system thresholds, sample rates, and shortcut mappings from a centralized configuration file.
- **`matplotlib`**: Generates experimental validation figures, comparative spectrograms, and feature importance charts.

---

## 5. Why Baseline Machine Learning Was Not Optimal: The Noise Interference Problem

In the baseline version of AcousticTap (`v1-baseline-ml`), we relied solely on generic audio features (MFCCs and spectral moments) fed into classical machine learning classifiers.

While that baseline worked in quiet environments (88.9% accuracy on isolated taps), it degraded rapidly in realistic computer environments:
1. **Key clatter while typing**: Mechanical key strikes generate sharp acoustic transients that classical time-domain detectors easily mistake for palm taps.
2. **Desk vibrations**: Moving a mouse, placing a cup, or bumping the desk introduces low-frequency impulses.
3. **Ambient speech and background sounds**: Sudden airborne noises trigger the amplitude detector.

### Environmental Interference Benchmarks (Baseline ML)

| Test Condition | Environment Description | Overall Accuracy | False Positive Rate (Noise marked as Tap) | Primary Failure Mode |
| :--- | :--- | :---: | :---: | :--- |
| **Quiet Room (Baseline)** | Normal desk setting, no background conversation | **88.9%** | 20.0% (3/15) | Ambiguity between left and right palm rest modes |
| **Active Typing** | Continuous typing on the laptop keyboard during detection | **71.1%** | 46.7% (7/15) | Mechanical key strikes falsely trigger the impulse detector |
| **Loud Ambient (Cafe / Speech)** | Background speech and music (~65 dB SPL) | **57.8%** | 60.0% (9/15) | Airborne noise distorts spectral centroid and MFCC values |

### Why Pure Machine Learning Models Failed to Solve This
We evaluated several standard classifiers on these time-frequency features:
- **Logistic Regression**: Underfit the data because acoustic reverberation paths on a metal plate exhibit non-linear frequency interactions.
- **Random Forest & Extra Trees**: Performed well on clean training data but memorized ambient background levels, causing accuracy drops when room acoustics shifted.
- **Support Vector Machines (RBF Kernel)**: Achieved the highest baseline accuracy, but still misclassified key clicks and desk bumps when their spectral envelopes overlapped with tap frequencies.

The fundamental limitation is that generic MFCCs treat the laptop microphone like an open-air audio recorder. They do not model how mechanical stress waves travel across a solid aluminum plate.

---

## 6. The V2 Solution: Physical Acoustic Wave Propagation and Fourier Triangulation

To solve the noise interference problem without adding sensors, we looked into plate wave propagation. 

When a mechanical impulse strikes a thin metal plate, it generates flexural waves that propagate through the medium according to physical dispersion laws. We exploit two acoustic phenomena: Lamb-wave dispersion and spatial resonance modes.

```
                           [Palm Rest Strike]
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   [Lamb-Wave Dispersion]                    [Spatial Resonance Modes]
   High frequencies travel faster            Chassis boundaries produce 
   than low frequencies                      distinct nodal frequency bands
   (Delta t_disp = t_LF - t_HF)              (Left: 129 Hz, Right: 172 Hz)
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                      [Fourier Triangulation]
                      43-Dimensional Physical
                      Acoustic Feature Vector
```

### 1. Lamb-Wave Dispersion Arrival Timing
In thin solid plates, flexural waves are dispersive: phase velocity is proportional to the square root of frequency ($c_p \propto \sqrt{\omega}$). As a direct consequence, high-frequency wave components (3 kHz to 14 kHz) travel faster than low-frequency components (150 Hz to 900 Hz).

This produces a measurable arrival delay between the high-frequency and low-frequency wave envelopes:

$$\Delta t_{disp} = t_{LF} - t_{HF}$$

Because the internal microphone on a MacBook sits at an asymmetric location relative to the two palm rests, the mechanical path lengths differ. On Apple Silicon hardware, we measured the following arrival characteristics:

| Waveform Metric | Left Palm Tap | Right Palm Tap | Difference ($\Delta$) | Physical Acoustic Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **High-Frequency Wave Arrival ($t_{HF}$)** | **0.77 ms** | **4.13 ms** | **3.36 ms** | Direct path propagation to internal microphone |
| **Low-Frequency Wave Arrival ($t_{LF}$)** | **0.91 ms** | **0.88 ms** | **0.02 ms** | Chassis flexure resonance envelope |
| **Lamb Dispersion Delay ($\Delta t_{disp}$)** | **+0.14 ms** | **-3.24 ms** | **3.38 ms** | Path length alters frequency group velocities |

Airborne sounds (room speech, music) travel through air at ~343 m/s as pure compression waves without plate flexure. They show near-zero dispersion delay ($\Delta t_{disp} \approx 0$). This provides a physical test to reject airborne noise before it reaches the classifier.

### 2. Spatial Resonance Mode Distribution
The internal structure of the MacBook (battery packs, trackpad frame, display hinge, speaker grilles) forms boundary conditions that shape resonant standing waves. 

By computing the Fast Fourier Transform (FFT) across a 1024-sample post-transient window, we divide the spectrum into 12 log-spaced spatial resonance bands (150 Hz to 16 kHz):
- **Left Palm Taps** excite dominant resonance modes at **129 Hz**, coupling strongly into lower chassis flexure modes.
- **Right Palm Taps** excite dominant resonance modes at **172 Hz**, producing elevated energy in the mid-frequency resonance bands.

Sub-band energy ratios and group delays provide a 43-dimensional physical acoustic fingerprint that distinguishes left taps from right taps regardless of strike force.

### 3. Fourier Triangulator Classifier
The Fourier branch computes a 43-dimensional vector (12 onset band energies, 12 wake band energies, 11 sub-band ratios, Lamb dispersion lag, HF/LF energy ratio, group delays, and signal-to-noise ratio). 

This vector is evaluated by `FourierTriangulator`, which combines:
- **Linear Discriminant Analysis (LDA)** with automatic shrinkage, projecting the spatial features into a discriminative subspace.
- **Cosine Similarity Template Matching** with temperature calibration ($T = 2.5$) against calibrated spatial centroids.

The Fourier branch outputs its own independent class probabilities:

$$P_{Fourier} = \alpha P_{LDA} + (1 - \alpha) P_{Cosine}$$

---

## 7. The Hybrid Architecture: Probability Fusion

Rather than discarding the baseline machine learning model, V2 runs both pipelines in parallel on the same captured transient:

```
                          [Chassis Transient (128 ms)]
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
      [ML Feature Extraction]                 [Fourier Triangulation]
      70 Dimensions:                          43 Dimensions:
      - 20 Onset MFCCs                        - 12 Log Onset Bands
      - 20 Decay MFCCs                        - 12 Log Wake Bands
      - 20 Std MFCCs                          - 11 Sub-band Ratios
      - Spectral Centroid, Rolloff,           - Lamb Dispersion Delay (Delta t)
        Flatness, Crest Factor, ADR           - HF/LF Energy Ratio & Group Delay
                   │                                       │
                   ▼                                       ▼
      [Tuned ML Classifier]                   [Fourier Triangulator]
      (RBF-SVM / Random Forest)               (LDA + Cosine Temperature)
                   │                                       │
                   ▼                                       ▼
             P_ML Probabilities                   P_Fourier Probabilities
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                           [Probability Blending]
                     P_hybrid = (1 - w)*P_ML + w*P_Fourier
                                       │
                                       ▼
                           [Confidence & Suppression]
                         Threshold >= 0.70 & No Typing
                                       │
                                       ▼
                            [Dispatch OS Shortcut]
                              (Cmd+C / Cmd+V)
```

The combined probability vector is computed as:

$$P_{hybrid} = (1 - w) \cdot P_{ML} + w \cdot P_{Fourier}$$

During training, the fusion weight $w$ is searched over a cross-validation grid (`weight_search_grid`). The weight yielding the highest macro F1 score is selected (typically $w \approx 0.50$, giving equal voice to physical acoustics and pattern recognition).

---

## 8. Experimental Evaluation and Confusion Matrix

We evaluated the hybrid model on a full dataset of 589 recorded events collected on a MacBook Air palm rest under varied conditions.

### Confusion Matrix (Holdout Test Set, 118 samples)

| Actual \ Predicted | Predicted Left Tap | Predicted Right Tap | Predicted Noise | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Left Tap** | **36** | 2 | 5 | 43 |
| **Actual Right Tap** | 1 | **39** | 0 | 40 |
| **Actual Noise** | 2 | 3 | **30** | 35 |
| **Total Predicted** | 39 | 44 | 35 | 118 |

### Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Left Tap** | 0.923 | 0.837 | 0.878 | 43 |
| **Right Tap** | 0.886 | 0.975 | 0.929 | 40 |
| **Noise** | 0.857 | 0.857 | 0.857 | 35 |
| | | | | |
| **Overall Accuracy** | | | **0.8898** | 118 |
| **Macro Average** | 0.889 | 0.890 | 0.888 | 118 |
| **Weighted Average** | 0.891 | 0.890 | 0.889 | 118 |

- **5-Fold Cross-Validation Macro F1:** `0.8771`
- **Winning Architecture:** `Hybrid_SVM_RBF` (50% ML + 50% Fourier Triangulation)
- **Best Tuned Hyperparameters:** `{'clf__C': 5.0, 'clf__class_weight': {0: 1.3, 1: 1.0, 2: 1.0}, 'clf__gamma': 'scale'}`

### Why Hybrid Fourier Overcomes the Noise Problem
In active typing tests, false positive triggers dropped from 46.7% in the baseline model to under 8.5% in the hybrid model. Mechanical key clicks lack the characteristic plate dispersion delay and do not match the chassis resonance profile, causing the Fourier branch to output near-zero tap probabilities.

---

## 9. Academic Experimental Reports and Validation Suite

To verify system performance on real hardware, we ran our automated benchmark suite (`benchmark_report.py`) on an Apple Silicon MacBook Air (Darwin arm). All generated plots and Markdown reports are saved in the `reports/` folder:

1. **Comparative Spectrogram Analysis** ([`reports/spectrogram_comparison.png`](reports/spectrogram_comparison.png)): Visualizes time-frequency energy distribution across Left Palm Taps, Right Palm Taps, and Spacebar Strikes, confirming that key strikes concentrate energy above 8 kHz while palm taps excite chassis flexure below 2.5 kHz.
2. **Real-Time Latency Breakdown** ([`reports/latency_breakdown.md`](reports/latency_breakdown.md)): Profiles execution time across all pipeline stages, measuring an end-to-end latency of **16.94 ms** (Algorithmic: 5.33 ms, Hardware buffer: 11.61 ms), cleanly within the 20 ms interactive threshold.
3. **Surface Boundary Damping Analysis** ([`reports/surface_damping_analysis.md`](reports/surface_damping_analysis.md)): Measures acoustic decay times ($T_{60}$) and attack-to-decay ratios when the laptop rests on a Wooden Desk (88.9%), a Mousepad (82.2%), and a User Lap (75.6%).
4. **Feature Importance Ranking** ([`reports/feature_importance.png`](reports/feature_importance.png) and [`reports/feature_importance.md`](reports/feature_importance.md)): Ranks the most discriminative features using Random Forest Gini importances and ANOVA F-scores.
5. **Fourier Transform & Dispersion Timing** ([`reports/fourier_dispersion_timing.png`](reports/fourier_dispersion_timing.png), [`reports/fourier_frequency_spectrum.png`](reports/fourier_frequency_spectrum.png), and [`reports/fourier_analysis_report.md`](reports/fourier_analysis_report.md)): Documents the physical acoustic measurements, Lamb-wave dispersion delay ($\Delta t_{disp} = 3.38$ ms delta), and 12-band spatial resonance distributions.
6. **Consolidated Academic Report** ([`reports/instructor_expansion_report.md`](reports/instructor_expansion_report.md)): Master summary document integrating all benchmark findings.

### Running the MacBook Validation Suite
MacBook users can re-run the benchmark suite locally with a single command:
```bash
# Automated run (generates all reports in ~5 seconds):
uv run python benchmark_report.py

# Interactive guided run (prompts you to tap your MacBook step-by-step):
uv run python benchmark_report.py --interactive
```

For setup instructions and macOS microphone permission guidance, see [`MAC_TESTER_GUIDE.md`](MAC_TESTER_GUIDE.md).

---

## 10. How to Use

### Setup
Install dependencies with `uv`:
```bash
uv sync
```

### 1. Data Collection
Collect training samples using either randomized group mode or single-class mode.

Group interleaved collection (cycles randomly in bursts of 3 to 4 taps to eliminate session bias):
```bash
uv run python -m src.holo.collect --label all --samples 50
```

Single-class collection:
```bash
uv run python -m src.holo.collect --label left_tap --samples 50
uv run python -m src.holo.collect --label right_tap --samples 50
uv run python -m src.holo.collect --label noise --samples 50
```

*Note: CLI flags support cross-compatible aliases (`--class` for `--label`, `--count` for `--samples`).*

### 2. Hybrid Model Training and Benchmarking
Train and benchmark all hybrid architectures (SVM, Random Forest, Extra Trees, each combined with the Fourier branch):
```bash
uv run python -m src.holo.train --features enhanced
```

To fix the Fourier fusion weight instead of auto-tuning (for example, 50% ML and 50% Fourier):
```bash
uv run python -m src.holo.train --features enhanced --fourier-weight 0.50
```

Training automatically:
- Evaluates 5-fold cross-validation F1 across candidate classifiers and fusion weights.
- Generates a timestamped candidate bundle in `models/`.
- Updates `models/leaderboard.csv`.
- Saves the winning production model for the listener daemon.

### 3. Model Evaluation
Evaluate the production model on the holdout test set:
```bash
uv run python -m src.holo.evaluate --holdout-only
```

### 4. Live Listener Daemon
Test live predictions safely in dry-run mode (logs predictions without dispatching OS keystrokes):
```bash
uv run python -m src.holo.listen --dry-run
```

Run the live listener daemon:
```bash
uv run python -m src.holo.listen --confidence 0.70
```
- Left palm rest tap triggers **Copy** (`Cmd+C` on macOS, `Ctrl+C` on Windows/Linux).
- Right palm rest tap triggers **Paste** (`Cmd+V` on macOS, `Ctrl+V` on Windows/Linux).
