# AcousticTap (V1: Baseline ML) — Turning a Laptop Palm Rest into Macro Buttons

> **Project Log — Chapter 1 (`v1-baseline-ml` branch):** This branch documents the first generation of AcousticTap. Here, we set out to prove a simple idea: can we use a laptop's built-in microphone and classical machine learning to turn the empty aluminum spaces next to the trackpad into programmable `Copy` and `Paste` buttons or anything else? 
> 
> *For the second generation—where we solved the real-world typing and background noise problem using Lamb-wave dispersion physics and Fourier triangulation—switch to Chapter 2 on the `v2-hybrid-fourier` branch.*

---

## 1. The Idea & Problem Statement

If you watch how people use a laptop, almost all physical interaction happens on the keyboard and the trackpad. Meanwhile, the two large slabs of machined aluminum flanking the trackpad do nothing at all—they just sit there under your wrists while you break your typing rhythm hundreds of times a day to hit `Cmd+C` and `Cmd+V`.

Other projects have tried turning the laptop body into an input surface before, but they usually rely on adding hardware: sticking piezo contact mics to the case, wiring up capacitive copper tape, or blasting ultrasonic chirps through stereo speaker arrays. Practically speaking, nobody wants to glue sensors to their MacBook. 

Our goal with **AcousticTap V1** was to build virtual tap zones on the left and right palm rests using **zero external hardware**—just the standard, single-channel internal microphone already sitting inside any off-the-shelf laptop.

---

## 2. Why Single-Channel Audio Can "Feel" a Tap

When you tap the aluminum palm rest with your finger, you aren't just making a sound in the air; you are sending a mechanical stress wave directly through the metal unibody. Because the laptop's internal microphone is physically mounted to the inside of the chassis, it acts like an accidental seismograph, picking up both airborne sound and structure-borne vibrations.

Crucially, the inside of a laptop isn't an empty box. A tap on the **left palm rest** has to travel past a different set of physical obstacles (the left speaker housing, logic board mounts, and battery cells) to reach the microphone than a tap on the **right palm rest**. Those internal cavities and boundaries filter the vibration differently depending on where you strike.

### Hardware Compatibility (Why It Runs on Any MacBook Air)
Unlike academic systems that depend on multi-microphone phase arrays or ultrasonic transducers, we designed the V1 pipeline around strict hardware minimalism:
- **Single-channel capture (`channels: 1`)** at a standard $44.1\text{ kHz}$ sample rate.
- **No ultrasonic emissions** or stereo phase tricks required.
- **Lightweight CPU inference** taking under $6\text{ ms}$, so it runs silently in the background without draining the battery.
- **OS-aware shortcut routing** that automatically dispatches `Cmd` macros on macOS and `Ctrl` macros on Windows and Linux.

---

## 3. Core Objectives for V1

1. **Three-Way Classification:** Reliably tell the difference between a left palm rest tap, a right palm rest tap, and everyday background noise.
2. **Instantaneous Tactile Feel:** Keep total end-to-end latency under the $20\text{ ms}$ real-time audio threshold so triggering a macro feels like pressing a physical button.
3. **Typing Guardrails:** Monitor active keyboard events so normal typing doesn't accidentally fire off `Copy` and `Paste` commands.
4. **Modular Benchmarking:** Build a clean Python pipeline where detection, windowing, and feature extraction can be profiled stage by stage.

---

## 4. Tech Stack & Key Libraries

We built the pipeline in Python using standard scientific computing and audio libraries:

- **`sounddevice`**: Streams real-time audio blocks from the internal mic with minimal buffer latency via PortAudio bindings.
- **`numpy`**: Powers the pre-trigger ring buffer, fast array slicing, and asymmetric cosine windowing.
- **`librosa`**: Extracts time-frequency audio features—specifically Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, rolloff, flatness, and zero-crossing rate.
- **`scikit-learn`**: Provides our classical ML classifiers (RBF-SVM, Random Forest, Extra Trees, Logistic Regression), feature scalers, stratified cross-validation, and ANOVA tools.
- **`joblib`**: Serializes trained model pipelines and scalers for fast startup in the background daemon.
- **`pynput`**: Listens for physical keyboard activity to suppress tap detection while typing, and injects synthetic `Cmd+C` / `Cmd+V` keystrokes when a valid tap lands.
- **`pyyaml`**: Manages sample rates, RMS trigger thresholds, and keybindings in a single config file.
- **`matplotlib`**: Generates our spectrograms and feature importance plots.

---

## 5. Inside the V1 Signal & Feature Pipeline

When the ring buffer detects a sharp rise in root-mean-square (RMS) energy above the dynamic exponential moving average (EMA) noise floor, it captures a $128\text{ ms}$ transient window aligned around the impact peak.

Rather than feeding that entire $128\text{ ms}$ clip into `librosa` as one blurry average, we split the transient into two distinct physical phases:
1. **The Onset Attack (first $5.8\text{ ms}$):** Captures the initial ballistic strike before room echoes have time to bounce back into the microphone.
2. **The Resonance Decay Tail:** Captures how the aluminum shell and internal cavities "ring" after the finger leaves the surface.

From each captured event, we extract a **70-dimensional feature vector**:
- **20 Onset MFCCs** (attack envelope shape)
- **20 Decay MFCCs** (chassis reverberation tail)
- **20 Standard Deviation MFCCs** (how the spectrum shifts over the window)
- **10 Spectral & Temporal Dynamics** (Spectral Centroid, Spectral Rolloff, Spectral Flatness, Zero-Crossing Rate, Crest Factor, and Attack-to-Decay Ratio)

### Visualizing the Signal: Left Tap vs. Right Tap vs. Keyboard

Before training any models, we wanted to see if the acoustic difference was actually visible to the naked eye. Below is a side-by-side spectrogram comparison recorded on a MacBook chassis:

![Comparative Spectrograms](reports/spectrogram_comparison.png)

Looking at the three spectrograms, two patterns stand out right away:
1. **Left vs. Right Modal Shift:** Both palm rest taps pack their energy into the lower frequencies ($150\text{ Hz}$ to $2500\text{ Hz}$), but they don't look the same. A **Left Palm Rest Tap** (closer to the internal mic and left speaker enclosure) lights up a tight, intense low-frequency band between $150\text{ Hz}$ and $1500\text{ Hz}$. A **Right Palm Rest Tap** has to travel across the trackpad cutout, shifting its resonance peaks higher ($300\text{ Hz}$ to $2200\text{ Hz}$) with noticeably more structural damping.
2. **Palm Tap vs. Spacebar Strike:** Hitting the keyboard spacebar creates a broadband airborne click that smears high-frequency friction energy all the way past $8\text{ kHz}$ to $10\text{ kHz}+$, whereas structure-borne palm taps stay concentrated in the low-frequency flexure bands of the aluminum plate.

---

## 6. What the Baseline Model Actually Learned

We trained several classifiers (Logistic Regression, Random Forest, Extra Trees, and RBF-SVM) on our 70-dimensional feature set. To understand *how* the tree-based models were telling left and right taps apart, we ran a Random Forest Gini importance analysis alongside one-way ANOVA F-tests across all 70 features.

### Top 15 Most Discriminative Features (Gini & ANOVA Rankings)

| Rank | Feature Name | Domain Category | Random Forest Importance | ANOVA F-Score | Physical Interpretation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **#01** | `Decay_MFCC_08` | Resonance Decay | **8.24%** | 31.8 | Cavity resonance and speaker grille boundary reflections. |
| **#02** | `Decay_MFCC_07` | Resonance Decay | **6.74%** | 472.6 | Cavity resonance and speaker grille boundary reflections. |
| **#03** | `Decay_MFCC_03` | Resonance Decay | **5.05%** | 307.7 | Cavity resonance and speaker grille boundary reflections. |
| **#04** | `Decay_MFCC_11` | Resonance Decay | **5.01%** | 146.8 | Cavity resonance and speaker grille boundary reflections. |
| **#05** | `Std_MFCC_02` | Spectral Variance | **4.23%** | 229.1 | Frequency variation over time across the impact window. |
| **#06** | `Onset_MFCC_12` | Onset Attack | **4.15%** | 95.0 | Ballistic strike energy distribution before chassis reflections. |
| **#07** | `Decay_MFCC_17` | Resonance Decay | **4.15%** | 433.5 | Cavity resonance and speaker grille boundary reflections. |
| **#08** | `Decay_MFCC_12` | Resonance Decay | **4.02%** | 235.6 | Cavity resonance and speaker grille boundary reflections. |
| **#09** | `Decay_MFCC_02` | Resonance Decay | **3.67%** | 67.7 | Cavity resonance and speaker grille boundary reflections. |
| **#10** | `Onset_MFCC_11` | Onset Attack | **3.54%** | 165.6 | Ballistic strike energy distribution before chassis reflections. |
| **#11** | `Onset_MFCC_07` | Onset Attack | **3.43%** | 128.8 | Ballistic strike energy distribution before chassis reflections. |
| **#12** | `Decay_MFCC_10` | Resonance Decay | **3.14%** | 315.7 | Cavity resonance and speaker grille boundary reflections. |
| **#13** | `Decay_MFCC_15` | Resonance Decay | **2.80%** | 187.5 | Cavity resonance and speaker grille boundary reflections. |
| **#14** | `Decay_MFCC_04` | Resonance Decay | **2.65%** | 161.0 | Cavity resonance and speaker grille boundary reflections. |
| **#15** | `Onset_MFCC_17` | Onset Attack | **2.59%** | 217.5 | Ballistic strike energy distribution before chassis reflections. |

![Feature Importance Ranking](reports/feature_importance.png)

### What This Chart Told Us (And Why It Foreshadowed a Problem)
1. **Heavy Reliance on the Decay Tail:** Look closely at the top four bars in the chart (`Decay_MFCC_08`, `07`, `03`, and `11`). Ten of the top fifteen features belong to the **Resonance Decay** category. In other words, the baseline ML model was making a huge portion of its decision based on how the laptop *echoed* after the tap.
2. **The Value of the Onset Window:** The `Onset_MFCC` features (capturing the first $5.8\text{ ms}$) still accounted for over $40\%$ of the overall decision weight across the full 70-feature set, giving us a solid anchor before room reflections arrived.
3. **Intensity Invariance:** Mel-scale normalization combined with the Attack-to-Decay Ratio (ADR) and Crest Factor helped the model ignore *how hard* you tapped and focus on the spectral shape.

---

## 7. Real-Time Latency & Surface Damping Benchmarks

### 7.1 Real-Time Pipeline Latency Breakdown
A macro button is useless if it lags. We profiled the entire V1 pipeline across 100 iterations on Apple Silicon (Darwin 27.0.0) to see how fast we could go from a physical finger strike to an OS keystroke dispatch:

| Pipeline Stage | Implementation Details | Mean (ms) | Std Dev (ms) | 95th Percentile (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **0. Hardware Buffer Acquisition** | 512 samples @ $44.1\text{ kHz}$ (Single audio block) | **11.61** | *Fixed* | 11.61 |
| **1. Transient Impulse Detection** | Ring buffer RMS check + dynamic EMA floor | 0.021 | 0.004 | 0.024 |
| **2. Peak Alignment & Windowing** | Asymmetric cosine onset window (peak @ 32) | 0.047 | 0.004 | 0.049 |
| **3. Audio Feature Extraction** | 20 MFCCs (attack/decay) + Spectral Dynamics | 1.424 | 0.027 | 1.453 |
| **4. Classifier Inference** | Model probability prediction (`RandomForest`) | 3.838 | 0.083 | 3.924 |
| **5. Macro & Suppression Logic** | Active keyboard check & shortcut routing | 0.001 | 0.000 | 0.001 |
| | | | | |
| **Total Algorithmic Processing** | CPU calculation time | **5.33 ms** | $\pm 0.11\text{ ms}$ | 6.40 ms |
| **Total End-to-End Latency** | Physical finger strike to OS key dispatch | **16.94 ms** | $\pm 0.03\text{ ms}$ | 18.01 ms |

- **Human Tactile Perception Threshold:** $\approx 50.0\text{ ms}$
- **Standard Real-Time Audio Deadline:** $20.0\text{ ms}$
- **AcousticTap Headroom:** **$3.06\text{ ms}$ margin** under the $20\text{ ms}$ real-time budget.

With only $5.33\text{ ms}$ spent in Python CPU execution, the response feels genuinely instantaneous.

### 7.2 Surface Boundary Damping Experiments
Because our feature importance analysis showed that the model loved `Decay_MFCC` reverberation features, we asked a practical question: *what happens when you take the laptop off a hard desk and put it on your lap?*

| Surface Condition | Acoustic Boundary Physics | Resonance Decay ($T_{60}$) | Attack/Decay Ratio | Classification Accuracy | Failure Mode / Observation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Hard Wooden Desk** | Rigid boundary. Minimal chassis absorption, high structural ringing. | **13.8 ms** | **7.8 dB** | **88.9%** | Optimal baseline. Clear separation between onset and reverberation decay. |
| **Soft Mousepad / Mat** | Viscoelastic boundary. Neoprene absorbs high-frequency chassis flexure. | **10.6 ms** | **10.3 dB** | **82.2%** | Decay tail attenuated by $\approx 30\%$. Slight drop in right-tap resonance distinction. |
| **User Lap (Fabric / Flesh)** | High mechanical damping. Body tissue absorbs bottom case vibrations. | **7.1 ms** | **13.4 dB** | **75.6%** | Resonance decay squashed quickly. High-confidence taps require firmer strike. |

Resting the laptop on a soft lap acted like a giant acoustic damper on the bottom case, cutting the $T_{60}$ ringing time nearly in half ($13.8\text{ ms} \rightarrow 7.1\text{ ms}$) and dropping accuracy from $88.9\%$ down to $75.6\%$. Separating the $5.8\text{ ms}$ onset window kept the system usable on soft surfaces, but the vulnerability of the decay tail was becoming clear.

---

## 8. Where V1 Hit a Wall: The Real-World Noise Problem

On paper—and in a quiet bedroom on a wooden desk—V1 looked great. Achieving **$88.9\%$ accuracy** with zero external hardware felt like a solid milestone.

Then we actually left the background daemon running while doing real work. Within twenty minutes, the limitations of a pure MFCC machine learning approach became painfully obvious:
1. **Typing Clatter:** Even with software suppression during active keystrokes, the stray mechanical clicks of keys, trackpad clicks, and thumb brushes triggered the RMS detector and fooled the classifier.
2. **Desk Thumps:** Setting down a coffee mug, bumping the table, or resetting a wireless mouse sent low-frequency thuds into the chassis.
3. **Cafe & Room Conversation:** Human speech and background music at $\approx 65\text{ dB SPL}$ constantly warped the `Decay_MFCC` and Spectral Centroid values.

### Environmental Interference Benchmarks (`v1-baseline-ml`)

| Test Condition | Environment Description | Overall Accuracy | False Positive Rate (Noise marked as Tap) | Primary Failure Mode |
| :--- | :--- | :---: | :---: | :--- |
| **Quiet Room (Baseline)** | Normal desk setting, no background conversation | **88.9%** | 20.0% (3/15) | Ambiguity between left and right palm rest modes |
| **Active Typing** | Continuous typing on the laptop keyboard during detection | **71.1%** | **46.7% (7/15)** | Mechanical key strikes falsely trigger the impulse detector |
| **Loud Ambient (Cafe / Speech)** | Background speech and music ($\approx 65\text{ dB SPL}$) | **57.8%** | **60.0% (9/15)** | Airborne noise distorts spectral centroid and MFCC values |

### Why Tuning Classical ML Models Couldn't Fix This
We spent days trying to tune our way out of this table using standard ML techniques:
- **Logistic Regression** underfit badly because acoustic reflections inside a contoured aluminum shell are highly non-linear.
- **Random Forests & Extra Trees** aced the clean training set, but they essentially memorized the quiet room's background noise floor. Walk into a noisy room, and their F1 score tanked.
- **RBF-Kernel SVMs** gave us our best V1 decision boundaries, yet still suffered a **$46.7\%$ false positive rate** during typing tests whenever a sharp key click happened to share a similar spectral envelope with a palm tap.

**The Root Cause:** Generic audio features like MFCCs were invented for speech recognition in open air. They treat the MacBook's microphone like a studio vocal mic—they have **zero concept of solid-state physics**. They don't know whether a sound wave traveled through the air at $343\text{ m/s}$ or flexed through a $1.2\text{ mm}$ sheet of aircraft-grade aluminum.

> **Next Step $\rightarrow$ Chapter 2 (`v2-hybrid-fourier`):** Realizing that pure pattern recognition wasn't enough pushed us to build **AcousticTap V2**, where we model physical **Lamb-wave dispersion** and **12-band Fourier spatial resonance** to verify that an impulse actually traveled through the metal chassis—cutting our active-typing false positive rate from $46.7\%$ down to **under $8.5\%$**.

---

## 9. How to Run the V1 Baseline Branch

### Setup
Install dependencies using `uv`:
```bash
uv sync
```

### 1. Collect Training Samples
You can record samples class-by-class or in an interleaved group session (which cycles randomly in bursts of 3 to 4 taps so the model doesn't overfit to background noise from a single recording take):
```bash
# Group interleaved mode (recommended):
uv run python -m src.holo.collect --label all --samples 50

# Or single-class mode:
uv run python -m src.holo.collect --label left_tap --samples 50
uv run python -m src.holo.collect --label right_tap --samples 50
uv run python -m src.holo.collect --label noise --samples 50
```

### 2. Train the Baseline ML Models
Train and cross-validate the classical ML classifiers on the 70-dimensional MFCC/spectral feature vector:
```bash
uv run python -m src.holo.train
```

### 3. Evaluate on the Holdout Set
Run holdout evaluation and inspect the baseline confusion matrix:
```bash
uv run python -m src.holo.evaluate --holdout-only
```

### 4. Run the Live Listener Daemon
Test live tap recognition in dry-run mode (prints predictions to the terminal without firing OS shortcuts):
```bash
uv run python -m src.holo.listen --dry-run
```

Start the active background daemon:
```bash
uv run python -m src.holo.listen --confidence 0.70
```
- **Left Palm Rest Tap** $\rightarrow$ Triggers **Copy** (`Cmd+C` on macOS, `Ctrl+C` on Windows/Linux).
- **Right Palm Rest Tap** $\rightarrow$ Triggers **Paste** (`Cmd+V` on macOS, `Ctrl+V` on Windows/Linux).

---

## 10. V1 Repository Structure

```text
├── MAC_TESTER_GUIDE.md           # Setup & microphone permission guide for macOS testers
├── README.md                     # Chapter 1: Baseline ML documentation (this file)
├── pyproject.toml                # Project dependencies and metadata
├── dataset/                      # Recorded .npy transient audio samples
├── models/                       # Serialized baseline ML models and scalers
├── reports/                      # V1 benchmark tables and diagnostic visualizations
│   ├── feature_importance.md     # Full 70-feature ANOVA & Gini importance report
│   ├── feature_importance.png    # Top 15 discriminative feature bar chart
│   ├── latency_breakdown.md      # Real-time stage-by-stage latency measurements
│   ├── spectrogram_comparison.png# Left vs. Right vs. Key Strike spectrogram plot
│   └── surface_damping_analysis.md # Desk vs. Mousepad vs. Lap boundary benchmarks
└── src/holo/                     # Core V1 source code
    ├── config.py                 # Thresholds, audio constants, and YAML loader
    ├── detector.py               # Pre-trigger ring buffer & RMS impulse detector
    ├── features.py               # 70-dim Onset/Decay MFCC and spectral extractor
    ├── collect.py                # Interactive CLI for recording tap datasets
    ├── train.py                  # Classical ML training & hyperparameter search
    ├── evaluate.py               # Holdout evaluation and metric reporting
    └── listen.py                 # Real-time listener daemon with typing suppression
```