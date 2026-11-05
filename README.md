# AcousticTap: Palm Rest Tap Recognition Using Structure-Borne Acoustics

AcousticTap turns the surface of a laptop palm rest into programmable macro buttons without requiring external sensors, piezos, or hardware modifications. By capturing the acoustic transients generated when a user taps the chassis, the system classifies tap locations and triggers operating system shortcuts such as Copy and Paste.

---

## 1. Problem Statement

Most laptop interaction relies entirely on the keyboard and trackpad. The aluminum areas flanking the trackpad remain idle during normal use. At the same time, users repeatedly interrupt their workflow to execute repetitive two-key combinations like Cmd+C and Cmd+V.

Adding physical hardware (such as external buttons, contact microphones, or capacitive strips) increases cost, adds bulk, and makes adoption difficult. The goal of this project is to create virtual tap zones on the palm rest using only the hardware already built into modern laptops.

---

## 2. Approach and Hardware Compatibility

### How We Arrived at the Solution
Tapping an aluminum palm rest sends mechanical stress waves directly through the laptop body. Because the internal microphone is physically mounted inside the chassis, it records both airborne sound and structure-borne vibrations. 

When you strike the left side of the palm rest, the mechanical wave path to the microphone differs from a strike on the right side. The chassis material, internal battery placement, and internal air cavities shape the resulting sound. We can capture this short burst, extract frequency and transient features, and train a lightweight classifier to recognize where the tap occurred.

### Why It Works on Any MacBook Air
Many acoustic sensing research projects depend on stereo microphone arrays, specialized ultrasound transducers, or dual-channel audio interfaces. 

Our implementation is designed to run on standard single-channel microphone hardware, making it compatible with any standard MacBook Air.
- It requires only a single microphone channel (`channels: 1`) sampled at standard 44.1 kHz.
- It does not depend on stereo phase differences or ultrasound emission.
- It runs locally on the CPU with inference latency under 5 milliseconds, consuming negligible system resources.
- It adapts automatically to the host operating system, using `Cmd` shortcuts on macOS and `Ctrl` shortcuts on Windows or Linux.

---

## 3. Project Objectives

1. Accurately distinguish between left palm rest taps, right palm rest taps, and unrelated background noise.
2. Maintain low real-time latency so triggering a macro feels instantaneous.
3. Prevent false activations during normal computer use through active typing suppression.
4. Keep the architecture modular so individual signal processing steps can be evaluated independently.

---

## 4. Key Libraries Used

The implementation relies on standard Python scientific computing and signal processing libraries:

- **`sounddevice`**: Manages real-time audio input streams with low buffer latency using PortAudio bindings.
- **`numpy`**: Handles ring buffers, array slicing, fast vector operations, and signal normalization.
- **`librosa`**: Extracts audio features including Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, rolloff, flatness, and zero-crossing rates.
- **`scikit-learn`**: Provides classification algorithms (SVM, Random Forest, Extra Trees, Logistic Regression), stratified cross-validation, and performance metrics.
- **`joblib`**: Serializes trained models and configuration pipelines for fast runtime loading.
- **`pynput`**: Monitors system keyboard events to suppress tap detection while typing, and synthesizes macro keystrokes when a valid tap is detected.
- **`pyyaml`**: Reads system thresholds, sample rates, and shortcut mappings from a centralized configuration file.

---

## 5. How It Works

The system operates as a four-stage pipeline:

```
[Chassis Vibration] 
       │
       ▼
[1. Impulse Detector]      --> Ring buffer tracks background noise floor;
       │                        triggers on sharp acoustic transients (~23 ms window).
       ▼
[2. Feature Extraction]    --> Computes 20 MFCCs across attack and decay phases,
       │                        spectral centroid, rolloff, flatness, and crest factor.
       ▼
[3. Model Classification]  --> Classifier evaluates feature vector and outputs class
       │                        probabilities for left_tap, right_tap, and noise.
       ▼
[4. Listener & Macro]      --> Verifies confidence threshold; checks typing suppression;
                                dispatches OS shortcut (Cmd+C / Cmd+V).
```

### Why the Physics Makes It Work
When a finger strikes the laptop palm rest, it generates flexural waves in the aluminum top case. Because the strike is an impulse, it excites the natural resonant frequencies of the chassis. A left tap excites resonance modes closer to the left speaker grille and battery edge, while a right tap excites resonance modes on the opposing side. 

By analyzing the onset transient (the first 5 milliseconds of impact) separately from the decaying reverberation tail, the feature extractor captures the physical attack profile of the strike.

---

## 6. Baseline Performance and Evaluation

We evaluated the baseline machine learning model across 45 test events (15 left taps, 15 right taps, and 15 noise events) recorded on a MacBook Air palm rest.

### Confusion Matrix

| Actual \ Predicted | Predicted Left Tap | Predicted Right Tap | Predicted Noise | Total Actual |
| :--- | :---: | :---: | :---: | :---: |
| **Actual Left Tap** | **13** | 2 | 0 | 15 |
| **Actual Right Tap** | 0 | **15** | 0 | 15 |
| **Actual Noise** | 1 | 2 | **12** | 15 |
| **Total Predicted** | 14 | 19 | 12 | 45 |

### Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Left Tap** | 0.929 | 0.867 | 0.897 | 15 |
| **Right Tap** | 0.789 | 1.000 | 0.882 | 15 |
| **Noise** | 1.000 | 0.800 | 0.889 | 15 |
| | | | | |
| **Overall Accuracy** | | | **0.889** | 45 |
| **Macro Average** | 0.906 | 0.889 | 0.889 | 45 |
| **Weighted Average** | 0.906 | 0.889 | 0.889 | 45 |

### Analysis of the Results
- Right taps achieved 100% recall (all 15 were identified as right taps).
- Left taps suffered minor cross-talk, with 2 out of 15 being misclassified as right taps due to shared chassis reverberation.
- Noise rejection achieved 100% precision (when the model said noise, it was never a real tap), but recall was 80% because 3 noise events produced transient spikes that triggered false positive tap predictions (1 left, 2 right).

---

## 7. Why This Baseline Is Not Optimal: The Noise Interference Problem

While 88.9% accuracy is acceptable in a quiet environment, real-world laptop usage involves constant acoustic and mechanical interference:
1. **Key clatter while typing**: Striking keys sends high-amplitude mechanical shocks through the chassis that closely resemble palm taps.
2. **Desk vibrations**: Moving a mouse, placing a cup, or bumping the desk introduces low-frequency impulses.
3. **Ambient speech and background sounds**: Sudden airborne sounds can trigger the RMS threshold.

### Environmental Interference Benchmarks

To quantify this limitation, we measured performance across three operational environments:

| Test Condition | Environment Description | Overall Accuracy | False Positive Rate (Noise marked as Tap) | Primary Failure Mode |
| :--- | :--- | :---: | :---: | :--- |
| **Quiet Room (Baseline)** | Normal desk setting, no background conversation | **88.9%** | 20.0% (3/15) | Ambiguity between left and right palm rest modes |
| **Active Typing** | Continuous typing on the laptop keyboard during detection | **71.1%** | 46.7% (7/15) | Mechanical key strikes falsely trigger the impulse detector |
| **Loud Ambient (Cafe / Speech)** | Background speech and music (~65 dB SPL) | **57.8%** | 60.0% (9/15) | Airborne noise distorts spectral centroid and MFCC values |

### Why Pure Machine Learning Falls Short Here
We trained and evaluated multiple standard classifiers on these time-frequency features:
- **Logistic Regression**: Underfit the data because acoustic reverberation paths on a metal plate exhibit non-linear frequency interactions.
- **Random Forest & Extra Trees**: Performed well on clean training data but memorized ambient background levels, causing accuracy drops when room acoustics shifted.
- **Support Vector Machines (RBF Kernel)**: Achieved the strongest baseline accuracy, but still struggled when noise impulses overlapped with tap frequencies.

The core limitation is that generic MFCCs treat the laptop microphone like an open-air audio recorder. They do not model the physics of how mechanical waves disperse across solid aluminum.

---

## 8. Next Step: Fourier Transform and Spatial Triangulation

To solve the noise interference problem without adding sensors, we looked into structural acoustic wave propagation. 

When you strike a thin metal plate, high-frequency wave components travel faster than low-frequency components. This phenomenon, known as Lamb-wave dispersion, produces a measurable arrival time difference between frequency bands over short distances.

In the next phase of the project (developed on the `v2-hybrid-fourier` branch):
- We use the **Fast Fourier Transform (FFT)** to divide the captured transient into narrow spatial resonance bands.
- We measure the arrival delay between high-frequency (3 kHz to 14 kHz) and low-frequency (150 Hz to 900 Hz) wave envelopes.
- We combine a physics-based spatial template matcher with the machine learning classifier in a hybrid fusion model.

By checking whether a detected impulse displays true structural wave dispersion, the system can reject airborne speech and keyboard clatter before it ever reaches the classifier.

---

## 9. How to Use

### Setup
Install dependencies using `uv`:
```bash
uv sync
```

### 1. Data Collection
Collect training samples using either randomized group mode or single-class mode.

Group interleaved collection (recommended, cycles randomly in bursts of 3 to 4 taps to eliminate session bias):
```bash
uv run python -m src.holo.collect --label all --samples 50
```

Single-class collection:
```bash
uv run python -m src.holo.collect --label left_tap --samples 50
uv run python -m src.holo.collect --label right_tap --samples 50
uv run python -m src.holo.collect --label noise --samples 50
```

### 2. Model Training and Benchmarking
Train and benchmark all candidate models:
```bash
uv run python -m src.holo.train --features enhanced
```
The script saves the best-performing model to `models/acoustictap_model.joblib` and updates `models/leaderboard.csv`.

### 3. Model Evaluation
Evaluate the saved model on the test dataset:
```bash
uv run python -m src.holo.evaluate --holdout-only
```

### 4. Live Listener Daemon
Test live predictions safely in dry-run mode (logs predictions without firing keyboard shortcuts):
```bash
uv run python -m src.holo.listen --dry-run
```

Run the live listener daemon:
```bash
uv run python -m src.holo.listen --confidence 0.70
```
- Left palm rest tap triggers **Copy** (`Cmd+C` on macOS, `Ctrl+C` on Windows/Linux).
- Right palm rest tap triggers **Paste** (`Cmd+V` on macOS, `Ctrl+V` on Windows/Linux).

---

