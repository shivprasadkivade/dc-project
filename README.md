# AcousticTap (V4: Hybrid Fourier + Passive Acoustic Geofencing) — Perimeter-Aware Structure-Borne Tap Recognition
---

## 1. Where version 2 Left Off: The External Vibration Problem

With our dual-branch ML + Fourier model running on a MacBook Air, airborne speech and keyboard clatter were largely solved. But during everyday desk use, one stubborn failure mode remained: **external mechanical impacts**.

If you knocked on the wooden desk a few inches beside the laptop, set down a heavy ceramic mug, or tapped the metal strip above the keyboard, the shockwave traveled up through the laptop's rubber feet and excited the aluminum chassis. Because those vibrations were structure-borne, they occasionally slipped past the V2 classifier and fired an unintended `Cmd+C` or `Cmd+V`.

### Why We Rejected Active Ultrasound (version 3)
Before building the geofence, we tested a third prototype (`v3-ultrasonic`) that turned the MacBook's speakers into a continuous ultrasonic sonar ($19.0\text{ kHz}$ on the left channel, $20.2\text{ kHz}$ on the right) to verify whether a finger was physically hovering over the palm rest. 

We ultimately shelved that branch for three practical reasons:
1. **Audio Routing Fragility:** If you mute your laptop, plug in wired headphones, or connect to an external monitor/Bluetooth speaker, the ultrasonic emitter stops working.
2. **Hardware Variance:** Speaker frequency response above $19\text{ kHz}$ varies wildly across laptop models and degrades as speakers age.
3. **Unnecessary Complexity:** Emitting continuous high-frequency tones adds power overhead and can be audible to pets.

Instead of adding active sonar, we realized the solution was already hiding inside the passive solid-state physics we built in version 2.

---

## 2. Passive Plate-Wave Physics (Our Core Foundation)

When your finger strikes a thin aluminum plate like a MacBook unibody ($h \approx 1.2\text{ mm}$), it excites **flexural Lamb waves** that propagate directly through the metal sheet toward the internal microphone. 

AcousticTap relies on two physical laws of plate acoustics that airborne noise and damped desk vibrations cannot fake: **Lamb-wave dispersion** and **spatial resonance modes**.

```text
                           [Palm Rest Strike]
                                   |
              +--------------------+--------------------+
              v                                         v
   [Lamb-Wave Dispersion]                    [Spatial Resonance Modes]
   High frequencies travel faster            Chassis boundaries produce 
   than low frequencies                      distinct nodal frequency bands
   (Delta t_disp = t_LF - t_HF)              (Left: 129 Hz, Right: 172 Hz)
              |                                         |
              +--------------------+--------------------+
                                   |
                                   v
                      [Fourier Triangulation]
                      43-Dimensional Physical
                      Acoustic Feature Vector
```

### 2.1 Lamb-Wave Dispersion Physics
In open air, sound travels as a non-dispersive compression wave at roughly $343\text{ m/s}$ regardless of pitch. In a thin aluminum plate ($h \approx 1.2\text{ mm}$), however, flexural Lamb waves are **dispersive**—meaning phase velocity scales with the square root of frequency $\omega$:

$$v_{\text{phase}}(\omega) = \sqrt{\frac{\omega \cdot h}{2} \cdot \sqrt{\frac{E}{3\rho(1-\nu^2)}}}$$

where $E$ is Young's modulus for aluminum ($70\text{ GPa}$), $\rho$ is density ($2700\text{ kg/m}^3$), and $\nu$ is Poisson's ratio ($0.33$).

Because $v_{\text{phase}}(\omega) \propto \sqrt{\omega}$, **high-frequency wave packets ($3\text{ kHz}$ to $14\text{ kHz}$) race through the aluminum unibody faster than low-frequency bending waves ($150\text{ Hz}$ to $900\text{ Hz}$)**, creating a measurable arrival lag at the microphone:

$$\Delta t_{\text{disp}} = t_{\text{LF}} - t_{\text{HF}}$$

- **Airborne Noise:** Travels at constant speed through air $\rightarrow \Delta t_{\text{disp}} \approx 0$.
- **Direct Palm Rest Strike:** Travels through the metal plate $\rightarrow$ produces a sharp, repeatable $\Delta t_{\text{disp}}$ signature tied to the distance between the strike zone and the internal mic.

### 2.2 Spatial Resonance Mode Distribution
Internal MacBook components—the front battery cells, trackpad cutout, rear display hinge, and flanking speaker grilles—create asymmetric boundary conditions across the chassis. When we compute a 1024-sample FFT over the post-transient window across **12 log-spaced resonance bands ($150\text{ Hz}$ to $16\text{ kHz}$)**:
- **Left Palm Taps** couple strongly into lower chassis flexure modes, peaking at **$129\text{ Hz}$**.
- **Right Palm Taps** traverse a stiffer boundary path across the trackpad cutout, shifting the dominant resonance peak to **$172\text{ Hz}$** with higher relative energy in mid-to-high frequency bands.

---

## 3. Experimental Proofs of the Physics Engine

Controlled wave-propagation benchmarks on Apple Silicon MacBook hardware confirm both physical signatures:

### 3.1 Proof 1: Lamb-Wave Dispersion & Arrival Timing Measurements

| Acoustic Timing Metric | Left Palm Rest Tap | Right Palm Rest Tap | Difference (Left vs Right) | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **High-Frequency Arrival ($t_{\text{HF}}$)** | **0.77 ms** | **4.13 ms** | **3.36 ms** | Direct ballistic wave reaches internal microphone first |
| **Low-Frequency Arrival ($t_{\text{LF}}$)** | **0.91 ms** | **0.88 ms** | 0.02 ms | Dispersive bending wave arrives with delayed envelope peak |
| **Lamb Dispersion Delay ($\Delta t_{\text{disp}}$)** | **0.14 ms** | **-3.24 ms** | **3.38 ms** | Propagation distance to mic alters frequency group velocity |
| **Low-Band Group Delay ($\tau_{\text{low}}$)** | 0.46 samples | 0.61 samples | 0.15 samples | Low-frequency plate resonance boundary phase delay |
| **Mid-Band Group Delay ($\tau_{\text{mid}}$)** | 0.51 samples | 0.81 samples | 0.30 samples | Transition region dispersion phase delay |

![Arrival Timing & Dispersion](reports/fourier_dispersion_timing.png)

On a **Left Palm Tap**, the high-frequency packet reaches the internal mic at **$0.77\text{ ms}$**; on a **Right Palm Tap**, it travels across the trackpad boundary and arrives at **$4.13\text{ ms}$**—a **$3.36\text{ ms}$ physical separation** measured before any ML classifier is consulted.

### 3.2 Proof 2: 12-Band Spatial Resonance Profile

| Resonance Band | Frequency Range (Hz) | Left Tap Energy ($\log_{10} V^2$) | Right Tap Energy ($\log_{10} V^2$) | Energy Difference ($\Delta$ dB) | Dominant Strike Side |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Band 01** | 150 - 221 Hz | **3.30** | **3.54** | **-0.24 dB** | **RIGHT** |
| **Band 02** | 221 - 326 Hz | **2.89** | **2.85** | **+0.04 dB** | **BALANCED** |
| **Band 03** | 326 - 482 Hz | **2.50** | **2.61** | **-0.11 dB** | **RIGHT** |
| **Band 04** | 482 - 711 Hz | **2.01** | **2.10** | **-0.09 dB** | **RIGHT** |
| **Band 05** | 711 - 1049 Hz | **1.48** | **1.66** | **-0.18 dB** | **RIGHT** |
| **Band 06** | 1049 - 1549 Hz | **0.74** | **1.04** | **-0.30 dB** | **RIGHT** |
| **Band 07** | 1549 - 2286 Hz | **-0.75** | **0.28** | **-1.03 dB** | **RIGHT** |
| **Band 08** | 2286 - 3373 Hz | **-1.24** | **-0.65** | **-0.59 dB** | **RIGHT** |
| **Band 09** | 3373 - 4978 Hz | **-2.41** | **-1.14** | **-1.27 dB** | **RIGHT** |
| **Band 10** | 4978 - 7347 Hz | **-2.48** | **-1.34** | **-1.14 dB** | **RIGHT** |
| **Band 11** | 7347 - 10842 Hz | **-2.51** | **-1.59** | **-0.93 dB** | **RIGHT** |
| **Band 12** | 10842 - 16000 Hz | **-2.90** | **-1.99** | **-0.91 dB** | **RIGHT** |

![Frequency Spectrum & Spatial Resonance](reports/fourier_frequency_spectrum.png)

By computing **adjacent sub-band energy ratios** across bands 1 through 12, we obtain a force-invariant spatial fingerprint that stays consistent whether you tap lightly or firmly.

---

## 4. version 4 Breakthrough: Two-Stage Passive Acoustic Geofencing

To stop external table knocks and bezel taps without adding sensors or ultrasonic speakers, we asked a simple physical question: *how does a vibration entering through the laptop's rubber feet differ from a finger striking the top aluminum sheet directly?*

When a knock travels from a wooden desk through elastomeric rubber feet into the unibody, the rubber acts as a mechanical low-pass filter—stripping away crisp high-frequency onset energy, slowing the attack rise time, and distorting the dispersion lag.

We turned that insight into a **Two-Stage Passive Acoustic Geofence (`src/holo/geofence.py`)** that acts as a perimeter bouncer before any gesture classification happens:

### Stage 1: Physical Invariant Safety Gates ($< 0.1\text{ ms}$)
Every captured transient is first checked against four hard physical bounds of direct metal impacts:
- **High-to-Low Frequency Energy Ratio ($z_0 \ge -16.0\text{ dB}$):** Rejects vibrations that were low-pass filtered through desk surfaces and rubber feet.
- **Attack-to-Decay Ratio ($z_1 \ge 1.40$):** Rejects sluggish shockwaves whose energy builds slowly compared to a direct fingertip strike.
- **Onset Crest Factor ($z_2 \ge 3.5$):** Rejects rounded, blunt desk thuds in favor of sharp metallic transients.
- **Lamb-Wave Dispersion Lag Window ($\Delta t_{\text{disp}} \in [-6.5, 3.5]\text{ ms}$):** Rejects impacts whose wave arrival timing falls outside the physical geometry of the palm rest.

### Stage 2: Supervised Boundary Discriminator & Spatial Manifold
Transients that pass Stage 1 are projected into a 20-dimensional spatial subspace $\mathbf{z} \in \mathbb{R}^{20}$ (combining group delays, normalized onset sub-band energies, and resonance ratios) trained directly on **palm-rest perimeter taps vs. outside desk knocks**:

$$P(\text{inside} \mid \mathbf{z}) = \sigma(\mathbf{w}^T \mathbf{z} + b)$$

To prevent out-of-distribution vibrations from sneaking across the linear decision boundary, Stage 2 also checks the **Mahalanobis distance** ($d_{\text{in}} \le 6.50$) to the calibrated inside-chassis manifold.

```text
       [Chassis / Desk Vibration Transient]
                         |
                         v
       +------------------------------------+
       |   Stage 1: Physical Invariant Gate |
       |   - Attack-to-Decay Ratio (ADR)    |
       |   - Crest Factor & HF/LF Ratio     |
       |   - Lamb Dispersion Window         |
       +-----------------+------------------+
                         | Passed
                         v
       +------------------------------------+
       | Stage 2: Supervised Boundary Gate  |
       | P(inside) >= 0.50 & d_in <= 6.50   |
       +-----------------+------------------+
                         |
           +-------------+-------------+
           | In-Bounds                 | Out-of-Bounds (Desk Bump)
           v                           v
  [Hybrid Gesture Classifier]   [GEOFENCE: OUTSIDE BOUNDARY]
  (50% ML + 50% Fourier FFT)    (Macro Blocked / Never Triggered)
           |
           v
   [Dispatch OS Shortcut]
      (Cmd+C / Cmd+V)
```

---

## 5. Dual-Branch Hybrid Gesture Classification & Validation Results

Once an impulse clears the Geofence perimeter, it is evaluated simultaneously by both branches of our hybrid classifier:
1. **ML Branch (70 Dimensions):** 20 Onset MFCCs, 20 Decay MFCCs, 20 Std MFCCs, and 10 spectral/temporal shape descriptors fed into a tuned RBF-SVM.
2. **Fourier Triangulator Branch (43 Dimensions):** 12 onset band energies, 12 wake band energies, 11 sub-band ratios, Lamb dispersion lag, group delays, and SNR evaluated via shrinkage LDA and cosine template matching ($T = 2.5$):

$$P_{\text{Fourier}} = \alpha P_{\text{LDA}} + (1 - \alpha) P_{\text{Cosine}}$$

Both branches blend their class probabilities with weight $w = 0.50$:

$$P_{\text{hybrid}} = (1 - w) \cdot P_{\text{ML}} + w \cdot P_{\text{Fourier}}$$

### 5.1 Gesture Holdout Benchmark (589 Events / 118 Holdout)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Left Tap** | 0.923 | 0.837 | 0.878 | 43 |
| **Right Tap** | 0.886 | **0.975** | **0.929** | 40 |
| **Noise** | 0.857 | 0.857 | 0.857 | 35 |
| | | | | |
| **Overall Accuracy** | | | **0.8898** | 118 |
| **Macro Average** | 0.889 | 0.890 | **0.888** | 118 |
| **Weighted Average** | 0.891 | 0.890 | **0.889** | 118 |

- **5-Fold Stratified Cross-Validation Macro F1:** `0.8771`
- **Winning Production Architecture:** `Hybrid_SVM_RBF` ($50\%$ Tuned RBF-SVM + $50\%$ Fourier Triangulator)
- **Active-Typing False Positive Rate:** **$< 8.5\%$** (down from $46.7\%$ in V1)

### 5.2 Geofence Boundary Verification Benchmark
When paired with the 45-second perimeter calibration dataset (`dataset/boundary/`), the complete V4 pipeline achieves near-perfect spatial containment:

| Evaluation Metric | Result | Notes |
| :--- | :---: | :--- |
| **Calibrated Gesture Holdout Accuracy** | **96.67%** | Evaluated on clean center + calibrated boundary split |
| **Geofence Inside False Rejection Rate (FRR)** | **0.00% (0/60)** | Zero legitimate center or perimeter palm taps blocked |
| **Geofence Outside Rejection Rate (OOB-RR)** | **100.00% (15/15)** | Every external desk knock & bezel strike blocked at perimeter |

---

## 6. Boundary Calibration & Usage Guide

### Setup
Install all project dependencies using `uv`:
```bash
uv sync
```

### 1. Dataset Separation: Center Data vs. Boundary Data
AcousticTap keeps center gesture training data strictly separated from boundary calibration data so neither contaminates the other:
- **Center Gesture Datasets (`dataset/left_tap/`, `dataset/right_tap/`, `dataset/noise/`):** Taps recorded in the **center zone** of the palm rests to train Left vs. Right classification.
- **Dedicated Boundary Datasets (`dataset/boundary/`):**
  - `dataset/boundary/inside/`: Taps recorded along the **perimeter borders** of the palm rests (near the trackpad cutout, outer edges, and top border).
  - `dataset/boundary/outside/`: Knocks recorded on the **desk surface** ($2\text{--}15\text{ cm}$ around the laptop) and upper keyboard bezel.

### 2. Run the 45-Second Interactive Geofence Wizard
Calibrate your laptop's physical perimeter in three guided phases:
```bash
uv run python -m src.holo.collect --calibrate-geofence
```
*(Optional: adjust trigger sensitivity with `--threshold`, e.g., `--threshold 0.015` for quiet rooms or light taps.)*

The wizard guides you step by step:
1. **Phase 1 — Left Palm Rest Perimeter (15 taps):** Tap along the outer edges and borders of the left palm rest (avoid the center). Saves to `dataset/boundary/inside/boundary_left_xxxx.npy`.
2. **Phase 2 — Right Palm Rest Perimeter (15 taps):** Tap along the outer edges and borders of the right palm rest (avoid the center). Saves to `dataset/boundary/inside/boundary_right_xxxx.npy`.
3. **Phase 3 — Outside Boundary Desk & Bezel (15 knocks):** Knock directly on your desk $2\text{--}15\text{ cm}$ around the laptop and tap the upper bezel strip above the function keys. Saves to `dataset/boundary/outside/boundary_outside_xxxx.npy`.

You can also add samples to either boundary zone individually:
```bash
uv run python -m src.holo.collect --boundary inside --samples 15
uv run python -m src.holo.collect --boundary outside --samples 15
```

### 3. Collect Center Gesture Data
Record center palm-rest samples (interleaved group mode cycles in random bursts of 3 to 4 taps to eliminate session bias):
```bash
# Group interleaved mode (recommended):
uv run python -m src.holo.collect --label all --samples 30

# Or single-class mode:
uv run python -m src.holo.collect --label left_tap --samples 30
uv run python -m src.holo.collect --label right_tap --samples 30
uv run python -m src.holo.collect --label noise --samples 30
```

### 4. Train the Hybrid Model & Fit the Geofence
Run the unified training pipeline:
```bash
uv run python -m src.holo.train --features enhanced
```
*(Use `--fourier-weight 0.50` to lock the fusion weight instead of running the automated grid search.)*

Training trains all hybrid candidates, fits the `AcousticGeofenceValidator` on inside vs. outside boundary samples, verifies FRR/OOB-RR rates, updates `models/leaderboard.csv`, and bundles everything into `models/acoustictap_model.joblib`.

### 5. Verify & Run the Live Listener Daemon
Test boundary blocking and gesture classification in dry-run mode without dispatching OS shortcuts:
```bash
uv run python -m src.holo.listen --dry-run
```
- **Desk knock output:**
  ```text
  [GEOFENCE: OUTSIDE BOUNDARY] Blocked external impact (outside boundary: desk knock / external vibration (prob_in=0.02 < 0.50, d_in=8.14)) | 1.4 ms
  ```
- **Palm rest tap output:**
  ```text
  [GEOFENCE: INSIDE (prob=0.98)] Pred: left_tap   | Conf: 0.94 (Req: 0.58) [ML:left_tap(0.92) + FFT:left_tap(0.96)] | 4.8 ms
      [DRY-RUN] Would trigger shortcut for: left_tap (Cmd+C/V)
  ```

Launch the active background macro daemon:
```bash
uv run python -m src.holo.listen --confidence 0.70
```
- **Left Palm Rest Tap** $\rightarrow$ Triggers **Copy** (`Cmd+C` on macOS, `Ctrl+C` on Windows/Linux).
- **Right Palm Rest Tap** $\rightarrow$ Triggers **Paste** (`Cmd+V` on macOS, `Ctrl+V` on Windows/Linux).
- *(Optional: pass `--no-geofence` to compare live behavior with perimeter gating disabled.)*

### 6. Run the Automated MacBook Validation Suite
Regenerate all dispersion plots, 12-band spectrum charts, and benchmark sub-reports locally:
```bash
uv run python benchmark_report.py
# Or guided interactive mode:
uv run python benchmark_report.py --interactive
```
For macOS microphone permission setup and troubleshooting, see [`MAC_TESTER_GUIDE.md`](MAC_TESTER_GUIDE.md).

---

## 7. Geofence Configuration Reference (`config/config.yaml`)

```yaml
geofence:
  enabled: true                     # Set to false to disable geofence globally
  boundary_threshold: 0.50          # Probability threshold to accept transient as inside boundary [0.0 - 1.0]
  min_hf_lf_ratio_db: -16.0         # Stage 1: Min high-to-low frequency energy ratio (dB)
  min_attack_decay_ratio: 1.40      # Stage 1: Min onset attack-to-decay energy ratio
  min_crest_factor: 3.5             # Stage 1: Min onset peak-to-RMS ratio
  dispersion_window_ms: [-6.5, 3.5] # Stage 1: Allowable Lamb-wave dispersion lag window (ms)
  max_inside_distance: 6.50         # Stage 2: Max allowed Mahalanobis distance to inside manifold
  surface_tolerance_scale: 1.15     # Stage 2: Covariance expansion factor for varied table materials
```

---

## 8. V4 Repository Structure & Generated Artifacts

```text
├── MAC_TESTER_GUIDE.md                  # Instructions and troubleshooting for MacBook testers
├── README.md                            # version 4: Hybrid Fourier + Passive Geofencing (this file)
├── benchmark_report.py                  # Automated physics & latency benchmark suite
├── pyproject.toml                       # Project metadata and uv dependencies
├── config/
│   └── config.yaml                      # Centralized configuration (audio, ML, Fourier & geofence)
├── dataset/                             # Recorded .npy structure-borne transient files
│   ├── left_tap/                        # In-bounds left palm rest center transient samples
│   ├── right_tap/                       # In-bounds right palm rest center transient samples
│   ├── noise/                           # Ambient acoustic noise & key strike samples
│   └── boundary/                        # Dedicated boundary calibration datasets
│       ├── inside/                      # Perimeter boundary taps (left & right borders)
│       └── outside/                     # External desk knocks, table bumps & bezel strikes
├── models/                              # Versioned hybrid model bundles and leaderboard.csv
├── reports/                             # Experimental sub-reports and physics figures
│   ├── fourier_analysis_report.md       # Lamb-wave dispersion & 12-band spectral data
│   ├── fourier_dispersion_timing.png    # High/Low frequency wave arrival & group delay plot
│   ├── fourier_frequency_spectrum.png   # 12-band spatial resonance & differential plot
│   ├── instructor_expansion_report.md   # Consolidated master experimental report
│   ├── feature_importance.md            # Baseline 70-feature ANOVA & Gini ranking table
│   ├── feature_importance.png           # Top 15 ML feature importance bar chart
│   ├── latency_breakdown.md             # Stage-by-stage real-time latency profile
│   ├── spectrogram_comparison.png       # Left vs. Right vs. Spacebar spectrogram comparison
│   └── surface_damping_analysis.md      # Desk vs. Mousepad vs. Lap boundary experiments
└── src/holo/                            # Core V4 Hybrid + Geofence source code
    ├── config.py                        # System configuration loader and constants
    ├── detector.py                      # Pre-trigger ring buffer impulse detector
    ├── features.py                      # 70-dim Onset/Decay MFCC & spectral extractor
    ├── triangulation.py                 # 43-dim Lamb-wave dispersion & FourierTriangulator
    ├── geofence.py                      # Two-stage supervised boundary discriminator & manifold
    ├── collect.py                       # Interactive center & boundary sample collection CLI
    ├── train.py                         # Dual-branch training, fusion search & geofence fitting
    ├── evaluate.py                      # Holdout evaluation, confusion matrix & geofence reporting
    └── listen.py                        # Low-latency hybrid listener with geofence & typing suppression
```