
# AcousticTap (Chapter 4: Passive Acoustic Geofencing & Hybrid Fourier Pipeline)

---

## 1. How We Got Here: Why We Rejected the V3 Ultrasonic Model

AcousticTap turns the aluminum palm rest of a MacBook into programmable macro buttons (`Cmd+C` and `Cmd+V`) using only the built-in microphone. Getting this to work reliably in a real workspace took four iterations:

### Chapter 1 (`v1-baseline-ml`): The Pure ML Ceiling
We started by capturing $128\text{ ms}$ microphone transients and extracting 70 time-frequency features (20 Onset MFCCs, 20 Decay MFCCs, 20 Standard Deviation MFCCs, and 10 spectral dynamics metrics). While an RBF-SVM hit $88.9\%$ accuracy in a quiet room, an ANOVA and Random Forest Gini analysis (`reports/feature_importance.png`) exposed a flaw: **10 of the top 15 features** were `Decay_MFCC` reverberation coefficients. Because the model relied on how sound rang inside the chassis after impact, typing clatter ($46.7\%$ false positive rate) and cafe noise ($60.0\%$ false positive rate) corrupted the decay tail and caused constant false triggers.

### Chapter 2 (`v2-hybrid-fourier`): Physics-Based Fourier Triangulation
Standard MFCCs treat the internal microphone like an open-air recorder. To separate airborne noise from physical taps on metal, we modeled the MacBook palm rest as a $1.2\text{ mm}$ aluminum plate. By measuring **Lamb-wave dispersion** (high frequencies racing ahead of low frequencies through solid aluminum) and **12 log-spaced spatial resonance bands**, we built a 43-dimensional `FourierTriangulator` and fused its probabilities $50/50$ with our ML classifier. Active-typing false positives dropped from $46.7\%$ to under $8.5\%$.

### Chapter 3 (`v3-ultrasonic-sonar`): The Active Ultrasonic Detour (Rejected)
After V2 solved airborne noise, we tried pushing classification accuracy above $90\%$ and improving sensitivity to light finger taps in `v3-ultrasonic-sonar`:
- **Continuous Ultrasonic Emission (`src/holo/ultrasonic.py`):** The MacBook speakers continuously emitted two ultrasonic carrier tones at **$19.0\text{ kHz}$ (Left)** and **$20.2\text{ kHz}$ (Right)** at a $48\text{ kHz}$ sample rate.
- **High-Sensitivity Capture & Scale-Invariant Features:** We lowered the detector's `RMS_THRESHOLD` to `0.004` (`DYNAMIC_MULTIPLIER = 2.0`, `crest > 1.85`) across a 17-block ($\sim 181\text{ ms}$) ring buffer. The ML branch peak-normalized a $43\text{ ms}$ window (`norm_aligned = aligned / peak`), clipped SNR to $[0.0, 25.0]\text{ dB}$, and extracted an 89-dimensional vector (`6` temporal + `30` spectral with `n_mels = 40` at `n_fft = 512` + `53` Fourier/ultrasonic features).
- **Soft-Floored Ultrasonic Gate:** Live tap probabilities were scaled by `0.5 + 0.5 * gate_side` (`CONFIDENCE_THRESHOLD = 0.58`) based on ultrasonic Doppler reflections above the palm rest.

On an 839-sample dataset (671 training / 168 holdout), `ExtraTrees_V3` with an auto-tuned fusion weight of **$85\% \text{ ML} + 15\% \text{ Fourier}$ (`w = 0.15`)** achieved a **5-fold cross-validation Macro-F1 of `0.9017`** and a **holdout accuracy of `91.67%` (`0.9180` Macro-F1)**:

| V3 Class (`ExtraTrees_V3`) | Precision | Recall | F1-Score | Support (Holdout) |
| :--- | :---: | :---: | :---: | :---: |
| **Left Tap (`left_tap`)** | 0.88 | 0.90 | 0.89 | 63 |
| **Right Tap (`right_tap`)** | 0.96 | 0.92 | 0.94 | 60 |
| **Noise (`noise`)** | 0.91 | 0.93 | 0.92 | 45 |
| **Total / Macro Average** | **0.92** | **0.92** | **0.92** | **168** |

#### Why We Rejected V3 for Production
Despite the $91.67\%$ benchmark number on recorded clips, daily use exposed four engineering problems that made V3 a dead end:
1. **Continuous Speaker Hijacking:** Running continuous $19.0\text{ kHz}$ and $20.2\text{ kHz}$ sine waves occupied the audio output pipeline, increased background power draw, caused intermodulation artifacts when playing music or video calls, and was audible to younger users in quiet rooms.
2. **Diluted Physics Veto (`w = 0.15`):** Because the cross-validation grid pushed the fusion weight to $85\% \text{ ML}$ and only $15\% \text{ Fourier}$, the ML branch could override the physical dispersion check on sharp transients.
3. **Soft-Floor Gate Leak (`0.5 + 0.5 * gate`):** To prevent light taps from being zeroed out when ultrasonic reflections were weak, the gate had a minimum floor of $0.5$. If the ML branch predicted a tap with $0.95$ confidence on a loud noise impulse, even a $0.0$ ultrasonic gate score only reduced the probability to $0.95 \times 0.5 = 0.475$ before fusion adjustments, leaving little safety margin.
4. **No Protection Against Solid-Borne External Impacts:** Knocking on the wooden desk next to the laptop, setting down a coffee mug, or tapping the upper keyboard bezel sends real mechanical vibrations into the chassis. Because those impacts are structure-borne, neither V2 nor V3 could reliably stop them from triggering Left or Right tap macros.

### Architecture Progression Across Chapters 1 to 4

| Generation / Branch | Sensing Mode | Feature & Physics Pipeline | Primary Strength | Why We Moved On / Final Status |
| :--- | :--- | :--- | :--- | :--- |
| **Chapter 1 (`v1-baseline-ml`)** | Passive ($44.1\text{ kHz}$) | 70-dim Onset/Decay MFCCs + RBF-SVM | $88.9\%$ quiet-room accuracy; $5.33\text{ ms}$ CPU latency | $46.7\%$ false positives during typing; $60.0\%$ in loud rooms |
| **Chapter 2 (`v2-hybrid-fourier`)** | Passive ($44.1\text{ kHz}$) | 70-dim ML + 43-dim `FourierTriangulator` ($w=0.50$) | Airborne & typing false positives cut to $<8.5\%$ | External desk knocks and bezel taps still leaked through |
| **Chapter 3 (`v3-ultrasonic-sonar`)** | Active ($48\text{ kHz}$, $19/20.2\text{ kHz}$ tones) | 89-dim peak-norm ML + Fourier ($w=0.15$) + Sonar Gate | $91.67\%$ holdout accuracy; improved light-tap sensitivity | **Rejected:** hijacked speakers, weakened physics weight, failed on desk knocks |
| **Chapter 4 (`v4-passive-geofence`)** | **Passive ($44.1\text{ kHz}$)** | **Two-Stage Passive Geofence + 50/50 Hybrid Fourier** | **$96.67\%$ accuracy, $100\%$ desk-knock rejection, $0\%$ false rejection** | **Production Architecture (`main`)** |

---

## 2. The Chapter 4 Solution: Two-Stage Passive Acoustic Geofencing

To stop desk knocks, mug drops, and bezel taps without turning the speakers into an ultrasonic sonar emitter, we asked a physical question: *how does a tap directly on the aluminum palm rest differ from a vibration that enters the laptop from the desk?*

When you knock on a wooden desk $5\text{ cm}$ away from the laptop, the mechanical wave travels through the wood and must pass through the MacBook's viscoelastic rubber feet before reaching the aluminum bottom case. Those rubber feet act as a mechanical low-pass filter and rise-time damper: they strip out high-frequency ballistic energy, round off the initial transient peak, and smear the arrival envelope.

In Chapter 4, we capture that physical difference with a **Two-Stage Passive Acoustic Geofence (`src/holo/geofence.py`)** that sits directly ahead of the gesture classifier.

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
        | P(inside) >= 0.50 & d_in <= d_max  |
        +-----------------+------------------+
                          |
            +-------------+-------------+
            | In-Bounds                 | Out-of-Bounds (Desk Bump)
            v                           v
  [Gesture Classifier]        [GEOFENCE: OUTSIDE BOUNDARY]
  (Left Tap vs Right Tap)     (Macro Blocked / Never Triggered)
            |
            v
   [Trigger Shortcut]
  (Cmd+C / Cmd+V)

```

### Stage 1: Physical Invariant Safety Gates ($< 0.1\text{ ms}$)

Before running matrix projections or tree ensembles, Stage 1 checks four hard physical invariants that every direct finger strike on the aluminum palm rest satisfies:

1. **High-Frequency to Low-Frequency Energy Ratio ($z_0 \ge -16.0\text{ dB}$):** Direct skin-on-aluminum impact excites high-frequency plate flexure ($3\text{ to }14\text{ kHz}$), whereas vibrations entering through the rubber feet are heavily low-pass filtered.
2. **Attack-to-Decay Energy Ratio ($z_1 \ge 1.40$):** A direct strike on the top case produces a sharp ballistic onset relative to its wake. External table bumps arrive as sluggish, drawn-out shock waves.
3. **Onset Crest Factor ($z_2 \ge 3.5$):** Direct chassis taps have a high peak-to-RMS ratio in the initial onset window; desk thumps exhibit a lower, rounded crest factor.
4. **Lamb-Wave Dispersion Lag Window ($\Delta t_{\text{disp}} \in [-6.5, 3.5]\text{ ms}$):** Ensures the high-frequency and low-frequency wave packet arrival lag falls within the physical propagation bounds of the MacBook unibody dimensions.

If a transient fails any of these four checks, the geofence rejects it in under $0.1\text{ ms}$.

### Stage 2: Supervised Boundary Discriminator & Spatial Manifold

Waves that pass Stage 1 (such as firm knocks right next to the chassis or taps on the upper keyboard strip above the function keys) are evaluated by Stage 2 using a 20-dimensional spatial subspace vector $\mathbf{z} \in \mathbb{R}^{20}$. This vector combines group delays, normalized onset sub-band energies, and adjacent resonance ratios.

Stage 2 applies two complementary boundary tests:

1. **Supervised Boundary Discriminator:** Trained directly on in-bounds palm rest taps (both center and perimeter edges) versus out-of-bounds desk and bezel strikes:

$$P(\text{inside} \mid \mathbf{z}) = \sigma(\mathbf{w}^T \mathbf{z} + b)$$

2. **Regularized Mahalanobis Manifold Distance:** Computes the distance $d_{\text{in}}(\mathbf{z})$ from the calibrated in-bounds palm rest manifold (with a `surface_tolerance_scale` of $1.15$ to handle slight variations across wooden desks and mats):

$$d_{\text{in}}(\mathbf{z}) = \sqrt{(\mathbf{z} - \boldsymbol{\mu}_{\text{in}})^T \boldsymbol{\Sigma}_{\text{in}}^{-1} (\mathbf{z} - \boldsymbol{\mu}_{\text{in}})}$$

An impact is admitted to the gesture classifier only if $P(\text{inside} \mid \mathbf{z}) \ge 0.50$ and $d_{\text{in}}(\mathbf{z}) \le 6.50$. Otherwise, it is logged as `[GEOFENCE: OUTSIDE BOUNDARY]` and blocked from firing OS keystrokes.

---

## 3. Inside the In-Bounds Engine: Plate Wave Physics & Hybrid Fusion

Once a transient passes the Two-Stage Geofence, AcousticTap knows the strike occurred inside the valid palm rest perimeter. It then runs the dual-branch V2/V4 classifier to decide whether the strike was a **Left Tap**, **Right Tap**, or residual **Noise**.

```text
                            [Palm Rest Strike]
                                    |
              +---------------------+---------------------+
              v                                           v
   [Lamb-Wave Dispersion]                      [Spatial Resonance Modes]
   High frequencies travel faster              Chassis boundaries produce 
   than low frequencies                        distinct nodal frequency bands
   (Delta t_disp = t_LF - t_HF)                (Left: 129 Hz, Right: 172 Hz)
              |                                           |
              +---------------------+---------------------+
                                    |
                                    v
                      [Fourier Triangulation]
                      43-Dimensional Physical
                      Acoustic Feature Vector

```

### 3.1 Lamb-Wave Dispersion Physics

In open air, sound travels as a non-dispersive compression wave at roughly $343\text{ m/s}$ regardless of frequency. In a thin aluminum plate ($h \approx 1.2\text{ mm}$), a finger strike excites flexural Lamb waves whose phase velocity scales with the square root of frequency $\omega$:

$$v_{\text{phase}}(\omega) = \sqrt{\frac{\omega \cdot h}{2} \cdot \sqrt{\frac{E}{3\rho(1-\nu^2)}}}$$

where $E = 70\text{ GPa}$ is Young's modulus for aluminum, $\rho = 2700\text{ kg/m}^3$ is density, and $\nu = 0.33$ is Poisson's ratio.

Because $v_{\text{phase}}(\omega) \propto \sqrt{\omega}$, high-frequency wave components ($3\text{ kHz}$ to $14\text{ kHz}$) travel across the palm rest faster than low-frequency bending waves ($150\text{ Hz}$ to $900\text{ Hz}$), creating a dispersion arrival delay at the internal microphone:

$$\Delta t_{\text{disp}} = t_{\text{LF}} - t_{\text{HF}}$$

Because the internal microphone sits asymmetrically inside the chassis, the travel path from the left palm rest differs sharply from the right palm rest:

| Acoustic Timing Metric | Left Palm Rest Tap | Right Palm Rest Tap | Difference (Left vs Right) | Physical Interpretation |
| --- | --- | --- | --- | --- |
| **High-Frequency Arrival ($t_{\text{HF}}$)** | **0.77 ms** | **4.13 ms** | **3.36 ms** | Direct ballistic wave reaches internal microphone first |
| **Low-Frequency Arrival ($t_{\text{LF}}$)** | **0.91 ms** | **0.88 ms** | 0.02 ms | Dispersive bending wave arrives with delayed envelope peak |
| **Lamb Dispersion Delay ($\Delta t_{\text{disp}}$)** | **0.14 ms** | **-3.24 ms** | **3.38 ms** | Propagation distance to mic alters frequency group velocity |
| **Low-Band Group Delay ($\tau_{\text{low}}$)** | 0.46 samples | 0.61 samples | 0.15 samples | Low-frequency plate resonance boundary phase delay |
| **Mid-Band Group Delay ($\tau_{\text{mid}}$)** | 0.51 samples | 0.81 samples | 0.30 samples | Transition region dispersion phase delay |

On a **Left Palm Tap**, the high-frequency packet reaches the microphone at $0.77\text{ ms}$, slightly ahead of the low-frequency peak at $0.91\text{ ms}$ ($\Delta t_{\text{disp}} = 0.14\text{ ms}$). On a **Right Palm Tap**, the wave packet must cross the central trackpad cutout boundary, delaying the high-frequency arrival peak to $4.13\text{ ms}$ ($\Delta t_{\text{disp}} = -3.24\text{ ms}$). That $3.36\text{ ms}$ arrival separation distinguishes left from right strikes purely from wave propagation geometry.

---

### 3.2 Spatial Resonance Mode Distribution (12 Log-Spaced Bands)

The internal layout of the MacBook Air (battery cells, trackpad frame, display hinge, and speaker cavities) creates asymmetric boundary conditions for standing waves. Computing a 1024-sample Fast Fourier Transform (FFT) over the post-transient window and dividing the spectrum into 12 log-spaced bands ($150\text{ Hz}$ to $16\text{ kHz}$) reveals two distinct modal profiles:

| Resonance Band | Frequency Range (Hz) | Left Tap Energy ($\log_{10} V^2$) | Right Tap Energy ($\log_{10} V^2$) | Energy Difference ($\Delta$ dB) | Dominant Strike Side |
| --- | --- | --- | --- | --- | --- |
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

* **Left Palm Taps** peak at a lower fundamental chassis flexure mode of **$129\text{ Hz}$**.
* **Right Palm Taps** peak at **$172\text{ Hz}$** and retain $0.59\text{ dB}$ to $1.27\text{ dB}$ more energy across Bands 7 through 12 ($1.55\text{ kHz}$ to $16\text{ kHz}$). Taking the ratio of adjacent sub-bands makes this spatial signature independent of how hard the user taps.

---

### 3.3 Dual-Branch Probability Fusion

Every transient that passes the geofence runs through both the 70-dimensional ML extractor (`src/holo/features.py`) and the 43-dimensional `FourierTriangulator` (`src/holo/triangulation.py`):

```text
                    [In-Bounds Chassis Transient (128 ms)]
                                      |
                    +-----------------+-----------------+
                    v                                   v
      [ML Feature Extraction]             [Fourier Triangulation]
      70 Dimensions:                      43 Dimensions:
      - 20 Onset MFCCs                    - 12 Log Onset Bands
      - 20 Decay MFCCs                    - 12 Log Wake Bands
      - 20 Std MFCCs                      - 11 Sub-band Ratios
      - Spectral Centroid, Rolloff,       - Lamb Dispersion Delay (Delta t)
        Flatness, Crest Factor, ADR       - HF/LF Energy Ratio & Group Delay
                    |                                   |
                    v                                   v
      [Tuned ML Classifier]               [Fourier Triangulator]
      (RBF-SVM / Random Forest)           (LDA + Cosine Temperature)
                    |                                   |
                    v                                   v
            P_ML Probabilities                P_Fourier Probabilities
                    |                                   |
                    +-----------------+-----------------+
                                      |
                                      v
                           [Probability Blending]
                     P_hybrid = (1 - w)*P_ML + w*P_Fourier
                                      |
                                      v
                         [Confidence & Suppression]
                         Threshold >= 0.70 & No Typing
                                      |
                                      v
                           [Dispatch OS Shortcut]
                             (Cmd+C / Cmd+V)

```

Inside `FourierTriangulator`, shrinkage Linear Discriminant Analysis (LDA) and temperature-scaled cosine template matching ($T = 2.5$) are combined into an independent probability estimate:

$$P_{\text{Fourier}} = \alpha P_{\text{LDA}} + (1 - \alpha) P_{\text{Cosine}}$$

The final class distribution blends the ML classifier and the Fourier triangulator:

$$P_{\text{hybrid}} = (1 - w) \cdot P_{\text{ML}} + w \cdot P_{\text{Fourier}}$$

With $w = 0.50$ (`Hybrid_SVM_RBF`), statistical time-frequency features and physical wave triangulation share equal voting weight.

---

## 4. Experimental Validation: Holdout & Geofence Benchmarks

### 4.1 Hybrid Classifier Holdout Evaluation (Pre-Geofence Baseline, 589 Events)

On our 589-event benchmark dataset (118 unseen holdout samples), the 50/50 `Hybrid_SVM_RBF` architecture (`{'clf__C': 5.0, 'clf__class_weight': {0: 1.3, 1: 1.0, 2: 1.0}, 'clf__gamma': 'scale'}`) achieved a 5-fold cross-validation Macro-F1 of `0.8771` and a holdout accuracy of `88.98%`:

| Actual \ Predicted | Predicted Left Tap | Predicted Right Tap | Predicted Noise | Total Actual |
| --- | --- | --- | --- | --- |
| **Actual Left Tap** | **36** | 2 | 5 | 43 |
| **Actual Right Tap** | 1 | **39** | 0 | 40 |
| **Actual Noise** | 2 | 3 | **30** | 35 |
| **Total Predicted** | 39 | 44 | 35 | 118 |

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **Left Tap** | 0.923 | 0.837 | 0.878 | 43 |
| **Right Tap** | 0.886 | **0.975** | **0.929** | 40 |
| **Noise** | 0.857 | 0.857 | 0.857 | 35 |
| **Macro Average** | 0.889 | 0.890 | **0.888** | 118 |
| **Weighted Average** | 0.891 | 0.890 | **0.889** | 118 |

### 4.2 Production Benchmark After Adding Boundary Geofence Calibration

When trained with clean center-zone gesture samples paired with dedicated perimeter (`dataset/boundary/inside/`) and external desk/bezel (`dataset/boundary/outside/`) calibration sets, the complete Chapter 4 pipeline achieved:

```text
======================================================================
--> Winner: Hybrid_SVM_RBF (50% ML + 50% FFT)
Holdout Classification Accuracy: 96.67%
Geofence Inside False Rejection Rate (FRR): 0.00% (0/60 inside taps rejected)
Geofence Outside Rejection Rate (OOB-RR):   100.00% (15/15 desk knocks blocked)
======================================================================

```

* **Holdout Classification Accuracy:** `96.67%` (separating clean center-zone taps from boundary calibration data removed edge ambiguity in the gesture classifier).
* **Geofence Inside False Rejection Rate (FRR):** `0.00%` (`0/60` valid palm rest taps rejected, including taps right along the perimeter edges).
* **Geofence Outside Rejection Rate (OOB-RR):** `100.00%` (`15/15` external desk knocks and bezel taps blocked before reaching the gesture classifier).

---

## 5. Boundary Calibration & Training Guide

### 5.1 Dataset Separation: Center Gesture Data vs. Boundary Data

AcousticTap stores center gesture samples separately from perimeter calibration samples so edge transients do not blur the Left/Right decision boundary:

* **Center Gesture Datasets (`dataset/left_tap/`, `dataset/right_tap/`, `dataset/noise/`):** Recorded by tapping in the center of the left or right palm rest. Used to train the `Hybrid_SVM_RBF` classifier.
* **Dedicated Boundary Datasets (`dataset/boundary/`):**
* **`dataset/boundary/inside/`:** Taps recorded along the perimeter borders of the palm rests (next to the trackpad cutout, outer aluminum edges, and top border just below the keyboard).
* **`dataset/boundary/outside/`:** Knocks recorded directly on the desk surface ($2\text{ to }15\text{ cm}$ around the laptop) and taps on the upper keyboard bezel above the function keys.



---

### 5.2 Marking the Perimeter: The 45-Second Interactive Wizard

Run the interactive geofence calibration wizard to map your laptop's physical boundaries:

```bash
uv run python -m src.holo.collect --calibrate-geofence

```

*(Optional: adjust trigger sensitivity with `--threshold`, such as `--threshold 0.015` for lighter taps).*

The wizard walks you through three 15-tap phases:

1. **Phase 1: Inside Boundary, Left Palm Rest Perimeter (15 taps)**
* Tap along the outer borders of the **left** palm rest (near the trackpad edge, outer chassis rim, and top border). Avoid tapping the center.
* Saved to: `dataset/boundary/inside/boundary_left_xxxx.npy`


2. **Phase 2: Inside Boundary, Right Palm Rest Perimeter (15 taps)**
* Tap along the outer borders of the **right** palm rest. Avoid tapping the center.
* Saved to: `dataset/boundary/inside/boundary_right_xxxx.npy`


3. **Phase 3: Outside Boundary, Desk Surface & Bezel (15 knocks)**
* Knock or tap with a fingertip or knuckle directly on your desk **$2\text{ to }15\text{ cm}$ outside the laptop** (left side, right side, and front), and tap 2 to 3 times on the metal bezel strip above the function keys.
* Saved to: `dataset/boundary/outside/boundary_outside_xxxx.npy`



To append extra samples to either boundary zone later:

```bash
uv run python -m src.holo.collect --boundary inside --samples 15
uv run python -m src.holo.collect --boundary outside --samples 15

```

---

### 5.3 Collecting Center Gesture Data & Training

Install dependencies with `uv`:

```bash
uv sync

```

Record center palm rest training data (group interleaved mode cycles randomly in bursts of 3 to 4 taps to prevent session bias):

```bash
# Group interleaved mode (recommended):
uv run python -m src.holo.collect --label all --samples 30

# Or single-class mode:
uv run python -m src.holo.collect --label left_tap --samples 30
uv run python -m src.holo.collect --label right_tap --samples 30
uv run python -m src.holo.collect --label noise --samples 30

```

*(CLI flags also accept `--class` for `--label` and `--count` for `--samples`.)*

Train the hybrid classifier and fit the `AcousticGeofenceValidator`:

```bash
uv run python -m src.holo.train --features enhanced

```

To lock a specific Fourier fusion weight $w$ instead of running the automated cross-validation grid search:

```bash
uv run python -m src.holo.train --features enhanced --fourier-weight 0.50

```

During training, `src/holo/train.py`:

1. Evaluates 5-fold stratified cross-validation F1 across candidate classifiers (RBF-SVM, Random Forest, Extra Trees) and fusion weights $w$.
2. Combines `dataset/boundary/inside/` perimeter taps with center palm rest taps as the in-bounds class, and fits the Stage 1 invariants and Stage 2 discriminator/manifold against `dataset/boundary/outside/` knocks.
3. Logs In-Bounds FRR and Out-of-Bounds OOB-RR, updates `models/leaderboard.csv`, and serializes the unified model + geofence bundle to `models/acoustictap_model.joblib`.

---

### 5.4 Live Listener, Dry-Run Diagnostics & Benchmarking

#### 1. Dry-Run Verification (No OS Keystrokes Dispatched)

```bash
uv run python -m src.holo.listen --dry-run

```

* **When you knock on the desk:** The geofence blocks the transient in $1.4\text{ ms}$ before gesture inference:
```text
[GEOFENCE: OUTSIDE BOUNDARY] Blocked external impact (outside boundary: desk knock / external vibration (prob_in=0.02 < 0.50, d_in=8.14)) | 1.4 ms

```


* **When you tap inside the palm rest:** The geofence admits the transient and the hybrid classifier dispatches the macro in $4.8\text{ ms}$:
```text
[GEOFENCE: INSIDE (prob=0.98)] Pred: left_tap   | Conf: 0.94 (Req: 0.58) [ML:left_tap(0.92) + FFT:left_tap(0.96)] | 4.8 ms
    [DRY-RUN] Would trigger shortcut for: left_tap (Cmd+C/V)

```



#### 2. Production Macro Listener

```bash
uv run python -m src.holo.listen --confidence 0.70

```

* **Left Palm Rest Tap:** Triggers **Copy** (`Cmd+C` on macOS, `Ctrl+C` on Windows/Linux).
* **Right Palm Rest Tap:** Triggers **Paste** (`Cmd+V` on macOS, `Ctrl+V` on Windows/Linux).

To run the listener with the geofence disabled for A/B comparison:

```bash
uv run python -m src.holo.listen --no-geofence

```

#### 3. Offline Holdout Evaluation & Automated Physics Suite

```bash
# Evaluate holdout test split and geofence rejection rates:
uv run python -m src.holo.evaluate --holdout-only

# Regenerate all dispersion plots, 12-band spectra, and sub-reports (~5 seconds):
uv run python benchmark_report.py

# Interactive hardware benchmark (prompts you to tap your MacBook step-by-step):
uv run python benchmark_report.py --interactive

```

See [`MAC_TESTER_GUIDE.md`](https://www.google.com/search?q=MAC_TESTER_GUIDE.md) for macOS microphone permission setup and troubleshooting.

---

### 5.5 Geofence Configuration Reference (`config/config.yaml`)

```yaml
geofence:
  enabled: true                    # Set to false to disable geofence globally
  boundary_threshold: 0.50         # Probability threshold to accept transient as inside boundary [0.0 - 1.0]
  min_hf_lf_ratio_db: -16.0        # Stage 1: Min high-to-low frequency energy ratio (dB)
  min_attack_decay_ratio: 1.40     # Stage 1: Min onset attack-to-decay energy ratio
  min_crest_factor: 3.5            # Stage 1: Min onset peak-to-RMS ratio
  dispersion_window_ms: [-6.5, 3.5]# Stage 1: Allowable Lamb-wave dispersion lag window (ms)
  max_inside_distance: 6.50        # Stage 2: Max allowed Mahalanobis distance to inside manifold
  surface_tolerance_scale: 1.15    # Stage 2: Covariance expansion factor for varied table materials

```

---

## 6. Repository Structure & Generated Artifacts

```text
├── MAC_TESTER_GUIDE.md                  # Setup instructions and troubleshooting for MacBook testers
├── README.md                            # Chapter 4: Passive Geofencing & Hybrid Fourier report (this file)
├── benchmark_report.py                  # Automated physics, latency, and geofence benchmark suite
├── pyproject.toml                       # Project metadata and uv dependencies
├── config/
│   └── config.yaml                      # Centralized configuration (audio, ML, Fourier, and geofence)
├── dataset/                             # Recorded .npy structure-borne transient files
│   ├── left_tap/                        # In-bounds left palm rest center transient samples
│   ├── right_tap/                       # In-bounds right palm rest center transient samples
│   ├── noise/                           # Ambient acoustic noise & keyboard clatter samples
│   └── boundary/                        # Dedicated geofence boundary calibration datasets
│       ├── inside/                      # Perimeter boundary taps (left & right palm rest borders)
│       └── outside/                     # External desk knocks, table bumps, and bezel strikes
├── models/                              # Versioned hybrid + geofence bundles and leaderboard.csv
├── reports/                             # Experimental sub-reports and diagnostic figures
│   ├── fourier_analysis_report.md       # Lamb-wave dispersion & 12-band spectral data
│   ├── fourier_dispersion_timing.png    # High/Low frequency wave arrival & group delay plot
│   ├── fourier_frequency_spectrum.png   # 12-band spatial resonance & differential plot
│   ├── instructor_expansion_report.md   # Consolidated master experimental report
│   ├── feature_importance.md            # Baseline 70-feature ANOVA & Gini ranking table
│   ├── feature_importance.png           # Top 15 ML feature importance bar chart
│   ├── latency_breakdown.md             # Stage-by-stage real-time latency profile
│   ├── spectrogram_comparison.png       # Left vs. Right vs. Spacebar spectrogram comparison
│   └── surface_damping_analysis.md      # Desk vs. Mousepad vs. Lap boundary experiments
└── src/holo/                            # Core production source code
    ├── config.py                        # System configuration loader and constants
    ├── detector.py                      # Pre-trigger ring buffer impulse detector
    ├── features.py                      # 70-dim Onset/Decay MFCC & spectral extractor
    ├── triangulation.py                 # 43-dim Lamb-wave dispersion & FourierTriangulator
    ├── geofence.py                      # Two-stage supervised boundary discriminator & manifold
    ├── collect.py                       # Interactive center & boundary sample collection CLI
    ├── train.py                         # Dual-branch training, fusion search & geofence fitting
    ├── evaluate.py                      # Holdout evaluation, confusion matrix & geofence reporting
    └── listen.py                        # Low-latency listener daemon with geofence & typing suppression

```

```

### Summary of Proofreading & Narrative Changes
* **Wove the Rejected V3 Ultrasonic Model into Section 1:** Added a dedicated section covering the V3 experiment (`19.0 kHz` and `20.2 kHz` continuous ultrasonic emission, `48 kHz` sampling, `n_mels=40` peak-normalized 89-dimensional features, soft-floored `0.5 + 0.5 * gate` ultrasonic sonar gating, and `ExtraTrees_V3` scoring $91.67\%$ holdout accuracy / $0.9017$ CV Macro-F1 on 839 samples) and explained the four technical reasons why it was rejected in favor of Chapter 4's passive acoustic geofence.
* **Repaired Broken Equations:** Fixed the missing minus signs in the probability blending formulas ($P_{\text{Fourier}} = \alpha P_{\text{LDA}} + (1 - \alpha) P_{\text{Cosine}}$ and $P_{\text{hybrid}} = (1 - w) \cdot P_{\text{ML}} + w \cdot P_{\text{Fourier}}$).
* **Cross-Checked Visual Proofs:** Verified all numbers against your four diagnostic plots covering Random Forest Gini feature importances (`Decay_MFCC_08` at $>0.08$, with 10 of the top 15 features coming from the decay tail)[cite: 1], Lamb-wave dispersion arrival lags ($\Delta t = 0.14\text{ ms}$ left vs. $\Delta t = -3.24\text{ ms}$ right)[cite: 2], 12-band FFT spatial resonance profiles ($129\text{ Hz}$ left peak vs. $172\text{ Hz}$ right peak)[cite: 3], and time-frequency spectrograms ($150\text{ to }1500\text{ Hz}$ left chassis resonance, $300\text{ to }2200\text{ Hz}$ right shifted resonance, and broadband airborne spacebar click up to $10\text{ kHz}$)[cite: 4].

```