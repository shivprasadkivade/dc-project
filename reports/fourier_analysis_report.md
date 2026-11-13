# Fourier Transform & Physical Acoustic Dispersion Analysis

Detailed experimental evaluation of structural acoustic wave propagation across the MacBook unibody chassis.

---

## 1. Physical Principle: Lamb-Wave Dispersion
When a finger strikes the aluminum palm rest, the direct mechanical impact excites flexural **Lamb waves** in the metal unibody. Because the plate thickness ($h \approx 1.2$ mm) is comparable to acoustic wavelengths at high frequencies, flexural wave velocity depends on frequency:

$$v_{phase}(\omega) = \sqrt{\frac{\omega \cdot h}{2} \cdot \sqrt{\frac{E}{3\rho(1-\nu^2)}}}$$

where $E$ is Young's modulus of aluminum (70 GPa), $\rho$ is density ($2700 \text{ kg/m}^3$), and $\nu$ is Poisson's ratio (0.33).

As a direct consequence, **high-frequency wave components (3 kHz - 14 kHz) travel faster than low-frequency components (150 Hz - 900 Hz)**. This produces a measurable dispersion arrival delay $\Delta t_{disp} = t_{LF} - t_{HF}$ at the microphone.

---

## 2. Arrival Timing Differences Between Left & Right Taps

| Acoustic Timing Metric | Left Palm Rest Tap | Right Palm Rest Tap | Difference (Left vs Right) | Physical Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **High-Frequency Arrival ($t_{HF}$)** | **0.41 ms** | **0.43 ms** | 0.02 ms | Direct ballistic wave reaches internal microphone first |
| **Low-Frequency Arrival ($t_{LF}$)** | **0.86 ms** | **0.91 ms** | 0.05 ms | Dispersive bending wave arrives with delayed envelope peak |
| **Lamb Dispersion Delay ($\Delta t_{disp}$)** | **0.45 ms** | **0.48 ms** | **0.02 ms** | Propagation distance to mic alters frequency group velocity |
| **Low-Band Group Delay ($\tau_{low}$)** | 0.61 samples | 0.85 samples | 0.23 samples | Low-frequency plate resonance boundary phase delay |
| **Mid-Band Group Delay ($\tau_{mid}$)** | 0.49 samples | 0.47 samples | 0.01 samples | Transition region dispersion phase delay |

![Arrival Timing & Dispersion](fourier_dispersion_timing.png)

---

## 3. Difference of Frequencies & Spatial Resonance Distribution

Because the internal battery, speaker cavities, and trackpad cutouts create asymmetric mechanical boundaries, striking the left palm rest excites different unibody vibration modes than the right palm rest.

| Resonance Band | Frequency Range (Hz) | Left Tap Energy ($log_{10} V^2$) | Right Tap Energy ($log_{10} V^2$) | Energy Difference ($\Delta$ dB) | Dominant Strike Side |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Band 01** | 150 - 221 Hz | **3.56** | **2.89** | **+0.66 dB** | **LEFT** |
| **Band 02** | 221 - 326 Hz | **3.23** | **3.41** | **-0.17 dB** | **RIGHT** |
| **Band 03** | 326 - 482 Hz | **3.06** | **2.71** | **+0.35 dB** | **LEFT** |
| **Band 04** | 482 - 711 Hz | **2.47** | **3.06** | **-0.59 dB** | **RIGHT** |
| **Band 05** | 711 - 1049 Hz | **2.47** | **1.70** | **+0.77 dB** | **LEFT** |
| **Band 06** | 1049 - 1549 Hz | **1.76** | **2.21** | **-0.45 dB** | **RIGHT** |
| **Band 07** | 1549 - 2286 Hz | **0.83** | **1.45** | **-0.61 dB** | **RIGHT** |
| **Band 08** | 2286 - 3373 Hz | **-0.12** | **0.09** | **-0.20 dB** | **RIGHT** |
| **Band 09** | 3373 - 4978 Hz | **-0.62** | **-0.60** | **-0.03 dB** | **BALANCED** |
| **Band 10** | 4978 - 7347 Hz | **-0.88** | **-0.85** | **-0.03 dB** | **BALANCED** |
| **Band 11** | 7347 - 10842 Hz | **-1.16** | **-1.18** | **+0.02 dB** | **BALANCED** |
| **Band 12** | 10842 - 16000 Hz | **-1.42** | **-1.47** | **+0.05 dB** | **BALANCED** |

![Frequency Spectrum & Spatial Resonance](fourier_frequency_spectrum.png)

---

## 4. Why Fourier Triangulation Eliminates Ambient Noise Interference

1. **Airborne vs Structure-Borne Distinction:**
   Airborne sounds (such as room speech, music, or air conditioning) travel through air at ~343 m/s as pure compression waves without plate flexure. They **do not exhibit Lamb wave dispersion** ($\Delta t_{disp} \approx 0$).
2. **Key Clatter Separation:**
   Mechanical typing strikes excite high-frequency switch clicks (>5 kHz) without engaging the lower chassis plate resonance modes (150 - 900 Hz).
3. **Deterministic Spatial Template:**
   By matching measured 12-band spatial resonance distributions against calibrated left and right modal templates, Fourier Triangulation rejects false triggers before they reach the classifier.
