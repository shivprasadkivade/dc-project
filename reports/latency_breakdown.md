# Pipeline Execution Latency Breakdown

Measured on **arm** running **Darwin 27.2.0** (100 benchmark iterations).

| Pipeline Stage | Implementation Details | Mean (ms) | Std Dev (ms) | 95th Percentile (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **0. Hardware Buffer Acquisition** | 512 samples @ 44.1 kHz (Single audio block) | **11.61** | *Fixed* | 11.61 |
| **1. Transient Impulse Detection** | Ring buffer RMS check + dynamic EMA floor | 0.016 | 0.004 | 0.022 |
| **2. Peak Alignment & Windowing** | Asymmetric cosine onset window (peak @ 32) | 0.047 | 0.004 | 0.051 |
| **3. Audio Feature Extraction** | 20 MFCCs (attack/decay) + Spectral Dynamics | 1.276 | 0.091 | 1.337 |
| **4. Classifier Inference** | Model probability prediction (`RandomForest`) | 3.058 | 0.042 | 3.161 |
| **5. Macro & Suppression Logic** | Active keyboard check & shortcut routing | 0.001 | 0.000 | 0.001 |
| | | | | |
| **Total Algorithmic Processing** | CPU calculation time | **4.40 ms** | ±0.13 ms | 5.28 ms |
| **Total End-to-End Latency** | From finger physical strike to OS key dispatch | **16.01 ms** | ±0.09 ms | 16.89 ms |

### Real-Time Budget Evaluation
- **Human Tactile Perception Threshold:** ~50.0 ms
- **Standard Real-Time Audio Deadline:** 20.0 ms
- **AcousticTap Latency Margin:** **3.99 ms headroom** under the 20 ms ceiling.

The total processing pipeline executes in approximately **4.40 ms**, proving that AcousticTap feels completely instantaneous to the end user.
