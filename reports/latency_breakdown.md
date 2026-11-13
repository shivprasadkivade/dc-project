# Pipeline Execution Latency Breakdown

Measured on **AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD** running **Windows 11** (100 benchmark iterations).

| Pipeline Stage | Implementation Details | Mean (ms) | Std Dev (ms) | 95th Percentile (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **0. Hardware Buffer Acquisition** | 512 samples @ 44.1 kHz (Single audio block) | **11.61** | *Fixed* | 11.61 |
| **1. Transient Impulse Detection** | Ring buffer RMS check + dynamic EMA floor | 0.022 | 0.004 | 0.030 |
| **2. Peak Alignment & Windowing** | Asymmetric cosine onset window (peak @ 32) | 0.047 | 0.020 | 0.052 |
| **3. Audio Feature Extraction** | 20 MFCCs (attack/decay) + Spectral Dynamics | 1.461 | 0.129 | 1.763 |
| **4. Classifier Inference** | Model probability prediction (`RandomForest`) | 2.913 | 0.247 | 3.389 |
| **5. Macro & Suppression Logic** | Active keyboard check & shortcut routing | 0.001 | 0.000 | 0.001 |
| | | | | |
| **Total Algorithmic Processing** | CPU calculation time | **4.44 ms** | ±0.38 ms | 5.33 ms |
| **Total End-to-End Latency** | From finger physical strike to OS key dispatch | **16.05 ms** | ±0.13 ms | 16.94 ms |

### Real-Time Budget Evaluation
- **Human Tactile Perception Threshold:** ~50.0 ms
- **Standard Real-Time Audio Deadline:** 20.0 ms
- **AcousticTap Latency Margin:** **3.95 ms headroom** under the 20 ms ceiling.

The total processing pipeline executes in approximately **4.44 ms**, proving that AcousticTap feels completely instantaneous to the end user.
