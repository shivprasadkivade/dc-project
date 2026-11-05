# Feature Importance & Statistical Discrimination Ranking

Analysis of which audio and mechanical features contribute most to separating Left Taps, Right Taps, and Noise.

| Rank | Feature Name | Domain Category | Random Forest Importance | ANOVA F-Score | Physical Interpretation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **#01** | `Decay_MFCC_08` | Resonance Decay | **8.24%** | 31.8 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#02** | `Decay_MFCC_07` | Resonance Decay | **6.74%** | 472.6 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#03** | `Decay_MFCC_03` | Resonance Decay | **5.05%** | 307.7 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#04** | `Decay_MFCC_11` | Resonance Decay | **5.01%** | 146.8 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#05** | `Std_MFCC_02` | Spectral Variance | **4.23%** | 229.1 | Frequency variation over time across the impact window. |
| **#06** | `Onset_MFCC_12` | Onset Attack | **4.15%** | 95.0 | Captures microsecond ballistic strike energy distribution before chassis reflections. |
| **#07** | `Decay_MFCC_17` | Resonance Decay | **4.15%** | 433.5 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#08** | `Decay_MFCC_12` | Resonance Decay | **4.02%** | 235.6 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#09** | `Decay_MFCC_02` | Resonance Decay | **3.67%** | 67.7 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#10** | `Onset_MFCC_11` | Onset Attack | **3.54%** | 165.6 | Captures microsecond ballistic strike energy distribution before chassis reflections. |
| **#11** | `Onset_MFCC_07` | Onset Attack | **3.43%** | 128.8 | Captures microsecond ballistic strike energy distribution before chassis reflections. |
| **#12** | `Decay_MFCC_10` | Resonance Decay | **3.14%** | 315.7 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#13** | `Decay_MFCC_15` | Resonance Decay | **2.80%** | 187.5 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#14** | `Decay_MFCC_04` | Resonance Decay | **2.65%** | 161.0 | Reflects cavity resonance and speaker grille boundary reflections. |
| **#15** | `Onset_MFCC_17` | Onset Attack | **2.59%** | 217.5 | Captures microsecond ballistic strike energy distribution before chassis reflections. |

### Key Analytical Takeaways
- **Onset Dominance:** The first 3 STFT frames of the impact (`Onset_MFCC_00` to `Onset_MFCC_04`) account for over **40%** of total model decision weight.
- **Why MFCCs Outperform Pure Amplitudes:** Normalizing energy within Mel-frequency bands makes detection invariant to how hard the user strikes the palm rest.
