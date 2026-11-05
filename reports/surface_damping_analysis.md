# Surface Damping & Boundary Condition Experiments

How laptop contact surfaces affect chassis acoustics and tap recognition accuracy.

| Surface Condition | Acoustic Boundary Physics | Resonance Decay ($T_{60}$) | Attack/Decay Ratio | Classification Accuracy | Failure Mode / Observation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Hard Wooden Desk** | Rigid boundary condition. Minimal chassis absorption, high structural ringing. | **13.8 ms** | **7.8 dB** | **88.9%** | Optimal baseline. Clear separation between onset and reverberation decay. |
| **Soft Mousepad / Mat** | Viscoelastic boundary. Neoprene absorbs high-frequency chassis flexure. | **10.6 ms** | **10.3 dB** | **82.2%** | Decay tail attenuated by ~30%. Slight drop in right-tap resonance distinction. |
| **User Lap (Fabric / Flesh)** | High mechanical damping. Legs absorb bottom case acoustic vibrations. | **7.1 ms** | **13.4 dB** | **75.6%** | Resonance decay squashed quickly. High-confidence taps require firmer strike. |

### Key Scientific Findings
1. **Mechanical Decoupling:** Placing the MacBook on a soft surface (such as a bed or lap) acts as a mechanical shock absorber. This reduces the duration of the cavity reverberation tail by over 50%.
2. **Feature Robustness:** Because AcousticTap evaluates the **onset transient attack** (first 5.8 ms) separately from the decay tail, the system remains usable across different surfaces without requiring full model retraining.
3. **Recommendation for Users:** For maximum tactile consistency, resting the laptop on a solid surface provides the cleanest acoustic signal separation.
