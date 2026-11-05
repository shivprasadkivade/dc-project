# MacBook Tester Guide: Running AcousticTap Benchmarks

Thank you for helping validate **AcousticTap** on your MacBook! Because AcousticTap uses the laptop's aluminum chassis and internal microphone to turn the palm rest into programmable buttons (Copy & Paste), real MacBook hardware testing is essential for our academic evaluation report.

This guide provides step-by-step instructions to run the automated validation suite in under 3 minutes.

---

## 1. Quick Setup (1 Minute)

### Step 1: Open Terminal
Open **Terminal** on your Mac (`Cmd + Space`, type `Terminal`, press Enter).

### Step 2: Ensure `uv` is Installed
We use `uv` (the fast Python package manager). If you don't already have it, install it with one command:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*(Or if you use Homebrew: `brew install uv`)*

Restart your Terminal or run `source $HOME/.cargo/env` if prompted.

### Step 3: Switch to the Test Branch
Navigate to the project folder and switch to this branch:
```bash
git checkout test-mac-expansion
uv sync
```
`uv sync` will automatically set up Python and install all dependencies (`sounddevice`, `librosa`, `scikit-learn`, `matplotlib`, etc.) in seconds.

---

## 2. Important: macOS Microphone Permission

macOS requires you to grant microphone permission to your terminal app.

1. When you first run an audio command, macOS might pop up a prompt:
   > **"Terminal" would like to access the microphone.**
   > Click **OK / Allow**.

2. If you do not see a prompt or audio is silent:
   - Open **System Settings** -> **Privacy & Security** -> **Microphone**.
   - Make sure **Terminal** (or **iTerm** / **VS Code**) is toggled **ON**.
   - Also verify in **System Settings** -> **Sound** -> **Input** that your **MacBook Internal Microphone** is selected.

---

## 3. Running the Validation Suite

Choose either **Option A** (live tap recording) or **Option B** (fast automated run):

### Option A: Interactive Guided Mode (Recommended for MacBook)
This mode guides you step-by-step with countdowns on screen:
```bash
uv run python benchmark_report.py --interactive
```

**What you will be prompted to do:**
1. **Left Palm Rest Tap:** The script counts down `3... 2... 1... TAP NOW!`. Tap your finger firmly on the aluminum palm rest to the left of your trackpad.
2. **Right Palm Rest Tap:** The script counts down again. Tap the right palm rest.
3. **Keyboard Spacebar Strike:** Strike the Spacebar normally.
4. The script then automatically runs latency micro-benchmarks, surface damping evaluations, feature importance rankings, and Fourier Transform dispersion analysis.

### Option B: Fast Automated Mode (Instant Run)
If you just want to run the suite immediately without tapping:
```bash
uv run python benchmark_report.py
```
This mode uses existing samples and physical chassis simulations to generate all 5 benchmarks and plots in ~5 seconds.

### Option C: Try the Live Gesture Listener
If you want to experience the palm rest tap recognition in real time:
```bash
uv run python -m src.holo.listen --dry-run
```
- Tap on your left or right palm rest.
- Watch real-time predictions and latency logs appear in your terminal.
- Press `Ctrl + C` to stop.

---

## 4. What Gets Generated

The script automatically generates all charts, tables, and reports inside the **`reports/`** directory:

| Generated File | Content Description |
| :--- | :--- |
| **`reports/spectrogram_comparison.png`** | High-resolution side-by-side spectrograms of Left Tap vs Right Tap vs Spacebar strike. |
| **`reports/latency_breakdown.md`** | Microsecond timing table showing buffer capture, feature extraction, and inference latency (< 20 ms). |
| **`reports/surface_damping_analysis.md`** | Comparison table of acoustic decay and accuracy on Wooden Desk vs Mousepad vs Lap. |
| **`reports/feature_importance.png`** | Color-coded bar chart of the Top 15 most discriminative acoustic features. |
| **`reports/feature_importance.md`** | Statistical ANOVA F-score and Gini importance ranking table. |
| **`reports/fourier_dispersion_timing.png`** | Time difference between HF and LF arrivals (Lamb-wave dispersion delay $\Delta t$). |
| **`reports/fourier_frequency_spectrum.png`** | FFT magnitude overlay, 12-band spatial resonance bar chart, and differential spatial spectrum ($\Delta$ dB). |
| **`reports/fourier_analysis_report.md`** | Comprehensive physical acoustic write-up with equations, timing tables, and resonance shifts. |
| **`reports/instructor_expansion_report.md`** | Consolidated master report combining all five experiments for your instructor. |

---

## 5. Returning the Results

Once the script finishes:
1. Check that the files exist inside `reports/`.
2. Commit and push the `reports/` folder, or send the `reports/` folder back to the team.

Thank you for your help testing on macOS!
