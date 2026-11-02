from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time
from typing import Any
import numpy as np
import sounddevice as sd

try:
    from .config import BLOCK_SIZE, CLASSES, DATA_DIR, RMS_THRESHOLD, SAMPLE_RATE, ULTRASONIC_ENABLED
    from .detector import ImpulseDetector, get_input_channels
    from .ultrasonic import UltrasonicSonarGate
except (ImportError, ValueError):
    from config import BLOCK_SIZE, CLASSES, DATA_DIR, RMS_THRESHOLD, SAMPLE_RATE, ULTRASONIC_ENABLED  # type: ignore
    from detector import ImpulseDetector, get_input_channels  # type: ignore
    from ultrasonic import UltrasonicSonarGate  # type: ignore


def save_sample_atomic(save_dir: Path, label: str, waveform: np.ndarray) -> Path:
    """
    Writes a captured audio array to a .npy file atomically via a temporary file.
    Guarantees that interrupted recordings never leave 0-byte or corrupt .npy files.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(save_dir.glob(f"{label}_*.npy")))
    target_file = save_dir / f"{label}_{existing:04d}.npy"
    temp_file = save_dir / f".tmp_{label}_{existing:04d}_{os.getpid()}_{time.time_ns()}.npy"

    with open(temp_file, "wb") as f:
        np.save(f, waveform.astype(np.float32))
        f.flush()
        os.fsync(f.fileno())

    temp_file.replace(target_file)
    return target_file


def collect_single_class(label: str, num_samples: int, threshold: float) -> None:
    """Collects training taps for a single specified class."""
    save_dir = DATA_DIR / label
    channels = get_input_channels()
    detector = ImpulseDetector(threshold=threshold, sr=SAMPLE_RATE)
    sonar = UltrasonicSonarGate(sr=SAMPLE_RATE)

    if ULTRASONIC_ENABLED:
        sonar.start_emitter()
        print("[Ultrasonic] Sonar emitter active (19.0 kHz L / 20.2 kHz R).")

    print(f"\n[COLLECT - SINGLE CLASS] Recording {num_samples} samples for '{label}' (Channels: {channels}, Rate: {SAMPLE_RATE} Hz)...")
    print("Tap firmly on the target zone whenever ready. Press Ctrl+C to stop.\n")

    count = 0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=channels,
            callback=detector.audio_callback,
        ):
            while count < num_samples:
                try:
                    waveform = detector.event_queue.get(timeout=0.2)
                except Exception:
                    continue

                saved_file = save_sample_atomic(save_dir, label, waveform)
                count += 1
                peak = float(np.max(np.abs(waveform)))
                print(f"  Captured [{count}/{num_samples}] -> {saved_file.name} (Peak: {peak:.3f})")
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Saved {count} samples for '{label}'.")
    finally:
        sonar.stop_emitter()

    print(f"\nSaved {count} samples inside {save_dir}")


def collect_group_interleaved(samples_per_class: int, threshold: float) -> None:
    """Collects training taps using randomized group interleaving to eliminate session bias."""
    channels = get_input_channels()
    detector = ImpulseDetector(threshold=threshold, sr=SAMPLE_RATE)
    sonar = UltrasonicSonarGate(sr=SAMPLE_RATE)

    if ULTRASONIC_ENABLED:
        sonar.start_emitter()
        print("[Ultrasonic] Sonar emitter active (19.0 kHz L / 20.2 kHz R).")

    remaining = {c: samples_per_class for c in CLASSES}
    total_needed = samples_per_class * len(CLASSES)
    collected_total = 0

    print(f"\n{'='*65}")
    print(f" [COLLECT - RANDOMIZED GROUP MODE] (Channels: {channels}, Rate: {SAMPLE_RATE} Hz)")
    print(f" Target: {samples_per_class} samples per class ({total_needed} total across {len(CLASSES)} classes)")
    print(" Classes will cycle randomly in groups of 3 or 4 taps.")
    print(" Press Ctrl+C at any time to exit safely.")
    print(f"{'='*65}\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=channels,
            callback=detector.audio_callback,
        ):
            while any(v > 0 for v in remaining.values()):
                active_classes = [c for c, rem in remaining.items() if rem > 0]
                current_class = random.choice(active_classes)
                group_size = min(random.choice([3, 4]), remaining[current_class])

                instructions = {
                    "left_tap": "Tap on the LEFT palm rest",
                    "right_tap": "Tap on the RIGHT palm rest",
                    "noise": "Produce ambient noise / key clicks / desk vibrations",
                }
                desc = instructions.get(current_class, f"Perform {current_class}")

                print(f"\n{'*'*65}")
                print(f">>> TARGET: [ {current_class.upper()} ] <<<")
                print(f"Action: {desc}")
                print(f"Group quota: {group_size} tap(s) | {remaining[current_class]} remaining for this class")
                print(f"{'*'*65}")

                for i in range(group_size):
                    while True:
                        try:
                            waveform = detector.event_queue.get(timeout=0.2)
                            break
                        except Exception:
                            continue

                    save_dir = DATA_DIR / current_class
                    saved_file = save_sample_atomic(save_dir, current_class, waveform)
                    remaining[current_class] -= 1
                    collected_total += 1
                    peak = float(np.max(np.abs(waveform)))
                    print(
                        f"  [{i+1}/{group_size}] -> {saved_file.name} (Peak: {peak:.3f}) | Progress: {collected_total}/{total_needed}"
                    )

    except KeyboardInterrupt:
        print("\n\n[Collection Stopped by User]")
        print("Summary of samples recorded this session:")
        for c in CLASSES:
            done = samples_per_class - remaining[c]
            print(f"  - {c:10s}: {done}/{samples_per_class} captured")
    finally:
        sonar.stop_emitter()

    print(f"\n{'='*65}")
    print(f"Successfully collected {collected_total}/{total_needed} samples!")
    print(f"{'='*65}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect AcousticTap training samples with atomic file saving and active ultrasonic emission"
    )
    parser.add_argument(
        "--label",
        "--class",
        dest="label",
        choices=CLASSES + ["all"],
        default="all",
        help="Target class label (default: 'all' runs randomized group interleaving)",
    )
    parser.add_argument(
        "--samples",
        "--count",
        dest="samples",
        type=int,
        default=25,
        help="Number of taps to record per class (default: 25)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RMS_THRESHOLD,
        help="Mic RMS trigger threshold",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.label == "all":
        collect_group_interleaved(args.samples, args.threshold)
    else:
        collect_single_class(args.label, args.samples, args.threshold)


if __name__ == "__main__":
    main()
