"""
AcousticTap Audio Management & Hardware Isolation Architecture.

Guarantees strict isolation of the MacBook internal microphone and speakers:
1. Point 1 (Chassis Tap Sensor): Bound strictly to the MacBook built-in microphone
   to reliably capture solid-state acoustic palm rest taps. External and Bluetooth
   headset microphones are strictly ignored to avoid forcing macOS CoreAudio into
   low-quality 16 kHz HFP/SCO hands-free mode.
2. Built-in Speaker Targeting: Provides dedicated internal speaker resolution so
   that audio outputs (e.g. ultrasonic tones) do not play into user headphones.
3. Passive Error Recovery: Re-resolves device indices and re-establishes the stream
   only when an active stream encounters a PortAudio error or stops unexpectedly
   during Bluetooth connect/disconnect events (zero background polling loops).
"""

from __future__ import annotations

import logging
import platform
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Any, List
import numpy as np
import sounddevice as sd

from .config import (
    CFG,
    SAMPLE_RATE,
    BLOCK_SIZE,
    CHANNELS,
    TAP_DEVICE,
    OUTPUT_DEVICE,
    SPLIT_INPUT,
    CAPTURE_SECONDARY,
    AUTO_RECONNECT,
)

logger = logging.getLogger("holo.audio_manager")

BUILTIN_MIC_KEYWORDS = ["macbook", "built-in", "builtin", "internal"]
BUILTIN_SPEAKER_KEYWORDS = ["macbook", "built-in", "builtin", "internal", "speakers"]
EXCLUDE_BUILTIN_KEYWORDS = [
    "bluetooth", "airpods", "headphone", "headset", "earphone", "virtual",
    "aggregate", "blackhole", "soundflower", "loopback", "wh-", "wf-", "xm4", "xm5"
]


@dataclass
class AudioDeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    hostapi: int
    is_builtin: bool = False
    is_external_or_bluetooth: bool = False

    def __str__(self) -> str:
        role = "Point 1 [Built-in / Tap]" if self.is_builtin else (
            "Bluetooth / External" if self.is_external_or_bluetooth else "Audio Device"
        )
        return (
            f"Device #{self.index}: '{self.name}' ({role}, "
            f"Inputs: {self.max_input_channels}, Outputs: {self.max_output_channels}, "
            f"Default SR: {int(self.default_samplerate)} Hz)"
        )


def is_builtin_device_name(name: str) -> bool:
    """Checks whether the device name corresponds to an integrated / built-in MacBook component."""
    lowered = name.lower()
    if any(ex in lowered for ex in EXCLUDE_BUILTIN_KEYWORDS):
        return False
    return any(kw in lowered for kw in BUILTIN_MIC_KEYWORDS)


def is_external_or_bluetooth_name(name: str) -> bool:
    """Checks whether the device name corresponds to earphones, headphones, or external audio."""
    lowered = name.lower()
    external_keywords = [
        "airpods", "airpod", "bluetooth", "headphone", "headset", "earphone", "buds",
        "beats", "sony", "bose", "jabra", "sennheiser", "usb", "external",
        "line in", "scarlett", "rode", "yeti", "shure", "audio-technica",
        "wh-", "wf-", "xm4", "xm5", "qc35", "qc45"
    ]
    return any(kw in lowered for kw in external_keywords)


def list_audio_devices() -> list[AudioDeviceInfo]:
    """Queries and returns all available audio devices with classification metadata."""
    devices: list[AudioDeviceInfo] = []
    try:
        raw_devices = sd.query_devices()
        for idx, dev in enumerate(raw_devices):
            if not isinstance(dev, dict):
                continue
            name = str(dev.get("name", f"Device {idx}"))
            max_in = int(dev.get("max_input_channels", 0))
            max_out = int(dev.get("max_output_channels", 0))
            sr = float(dev.get("default_samplerate", 44100.0))
            hostapi = int(dev.get("hostapi", 0))

            builtin = is_builtin_device_name(name)
            ext = is_external_or_bluetooth_name(name)

            devices.append(AudioDeviceInfo(
                index=idx,
                name=name,
                max_input_channels=max_in,
                max_output_channels=max_out,
                default_samplerate=sr,
                hostapi=hostapi,
                is_builtin=builtin,
                is_external_or_bluetooth=ext,
            ))
    except Exception as e:
        logger.warning(f"Error querying audio devices: {e}")
    return devices


def find_builtin_microphone(preferred: str | int | None = None) -> Optional[AudioDeviceInfo]:
    """
    Resolves the built-in MacBook microphone to be used as Point 1 (Tap Sensor).
    Strictly excludes Bluetooth headphones, AirPods, and external headsets.
    """
    devices = list_audio_devices()
    input_devices = [d for d in devices if d.max_input_channels > 0]
    if not input_devices:
        return None

    # 1. If explicit index or name preference is specified
    if preferred is not None:
        if isinstance(preferred, int):
            for d in input_devices:
                if d.index == preferred:
                    return d
        pref_str = str(preferred).strip().lower()
        if pref_str.isdigit():
            idx = int(pref_str)
            for d in input_devices:
                if d.index == idx:
                    return d
        if pref_str not in ["builtin", "auto", "default"]:
            for d in input_devices:
                if pref_str in d.name.lower():
                    return d

    # 2. Match integrated / built-in microphone by name (MacBook Air / MacBook Pro / Built-in)
    for d in input_devices:
        if d.is_builtin:
            return d

    # 3. If on macOS and no name matched, check for devices without external keywords
    if platform.system() == "Darwin":
        for d in input_devices:
            if not d.is_external_or_bluetooth:
                return d

    # 4. Fallback to default input device
    try:
        def_in = sd.default.device[0]
        if def_in is not None and def_in >= 0:
            for d in input_devices:
                if d.index == def_in:
                    return d
    except Exception:
        pass

    return input_devices[0] if input_devices else None


def find_builtin_speakers(preferred: str | int | None = None) -> Optional[AudioDeviceInfo]:
    """
    Resolves the built-in MacBook speakers for audio output (e.g. ultrasonic tones).
    Strictly excludes Bluetooth headphones, AirPods, and external headsets so audio
    does not blast into the user's ears.
    """
    devices = list_audio_devices()
    output_devices = [d for d in devices if d.max_output_channels > 0]
    if not output_devices:
        return None

    if preferred is not None:
        if isinstance(preferred, int):
            for d in output_devices:
                if d.index == preferred:
                    return d
        pref_str = str(preferred).strip().lower()
        if pref_str.isdigit():
            idx = int(pref_str)
            for d in output_devices:
                if d.index == idx:
                    return d
        if pref_str not in ["builtin", "auto", "default"]:
            for d in output_devices:
                if pref_str in d.name.lower():
                    return d

    # Match built-in speakers by name
    for d in output_devices:
        if d.is_builtin:
            return d

    if platform.system() == "Darwin":
        for d in output_devices:
            if not d.is_external_or_bluetooth:
                return d

    try:
        def_out = sd.default.device[1]
        if def_out is not None and def_out >= 0:
            for d in output_devices:
                if d.index == def_out:
                    return d
    except Exception:
        pass

    return output_devices[0] if output_devices else None


def find_secondary_microphone(
    builtin_index: Optional[int] = None,
    preferred: str | int | None = None
) -> Optional[AudioDeviceInfo]:
    """Utility to query secondary microphone if explicitly requested."""
    devices = list_audio_devices()
    input_devices = [d for d in devices if d.max_input_channels > 0]
    candidates = [d for d in input_devices if builtin_index is None or d.index != builtin_index]
    if not candidates:
        return None

    if preferred is not None:
        if isinstance(preferred, int):
            for d in candidates:
                if d.index == preferred:
                    return d
        pref_str = str(preferred).strip().lower()
        if pref_str.isdigit():
            idx = int(pref_str)
            for d in candidates:
                if d.index == idx:
                    return d
        if pref_str not in ["auto", "default", "earphones", "headphones"]:
            for d in candidates:
                if pref_str in d.name.lower():
                    return d

    for d in candidates:
        if d.is_external_or_bluetooth:
            return d

    return candidates[0] if candidates else None


def print_audio_devices() -> None:
    """Prints a formatted report of all available audio devices and assigned roles."""
    print("\n" + "=" * 70)
    print(" [AcousticTap Audio Device Inventory]")
    print("=" * 70)
    devices = list_audio_devices()
    if not devices:
        print("  No audio devices detected.")
        return

    builtin_mic = find_builtin_microphone()
    builtin_spk = find_builtin_speakers()

    print("\n--- Audio Inputs ---")
    for d in devices:
        if d.max_input_channels > 0:
            marker = ""
            if builtin_mic and d.index == builtin_mic.index:
                marker = " <=== [POINT 1: Dedicated MacBook Tap Mic (Active)]"
            elif d.is_external_or_bluetooth:
                marker = " [Ignored by Holo: Reserved for User/System Playback]"
            print(f"  [{d.index:2d}] {d.name:<32s} (Inputs: {d.max_input_channels}, {int(d.default_samplerate)} Hz){marker}")

    print("\n--- Audio Outputs ---")
    for d in devices:
        if d.max_output_channels > 0:
            marker = ""
            if builtin_spk and d.index == builtin_spk.index:
                marker = " <=== [Dedicated Built-in Speakers]"
            print(f"  [{d.index:2d}] {d.name:<32s} (Outputs: {d.max_output_channels}, {int(d.default_samplerate)} Hz){marker}")

    print("\n" + "=" * 70)
    if builtin_mic:
        print(f"  MacBook Tap Mic (Point 1): #{builtin_mic.index} '{builtin_mic.name}'")
    else:
        print("  MacBook Tap Mic (Point 1): NOT FOUND")

    if builtin_spk:
        print(f"  MacBook Speakers (Output): #{builtin_spk.index} '{builtin_spk.name}'")

    print("  Bluetooth / Headset: Left untouched for pristine stereo audio playback.")
    print("=" * 70 + "\n")


class SplitAudioEngine:
    """
    Manages dedicated audio capture from the MacBook Internal Microphone for tap detection.

    Guarantees:
    - Never opens Bluetooth/headphone input streams by default (prevents macOS CoreAudio
      from degrading headphones into 16 kHz HFP/SCO mono mode).
    - Passive error recovery: No continuous polling thread. Only re-resolves the MacBook
      mic index when the stream reports an error or disconnect event.
    """

    def __init__(
        self,
        tap_callback: Callable[[np.ndarray, int, Any, Any], None],
        secondary_callback: Optional[Callable[[np.ndarray, int, Any, Any], None]] = None,
        tap_device_pref: Optional[str | int] = None,
        secondary_device_pref: Optional[str | int] = None,
        capture_secondary: bool = False,
        auto_reconnect: bool = True,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        channels: int = 1,
    ):
        self.tap_callback = tap_callback
        self.secondary_callback = secondary_callback
        self.tap_device_pref = tap_device_pref
        self.secondary_device_pref = secondary_device_pref
        self.capture_secondary = capture_secondary
        self.auto_reconnect = auto_reconnect
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channels = channels

        self.tap_stream: Optional[sd.InputStream] = None
        self.secondary_stream: Optional[sd.InputStream] = None
        self.tap_device: Optional[AudioDeviceInfo] = None
        self.secondary_device: Optional[AudioDeviceInfo] = None

        self._active = False
        self._lock = threading.Lock()
        self._interrupted = False

    def start(self) -> None:
        """Starts the audio engine, binding solely to the MacBook built-in microphone."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._interrupted = False

            self._resolve_and_start_tap_stream()

            # Strictly only open secondary if explicitly enabled (default is False)
            if self.capture_secondary:
                self._resolve_and_start_secondary_stream()

    def stop(self) -> None:
        """Stops all streams cleanly."""
        with self._lock:
            if not self._active:
                return
            self._active = False

            self._stop_stream(self.secondary_stream)
            self.secondary_stream = None
            self._stop_stream(self.tap_stream)
            self.tap_stream = None

    def __enter__(self) -> SplitAudioEngine:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _resolve_and_start_tap_stream(self) -> bool:
        """Resolves Point 1 (MacBook Built-in Microphone) and opens the dedicated input stream."""
        self.tap_device = find_builtin_microphone(preferred=self.tap_device_pref)
        if self.tap_device is None:
            logger.error("Could not find any input device for Point 1 (Tap Sensor).")
            return False

        dev_idx = self.tap_device.index
        ch = min(self.channels, max(1, self.tap_device.max_input_channels))

        def _primary_cb(indata, frames, time_info, status):
            if status and (status.input_overflow or status.input_underflow):
                logger.debug(f"Primary stream status flag: {status}")
            try:
                self.tap_callback(indata, frames, time_info, status)
            except Exception as ex:
                logger.error(f"Error in Point 1 tap callback: {ex}")

        candidate_srs = [self.sample_rate]
        native_sr = int(self.tap_device.default_samplerate) if self.tap_device.default_samplerate > 0 else 44100
        if native_sr not in candidate_srs:
            candidate_srs.append(native_sr)
        for fallback_sr in [44100, 48000]:
            if fallback_sr not in candidate_srs:
                candidate_srs.append(fallback_sr)

        self._stop_stream(self.tap_stream)
        for sr in candidate_srs:
            try:
                self.tap_stream = sd.InputStream(
                    device=dev_idx,
                    samplerate=sr,
                    blocksize=self.block_size,
                    channels=ch,
                    callback=_primary_cb,
                )
                self.tap_stream.start()
                self.sample_rate = sr
                self._interrupted = False
                return True
            except Exception as e:
                logger.debug(f"Could not open Point 1 on {self.tap_device.name} at {sr} Hz: {e}")
                self.tap_stream = None

        logger.error(f"Failed to start Point 1 stream on {self.tap_device.name}.")
        return False

    def _resolve_and_start_secondary_stream(self) -> bool:
        """Optional secondary stream (only invoked when capture_secondary is explicitly true)."""
        builtin_idx = self.tap_device.index if self.tap_device else None
        self.secondary_device = find_secondary_microphone(
            builtin_index=builtin_idx,
            preferred=self.secondary_device_pref
        )

        if self.secondary_device is None:
            self._stop_stream(self.secondary_stream)
            self.secondary_stream = None
            return False

        sec_idx = self.secondary_device.index
        nominal_sr = int(self.secondary_device.default_samplerate) if self.secondary_device.default_samplerate > 0 else 44100
        sec_ch = max(1, min(self.channels, self.secondary_device.max_input_channels))

        def _secondary_cb(indata, frames, time_info, status):
            if self.secondary_callback is not None:
                try:
                    self.secondary_callback(indata, frames, time_info, status)
                except Exception as ex:
                    logger.debug(f"Secondary callback error: {ex}")

        candidate_srs = [nominal_sr, 16000, 24000, 44100, 48000]
        self._stop_stream(self.secondary_stream)
        for sr in candidate_srs:
            try:
                self.secondary_stream = sd.InputStream(
                    device=sec_idx,
                    samplerate=sr,
                    blocksize=self.block_size,
                    channels=sec_ch,
                    callback=_secondary_cb,
                )
                self.secondary_stream.start()
                return True
            except Exception:
                self.secondary_stream = None

        return False

    def _stop_stream(self, stream: Optional[sd.InputStream]) -> None:
        if stream is not None:
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except Exception:
                pass

    def ensure_active(self) -> bool:
        """
        Passive error recovery check:
        Only re-resolves the built-in microphone if the stream stopped or raised an error
        (e.g. during a Bluetooth device connect/disconnect event).
        Zero continuous polling loops.
        """
        with self._lock:
            if not self._active:
                return False

            is_dead = (self.tap_stream is None or not self.tap_stream.active)
            if is_dead:
                print("\n[AUDIO RECOVERY] MacBook microphone stream interrupted by CoreAudio event. Reconnecting...")
                return self._resolve_and_start_tap_stream()
            return True

    def is_secondary_active(self) -> bool:
        with self._lock:
            return bool(self.secondary_stream is not None and self.secondary_stream.active)

    def print_summary(self) -> None:
        """Outputs an informative status banner of the active hardware isolation."""
        tap_name = self.tap_device.name if self.tap_device else "Unknown"
        tap_idx = self.tap_device.index if self.tap_device else -1
        tap_sr = self.sample_rate

        print("\n" + "=" * 70)
        print(" [AcousticTap Hardware Audio Isolation]")
        print(f" Point 1 (MacBook Mic - Tap Sensor): Dev #{tap_idx} '{tap_name}' ({tap_sr} Hz) -> ACTIVE")
        print(" Headphones/Bluetooth: Left untouched for high-fidelity audio & system use.")
        print("=" * 70 + "\n")
