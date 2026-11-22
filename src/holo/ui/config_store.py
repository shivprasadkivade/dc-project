"""Configuration store for Holo UI.

Provides read-only access to core engine settings and a line-preserving surgical
writer for the dedicated top-level `ui:` configuration block in config/config.yaml.
Zero bytes of any other configuration section or comments are ever modified.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import yaml

from src.holo.config import CONFIG_PATH, load_config

DEFAULT_UI_CONFIG: dict[str, Any] = {
    "theme": "system",  # "dark", "light", or "system"
    "show_spectrogram": True,
    "min_confidence": 0.65,
}


class UIConfigStore:
    """Manages UI preferences with non-destructive line-preserving persistence."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = Path(config_path) if config_path is not None else CONFIG_PATH

    def get_full_config(self) -> dict[str, Any]:
        """Safely loads full YAML config dictionary."""
        return load_config(self.config_path)

    def get_ui_config(self) -> dict[str, Any]:
        """Returns the UI preferences merged with defaults."""
        full = self.get_full_config()
        raw_ui = full.get("ui", {}) if isinstance(full, dict) else {}
        cfg = dict(DEFAULT_UI_CONFIG)
        if isinstance(raw_ui, dict):
            for k, v in raw_ui.items():
                if v is not None:
                    cfg[k] = v
        # Normalize types
        cfg["theme"] = str(cfg.get("theme", "system")).lower()
        if cfg["theme"] not in ("system", "dark", "light"):
            cfg["theme"] = "system"
        cfg["show_spectrogram"] = bool(cfg.get("show_spectrogram", True))
        try:
            cfg["min_confidence"] = float(cfg.get("min_confidence", 0.65))
            cfg["min_confidence"] = max(0.0, min(1.0, cfg["min_confidence"]))
        except (ValueError, TypeError):
            cfg["min_confidence"] = 0.65
        return cfg

    def get_core_config(self) -> dict[str, Any]:
        """Returns read-only core engine configuration excluding UI block."""
        full = self.get_full_config()
        return {k: v for k, v in full.items() if k != "ui"}

    def format_ui_block(self, ui_config: dict[str, Any], newline: str = "\n") -> str:
        """Formats the ui: YAML block cleanly."""
        theme = str(ui_config.get("theme", "system")).lower()
        show_spec = "true" if ui_config.get("show_spectrogram", True) else "false"
        min_conf = float(ui_config.get("min_confidence", 0.65))

        lines = [
            "ui:",
            f'  theme: "{theme}"        # "dark", "light", or "system"',
            f"  show_spectrogram: {show_spec}",
            f"  min_confidence: {min_conf:.2f}",
        ]
        return newline.join(lines) + newline

    def save_ui_config(self, ui_dict: dict[str, Any]) -> None:
        """Surgically updates ONLY the `ui:` block in config/config.yaml.

        Preserves all comments, line formatting, and other sections byte-for-byte.
        """
        current_ui = self.get_ui_config()
        current_ui.update(ui_dict)

        if not self.config_path.exists():
            # Create a new minimal config file if non-existent
            content = self.format_ui_block(current_ui, newline="\n")
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(content, encoding="utf-8")
            return

        raw_bytes = self.config_path.read_bytes()
        # Detect newline convention
        newline = "\r\n" if b"\r\n" in raw_bytes else "\n"
        text = raw_bytes.decode("utf-8")

        ui_block_text = self.format_ui_block(current_ui, newline=newline)

        # Regex to locate top-level `ui:`
        # Matches `ui:` at start of line
        pattern = re.compile(r"(^|\r?\n)ui:[^\r\n]*(\r?\n(?:[ \t]+[^\r\n]*\r?\n|\r?\n)*|$)", re.MULTILINE)
        match = pattern.search(text)

        if match:
            # We found an existing ui: block
            start_idx = match.start()
            # If match started with newline from (^|\r?\n), keep the preceding newline
            prefix = ""
            if text[start_idx : start_idx + len(match.group(1))] == match.group(1) and match.group(1):
                start_idx += len(match.group(1))
                prefix = match.group(1)

            end_idx = match.end()
            new_text = text[:start_idx] + ui_block_text.rstrip("\r\n") + newline + text[end_idx:]
        else:
            # Append ui: block at the end
            if not text.endswith("\n") and not text.endswith("\r"):
                text += newline
            if not text.endswith(newline + newline):
                text += newline
            new_text = text + ui_block_text

        self.config_path.write_bytes(new_text.encode("utf-8"))


# Global singleton store instance
config_store = UIConfigStore()
