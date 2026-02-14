#!/usr/bin/env python3
"""Beginner-friendly GUI for operating and testing the automation harness.

Features:
- Run harness commands (simulate/script/swarm/learn/play/doctor) without CLI memorization.
- Record test actions from key presses and drag gestures into script JSON.
- Save/load script files and execute them directly from the GUI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT_DIR = Path(__file__).resolve().parent
HARNESS_PATH = ROOT_DIR / "automation_harness.py"
DEFAULT_SCRIPT_PATH = ROOT_DIR / "sample_script.json"
DEFAULT_TELEMETRY_PATH = ROOT_DIR / "sample_telemetry.jsonl"


def safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: str, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


class AutomationGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Taxvasion Automation GUI")
        self.geometry("1180x760")

        self._proc: subprocess.Popen[str] | None = None
        self._drag_start: tuple[int, int] | None = None
        self.recorded_actions: list[dict] = []

        self.python_exe = tk.StringVar(value=sys.executable)
        self.script_file = tk.StringVar(value=str(DEFAULT_SCRIPT_PATH))
        self.policy_file = tk.StringVar(value=str(ROOT_DIR / "learned_policy.json"))
        self.telemetry_file = tk.StringVar(value=str(DEFAULT_TELEMETRY_PATH))
        self.uinput_profile = tk.StringVar(value=str(ROOT_DIR / "sample_uinput_profile.json"))
        self.emuinput_root = tk.StringVar(value="")
        self.emuinput_bin_dir = tk.StringVar(value="")
        self.endpoint = tk.StringVar(value="")
        self.state = tk.StringVar(value="unknown|sparse|high")
        self.anomaly_config = tk.StringVar(value=str(ROOT_DIR / "sample_anomaly_config.json"))

        self.transport = tk.StringVar(value="print")
        self.mode = tk.StringVar(value="script")

        self.ticks = tk.StringVar(value="100")
        self.tick_delay = tk.StringVar(value="0.05")
        self.seed = tk.StringVar(value="")
        self.bots = tk.StringVar(value="5")
        self.bot = tk.StringVar(value="0")
        self.epsilon = tk.StringVar(value="0.15")
        self.timeout = tk.StringVar(value="3.0")

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container, padding=10)
        right = ttk.Frame(container, padding=10)
        container.add(left, weight=2)
        container.add(right, weight=3)

        self._build_runner_panel(left)
        self._build_record_panel(left)
        self._build_output_panel(right)

    def _labeled_entry(self, parent: ttk.Frame, label: str, var: tk.StringVar, row: int, width: int = 34) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="we", pady=2)

    def _browse_into(self, var: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            var.set(path)

    def _build_runner_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Run Bot / Tests", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.mode, state="readonly", values=["simulate", "script", "swarm", "learn", "play", "doctor", "anomaly", "anomaly-scan"], width=18).grid(row=0, column=1, sticky="w")

        ttk.Label(box, text="Transport").grid(row=1, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.transport, state="readonly", values=["print", "websocket", "uinput", "emuinput"], width=18).grid(row=1, column=1, sticky="w")

        self._labeled_entry(box, "Python", self.python_exe, 2)
        self._labeled_entry(box, "Script file", self.script_file, 3)
        ttk.Button(box, text="Browse", command=lambda: self._browse_into(self.script_file, [("JSON", "*.json"), ("All", "*.*")])).grid(row=3, column=2, padx=4)

        self._labeled_entry(box, "Policy file", self.policy_file, 4)
        ttk.Button(box, text="Browse", command=lambda: self._browse_into(self.policy_file, [("JSON", "*.json"), ("All", "*.*")])).grid(row=4, column=2, padx=4)

        self._labeled_entry(box, "Telemetry file", self.telemetry_file, 5)
        ttk.Button(box, text="Browse", command=lambda: self._browse_into(self.telemetry_file, [("JSONL", "*.jsonl"), ("All", "*.*")])).grid(row=5, column=2, padx=4)

        self._labeled_entry(box, "Anomaly config", self.anomaly_config, 6)
        ttk.Button(box, text="Browse", command=lambda: self._browse_into(self.anomaly_config, [("JSON", "*.json"), ("All", "*.*")])).grid(row=6, column=2, padx=4)

        self._labeled_entry(box, "WebSocket endpoint", self.endpoint, 7)
        self._labeled_entry(box, "uinput profile", self.uinput_profile, 8)
        self._labeled_entry(box, "emuinput root", self.emuinput_root, 9)
        self._labeled_entry(box, "emuinput bin dir", self.emuinput_bin_dir, 10)
        self._labeled_entry(box, "State", self.state, 11)

        self._labeled_entry(box, "Ticks", self.ticks, 12)
        self._labeled_entry(box, "Tick delay", self.tick_delay, 13)
        self._labeled_entry(box, "Seed", self.seed, 14)
        self._labeled_entry(box, "Bots", self.bots, 15)
        self._labeled_entry(box, "Bot ID", self.bot, 16)
        self._labeled_entry(box, "Epsilon", self.epsilon, 17)
        self._labeled_entry(box, "Timeout", self.timeout, 18)

        button_row = ttk.Frame(box)
        button_row.grid(row=19, column=0, columnspan=3, sticky="we", pady=(8, 0))
        ttk.Button(button_row, text="Run", command=self.run_command).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row, text="Stop", command=self.stop_command).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_row, text="Copy command", command=self.copy_command).pack(side=tk.LEFT)

    def _build_record_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Record Keystrokes + Drags", padding=8)
        box.pack(fill=tk.BOTH, expand=True)

        instructions = (
            "Click inside the capture area, then press keys or drag mouse.\n"
            "- Key press -> {\"cmd\": \"keyevent\", \"key_code\": ...}\n"
            "- Drag -> {\"cmd\": \"swipe\", ...}\n"
            "Use Save to write a script JSON for harness script mode."
        )
        ttk.Label(box, text=instructions, justify=tk.LEFT).pack(anchor="w")

        self.capture = tk.Canvas(box, bg="#101820", height=200, highlightthickness=1, highlightbackground="#607080")
        self.capture.pack(fill=tk.X, pady=6)
        self.capture.create_text(12, 12, text="Capture area", fill="#d8e6ff", anchor="nw")

        self.capture.bind("<ButtonPress-1>", self.on_mouse_down)
        self.capture.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.capture.bind("<B1-Motion>", self.on_mouse_drag)
        self.capture.bind("<Key>", self.on_key_press)
        self.capture.bind("<FocusIn>", lambda _e: self.capture.configure(highlightbackground="#33aa55"))
        self.capture.bind("<FocusOut>", lambda _e: self.capture.configure(highlightbackground="#607080"))

        row = ttk.Frame(box)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Focus capture", command=lambda: self.capture.focus_set()).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Add tap (center)", command=self.add_tap).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Clear", command=self.clear_recording).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(row, text="Save script", command=self.save_recording).pack(side=tk.LEFT)

        self.record_list = tk.Listbox(box, height=10)
        self.record_list.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _build_output_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Output", padding=8)
        box.pack(fill=tk.BOTH, expand=True)

        self.output = tk.Text(box, wrap="word")
        self.output.pack(fill=tk.BOTH, expand=True)
        self.output.insert("end", "GUI ready.\n")

    def log(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _base_command(self) -> list[str]:
        mode = self.mode.get().strip()
        cmd = [self.python_exe.get().strip() or sys.executable, str(HARNESS_PATH), mode]

        if mode in {"script", "play", "swarm", "simulate"}:
            cmd += ["--tick-delay", str(safe_float(self.tick_delay.get(), 0.05))]
        if mode in {"script", "swarm", "play", "simulate"}:
            cmd += ["--ticks", str(safe_int(self.ticks.get(), 100))]

        seed = self.seed.get().strip()
        if seed and mode in {"simulate", "swarm", "play"}:
            cmd += ["--seed", seed]

        if mode == "script":
            cmd += ["--file", self.script_file.get().strip()]
        elif mode == "swarm":
            cmd += ["--bots", str(safe_int(self.bots.get(), 5))]
        elif mode == "learn":
            cmd += ["--input", self.telemetry_file.get().strip(), "--output", self.policy_file.get().strip()]
        elif mode == "play":
            cmd += [
                "--policy",
                self.policy_file.get().strip(),
                "--state",
                self.state.get().strip(),
                "--bot",
                str(safe_int(self.bot.get(), 0)),
                "--epsilon",
                str(safe_float(self.epsilon.get(), 0.15)),
            ]
        elif mode == "doctor":
            cmd += ["--timeout", str(safe_float(self.timeout.get(), 3.0)), "--telemetry", self.telemetry_file.get().strip(), "--check-uinput", "--check-emuinput", "--emuinput-root", self.emuinput_root.get().strip()]
        elif mode == "anomaly":
            cmd += ["--config", self.anomaly_config.get().strip(), "--cycles", "1"]
        elif mode == "anomaly-scan":
            cmd += ["--config", self.anomaly_config.get().strip()]

        if mode in {"script", "swarm", "play", "anomaly"}:
            transport = self.transport.get().strip()
            cmd += ["--transport", transport]
            if transport == "websocket" and self.endpoint.get().strip():
                cmd += ["--endpoint", self.endpoint.get().strip()]
            if transport == "uinput" and self.uinput_profile.get().strip():
                cmd += ["--uinput-profile", self.uinput_profile.get().strip()]
            if transport == "emuinput":
                if self.emuinput_root.get().strip():
                    cmd += ["--emuinput-root", self.emuinput_root.get().strip()]
                if self.emuinput_bin_dir.get().strip():
                    cmd += ["--emuinput-bin-dir", self.emuinput_bin_dir.get().strip()]

        return cmd

    def copy_command(self) -> None:
        cmd = self._base_command()
        text = " ".join(subprocess.list2cmdline([arg]) for arg in cmd)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.log(f"Copied command: {text}")

    def run_command(self) -> None:
        if self._proc and self._proc.poll() is None:
            messagebox.showwarning("Already running", "A command is already running. Stop it first.")
            return

        cmd = self._base_command()
        self.log(f"$ {' '.join(cmd)}")

        def worker() -> None:
            try:
                self._proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    self.after(0, self.log, line.rstrip("\n"))
                rc = self._proc.wait()
                self.after(0, self.log, f"[exit code {rc}]")
            except Exception as exc:
                self.after(0, self.log, f"[error] {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def stop_command(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.log("Sent terminate signal.")
        else:
            self.log("No running process.")

    def _push_action(self, action: dict) -> None:
        self.recorded_actions.append(action)
        self.record_list.insert(tk.END, json.dumps(action))

    def on_mouse_down(self, event: tk.Event) -> None:
        self.capture.focus_set()
        self._drag_start = (int(event.x), int(event.y))

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        self.capture.delete("drag_preview")
        x1, y1 = self._drag_start
        self.capture.create_line(x1, y1, event.x, event.y, fill="#ffaa00", width=2, tags="drag_preview")

    def on_mouse_up(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = int(event.x), int(event.y)
        self.capture.delete("drag_preview")
        self._drag_start = None

        dist = abs(x2 - x1) + abs(y2 - y1)
        if dist < 10:
            self._push_action({"cmd": "tap", "x": x2, "y": y2})
        else:
            self._push_action({"cmd": "swipe", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": 350, "steps": 24})

    def on_key_press(self, event: tk.Event) -> None:
        keycode = int(getattr(event, "keycode", 0) or 0)
        keysym = str(getattr(event, "keysym", ""))
        self._push_action({"cmd": "keyevent", "key_code": keycode, "key": keysym})

    def add_tap(self) -> None:
        w = int(self.capture.winfo_width() or 400)
        h = int(self.capture.winfo_height() or 200)
        self._push_action({"cmd": "tap", "x": w // 2, "y": h // 2})

    def clear_recording(self) -> None:
        self.recorded_actions.clear()
        self.record_list.delete(0, tk.END)

    def save_recording(self) -> None:
        if not self.recorded_actions:
            messagebox.showinfo("Nothing to save", "No recorded actions yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.recorded_actions, f, indent=2)
        self.script_file.set(path)
        self.log(f"Saved {len(self.recorded_actions)} recorded actions to {path}")


if __name__ == "__main__":
    if not HARNESS_PATH.exists():
        raise SystemExit("automation_harness.py not found in current directory")
    app = AutomationGUI()
    app.mainloop()
