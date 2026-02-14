#!/usr/bin/env python3
"""Generic game automation harness for scripted testing and world population.

Adds a lightweight "learn -> play" path:
- `learn` builds a simple state->action frequency policy from telemetry logs.
- `play` uses that learned policy (with epsilon exploration) to choose commands.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

DIRECTIONS = ["north", "south", "east", "west"]

# Linux input keycodes used by emuinput/uinputd (from emuinput prototype conventions).
KEYCODE_W = 17
KEYCODE_A = 30
KEYCODE_S = 31
KEYCODE_D = 32
KEYCODE_F = 33
KEYCODE_ENTER = 28


async def run_subprocess(cmd: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    out = out_b.decode("utf-8", errors="replace").strip()
    err = err_b.decode("utf-8", errors="replace").strip()
    return proc.returncode, out, err


async def run_subprocess_bytes(cmd: list[str]) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out_b, err_b = await proc.communicate()
    return proc.returncode, out_b, err_b


def subprocess_run(cmd: list[str], env: dict[str, str] | None = None, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


@dataclass
class Metrics:
    sent: int = 0
    started_at: float = field(default_factory=time.time)

    def mark_sent(self) -> None:
        self.sent += 1

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(time.time() - self.started_at, 1e-9)
        return {
            "commands_sent": self.sent,
            "elapsed_seconds": round(elapsed, 3),
            "commands_per_second": round(self.sent / elapsed, 2),
        }


class Controller:
    async def start(self) -> None:
        """Optional transport warmup hook called once per command run."""
        return None

    async def send(self, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class PrintController(Controller):
    async def send(self, payload: dict[str, Any]) -> None:
        print(json.dumps(payload, separators=(",", ":")))


class WebSocketController(Controller):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._ws = None

    async def _connect(self):
        if self._ws is not None:
            return
        import websockets

        self._ws = await websockets.connect(self.endpoint)

    async def send(self, payload: dict[str, Any]) -> None:
        await self._connect()
        assert self._ws is not None
        await self._ws.send(json.dumps(payload))

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


class UInputController(Controller):
    """Host-side uinput bridge.

    Uses a profile file that maps action names to command templates.
    Template variables are expanded from payload keys.
    """

    def __init__(self, profile_path: str):
        if not profile_path:
            raise ValueError("--uinput-profile is required for uinput transport")
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("uinput profile must be a JSON object")
        self.profile = data

    def _action_key(self, payload: dict[str, Any]) -> str:
        cmd = str(payload.get("cmd", ""))
        if cmd == "move":
            direction = str(payload.get("direction", "north"))
            return f"move_{direction}"
        return cmd

    @staticmethod
    def _coerce_cmd(template: Any, payload: dict[str, Any]) -> list[str]:
        ctx = {k: str(v) for k, v in payload.items()}
        if isinstance(template, str):
            return shlex.split(template.format(**ctx))
        if isinstance(template, list):
            out = []
            for item in template:
                if not isinstance(item, str):
                    raise ValueError("uinput template list items must be strings")
                out.append(item.format(**ctx))
            return out
        raise ValueError("uinput template must be string or string-list")

    async def send(self, payload: dict[str, Any]) -> None:
        action_key = self._action_key(payload)
        template = self.profile.get(action_key)
        if template is None:
            if payload.get("cmd") == "wait":
                return
            raise ValueError(f"No uinput mapping for action '{action_key}'")

        cmd = self._coerce_cmd(template, payload)
        code, out, err = await run_subprocess(cmd)
        if code != 0:
            raise RuntimeError(f"uinput command failed: {' '.join(cmd)} | {err or out}")


class EmuInputController(Controller):
    """Adapter for the external `emuinput` package from your frames repo.

    This backend manages the `uinputd` daemon and sends events through its socket protocol.
    """

    def __init__(
        self,
        emuinput_root: str,
        serial: str,
        host_port: int,
        adb_exe: str,
        adb_server_port: int,
        bin_dir: str,
        autofix: bool = True,
    ):
        if not emuinput_root:
            raise ValueError("--emuinput-root is required for emuinput transport")
        self.emuinput_root = emuinput_root
        if emuinput_root not in sys.path:
            sys.path.insert(0, emuinput_root)

        Adb = importlib.import_module("emuinput.adb").Adb
        EmuController = importlib.import_module("emuinput.controller").EmuController
        self.serial = serial
        self.host_port = int(host_port)
        self.adb_server_port = int(adb_server_port)
        self.autofix = bool(autofix)
        self.bin_dir = bin_dir
        self._adb_candidates = self._resolve_adb_candidates(
            adb_exe=adb_exe,
            emuinput_root=self.emuinput_root,
            bin_dir=self.bin_dir,
        )

        self._adb = Adb(
            adb_exe=self._adb_candidates[0],
            adb_server_port=self.adb_server_port,
            timeout=18.0,
        )
        self._controller = EmuController(
            serial=self.serial,
            adb=self._adb,
            host_port=self.host_port,
            bin_dir=self.bin_dir,
        )
        self._ready = False

    @staticmethod
    def _resolve_adb_candidates(adb_exe: str, emuinput_root: str = "", bin_dir: str = "") -> list[str]:
        raw = str(adb_exe or "adb").strip() or "adb"
        candidates: list[str] = []
        expanded = os.path.expandvars(raw)
        # Prefer adb binaries that already exist in the configured emuinput tree.
        root = str(emuinput_root or "").strip()
        root_expanded = os.path.expanduser(os.path.expandvars(root))
        if root_expanded:
            for path in [
                os.path.join(root_expanded, "adb"),
                os.path.join(root_expanded, "adb.exe"),
                os.path.join(root_expanded, "bin", "android", "adb"),
                os.path.join(root_expanded, "bin", "android", "adb.exe"),
                os.path.join(root_expanded, "emuinput", "adb"),
                os.path.join(root_expanded, "emuinput", "adb.exe"),
                os.path.join(root_expanded, "emuinput", "bin", "android", "adb"),
                os.path.join(root_expanded, "emuinput", "bin", "android", "adb.exe"),
            ]:
                if os.path.isfile(path):
                    candidates.append(path)
        explicit_bin_dir = str(bin_dir or "").strip()
        explicit_bin_dir = os.path.expanduser(os.path.expandvars(explicit_bin_dir))
        if explicit_bin_dir:
            for path in [
                os.path.join(explicit_bin_dir, "adb"),
                os.path.join(explicit_bin_dir, "adb.exe"),
            ]:
                if os.path.isfile(path):
                    candidates.append(path)
        candidates.append(raw)
        if expanded != raw:
            candidates.append(expanded)
        which = shutil.which(raw)
        if which:
            candidates.append(which)
        candidates.extend(
            [
                r"C:\Program Files\Netease\MuMuPlayer\nx_main\adb.exe",
                r"C:\Program Files\Netease\MuMuPlayer\nx_device\12.0\shell\adb.exe",
                r"C:\Program Files\Netease\MuMuPlayer\shell\adb.exe",
                "adb",
            ],
        )
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item.strip())
        return out

    def _rebuild_controller(self, adb_exe: str) -> None:
        Adb = importlib.import_module("emuinput.adb").Adb
        EmuController = importlib.import_module("emuinput.controller").EmuController
        self._adb = Adb(adb_exe=adb_exe, adb_server_port=self.adb_server_port, timeout=18.0)
        self._controller = EmuController(
            serial=self.serial,
            adb=self._adb,
            host_port=self.host_port,
            bin_dir=self.bin_dir,
        )

    @staticmethod
    def _run_adb_cmd(adb_exe: str, server_port: int, args: list[str], timeout_s: float = 10.0) -> bool:
        env = os.environ.copy()
        env["ADB_SERVER_PORT"] = str(int(server_port))
        proc = subprocess_run([adb_exe, *args], env=env, timeout=timeout_s)
        return proc.returncode == 0

    def _attempt_adb_repair(self, adb_exe: str) -> None:
        # Best-effort recovery mirrors prototype.py autofix sequence.
        self._run_adb_cmd(adb_exe, self.adb_server_port, ["kill-server"], timeout_s=8.0)
        self._run_adb_cmd(adb_exe, self.adb_server_port, ["start-server"], timeout_s=10.0)
        self._run_adb_cmd(adb_exe, self.adb_server_port, ["connect", self.serial], timeout_s=12.0)
        self._run_adb_cmd(adb_exe, self.adb_server_port, ["-s", self.serial, "get-state"], timeout_s=8.0)

    def _ensure_daemon_with_retry(self) -> None:
        last_exc: Exception | None = None
        for adb_idx, adb_exe in enumerate(self._adb_candidates[:6]):
            max_tries = 2 if self.autofix else 1
            for local_try in range(max_tries):
                try:
                    self._rebuild_controller(adb_exe)
                    self._controller.ensure_daemon()
                    return
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    try:
                        self._controller.close()
                    except Exception:
                        pass
                    if self.autofix and local_try + 1 < max_tries:
                        self._attempt_adb_repair(adb_exe)
                        time.sleep(0.3)
            if adb_idx + 1 < min(6, len(self._adb_candidates)):
                time.sleep(0.25)

        if last_exc is not None:
            raise RuntimeError(f"Failed to initialize emuinput after retries: {last_exc}") from last_exc
        raise RuntimeError("Failed to initialize emuinput after retries")

    async def _ensure_ready(self, *, force: bool = False) -> None:
        if self._ready and not force:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_daemon_with_retry)
        self._ready = True

    async def start(self) -> None:
        # Always verify/warm the uinput daemon at command start.
        await self._ensure_ready(force=True)

    async def _call_with_recover(self, fn) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, fn)
        except Exception:
            # Daemon/socket may have dropped after startup; force warmup and retry once.
            await self._ensure_ready(force=True)
            await loop.run_in_executor(None, fn)

    async def send(self, payload: dict[str, Any]) -> None:
        await self._ensure_ready()
        cmd = str(payload.get("cmd", ""))

        if cmd == "tap":
            x = int(payload["x"])
            y = int(payload["y"])
            await self._call_with_recover(lambda: self._controller.tap(x, y, int(payload.get("down_ms", 70))))
            return

        if cmd == "swipe":
            x1 = int(payload["x1"])
            y1 = int(payload["y1"])
            x2 = int(payload["x2"])
            y2 = int(payload["y2"])
            await self._call_with_recover(
                lambda: self._controller.drag(
                    x1,
                    y1,
                    x2,
                    y2,
                    duration_ms=int(payload.get("duration_ms", 350)),
                    steps=int(payload.get("steps", 24)),
                ),
            )
            return

        if cmd == "move":
            direction = str(payload.get("direction", "north"))
            steps = max(1, int(payload.get("steps", 1)))
            key_for = {
                "north": KEYCODE_W,
                "south": KEYCODE_S,
                "east": KEYCODE_D,
                "west": KEYCODE_A,
            }
            key_code = key_for.get(direction, KEYCODE_W)
            for _ in range(steps):
                await self._call_with_recover(lambda kc=key_code: self._controller.key_tap(kc))
            return

        if cmd == "interact":
            await self._call_with_recover(lambda: self._controller.key_tap(KEYCODE_F))
            return

        if cmd in {"say", "text"}:
            text = str(payload.get("message", payload.get("text", "")))
            await self._call_with_recover(lambda: self._controller.type_text(text))
            return

        if cmd == "keyevent":
            key_code = int(payload["key_code"])
            await self._call_with_recover(lambda: self._controller.key_tap(key_code))
            return

        if cmd == "press_enter":
            await self._call_with_recover(lambda: self._controller.key_tap(KEYCODE_ENTER))
            return

        if cmd == "wait":
            return

        raise ValueError(f"Unsupported emuinput payload cmd={cmd!r}")

    async def close(self) -> None:
        if self._ready:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._controller.close)
            self._ready = False


def probe_emuinput_root(emuinput_root: str) -> tuple[bool, str]:
    if not emuinput_root:
        return False, "missing --emuinput-root"
    expected = os.path.join(emuinput_root, "emuinput", "controller.py")
    if os.path.exists(expected):
        return True, expected
    return False, f"emuinput package not found at {expected}"


async def probe_websocket(endpoint: str, timeout_s: float) -> tuple[bool, str]:
    try:
        import websockets
    except ImportError:
        return False, "missing dependency: pip install websockets"

    try:
        ws = await asyncio.wait_for(websockets.connect(endpoint), timeout=timeout_s)
        await ws.close()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def probe_uinput() -> tuple[bool, str]:
    if os.path.exists("/dev/uinput"):
        return True, "/dev/uinput exists"
    return False, "/dev/uinput not found (load uinput kernel module / permissions)"


def probe_vision() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401

        return True, "opencv import ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"opencv unavailable: {exc}"


def resolve_powershell_executable() -> str | None:
    for candidate in ["powershell.exe", "powershell", "pwsh.exe", "pwsh"]:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def build_powershell_diagnostic_battery(args: argparse.Namespace) -> list[dict[str, str]]:
    serial = str(getattr(args, "serial", "127.0.0.1:5555"))
    emu_root = str(getattr(args, "emuinput_root", "")).strip()
    emu_bin = str(getattr(args, "emuinput_bin_dir", "")).strip()
    adb_exe = str(getattr(args, "adb_exe", "adb")).strip() or "adb"
    adb_candidates = EmuInputController._resolve_adb_candidates(adb_exe=adb_exe, emuinput_root=emu_root, bin_dir=emu_bin)[:8]

    def q(value: str) -> str:
        return value.replace("'", "''")

    quoted_candidates = ", ".join([f"'{q(item)}'" for item in adb_candidates])

    return [
        {
            "name": "host_summary",
            "command": "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture,LastBootUpTime | ConvertTo-Json -Compress",
        },
        {
            "name": "powershell_summary",
            "command": "$PSVersionTable | ConvertTo-Json -Compress",
        },
        {
            "name": "python_summary",
            "command": "python --version; py --version",
        },
        {
            "name": "emuinput_paths",
            "command": (
                f"$root='{q(emu_root)}'; $bin='{q(emu_bin)}'; "
                "[pscustomobject]@{ emuinput_root=$root; emuinput_root_exists=(Test-Path $root); "
                "emuinput_bin_dir=$bin; emuinput_bin_dir_exists=(Test-Path $bin) } | ConvertTo-Json -Compress"
            ),
        },
        {
            "name": "adb_candidate_inventory",
            "command": (
                f"$c=@({quoted_candidates}); $out=@(); foreach($p in $c){{ "
                "$exists=Test-Path $p; $ver=''; "
                "if($exists){ try{ $ver=& $p version 2>&1 | Out-String }catch{ $ver=$_.Exception.Message } }; "
                "$out += [pscustomobject]@{ path=$p; exists=$exists; version=$ver.Trim() } }; "
                "$out | ConvertTo-Json -Depth 4 -Compress"
            ),
        },
        {
            "name": "adb_server_health",
            "command": (
                f"$c=@({quoted_candidates}); $adb=($c | Where-Object {{ Test-Path $_ }} | Select-Object -First 1); "
                "if(-not $adb){ $adb='adb' }; & $adb start-server; & $adb devices -l"
            ),
        },
        {
            "name": "target_serial_state",
            "command": (
                f"$c=@({quoted_candidates}); $adb=($c | Where-Object {{ Test-Path $_ }} | Select-Object -First 1); "
                f"if(-not $adb){{ $adb='adb' }}; & $adb -s '{q(serial)}' get-state; "
                f"& $adb -s '{q(serial)}' shell getprop ro.product.model; "
                f"& $adb -s '{q(serial)}' shell getprop ro.build.version.release"
            ),
        },
        {
            "name": "screencap_probe",
            "command": (
                f"$c=@({quoted_candidates}); $adb=($c | Where-Object {{ Test-Path $_ }} | Select-Object -First 1); "
                f"if(-not $adb){{ $adb='adb' }}; $out=Join-Path $env:TEMP 'taxvasion_diag_screen.png'; "
                f"& $adb -s '{q(serial)}' exec-out screencap -p > $out; "
                "[pscustomobject]@{ path=$out; exists=(Test-Path $out); "
                "bytes=((Get-Item $out -ErrorAction SilentlyContinue).Length) } | ConvertTo-Json -Compress"
            ),
        },
        {
            "name": "port_probe",
            "command": "Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.LocalPort -in 5037,27183 } | Select-Object LocalAddress,LocalPort,OwningProcess | ConvertTo-Json -Compress",
        },
    ]


def command_powershell_diagnostics(args: argparse.Namespace) -> None:
    battery = build_powershell_diagnostic_battery(args)
    result: dict[str, Any] = {
        "mode": "powershell_diagnostics",
        "requested_serial": args.serial,
        "battery": battery,
    }

    if args.print_only:
        print(json.dumps(result, indent=2))
        return

    ps_exe = resolve_powershell_executable()
    if not ps_exe:
        result["ok"] = False
        result["error"] = "No PowerShell executable found (expected powershell or pwsh)."
        print(json.dumps(result, indent=2))
        return

    checks: list[dict[str, Any]] = []
    for item in battery:
        started = time.time()
        proc = subprocess_run(
            [ps_exe, "-NoProfile", "-NonInteractive", "-Command", item["command"]],
            timeout=float(args.timeout),
        )
        checks.append(
            {
                "name": item["name"],
                "returncode": proc.returncode,
                "elapsed_seconds": round(time.time() - started, 3),
                "stdout": (proc.stdout or "").strip(),
                "stderr": (proc.stderr or "").strip(),
            },
        )

    result["powershell_executable"] = ps_exe
    result["checks"] = checks
    result["ok"] = all(c["returncode"] == 0 for c in checks)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


def load_script(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Script file must be a list of command objects")
    return data


def load_policy(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Policy file must be a dict")
    return data


def load_json_object(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def match_template_path(
    screenshot_path: str,
    template_path: str,
    threshold: float,
) -> tuple[bool, float]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Template matching requires OpenCV: pip install opencv-python") from exc

    haystack = cv2.imread(screenshot_path, cv2.IMREAD_GRAYSCALE)
    needle = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if haystack is None:
        raise RuntimeError(f"Failed to read screenshot: {screenshot_path}")
    if needle is None:
        raise RuntimeError(f"Failed to read template: {template_path}")
    if needle.shape[0] > haystack.shape[0] or needle.shape[1] > haystack.shape[1]:
        return False, 0.0

    result = cv2.matchTemplate(haystack, needle, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    score = float(max_val)
    return score >= threshold, score


async def capture_frame(config: dict[str, Any], output_path: str) -> str:
    cmd_template = config.get("screenshot_cmd")
    if isinstance(cmd_template, str):
        cmd = cmd_template.format(output=output_path)
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out_b, err_b = await proc.communicate()
        code = proc.returncode
        out = out_b.decode("utf-8", errors="replace").strip()
        err = err_b.decode("utf-8", errors="replace").strip()
    elif isinstance(cmd_template, list) and all(isinstance(item, str) for item in cmd_template):
        cmd = [item.format(output=output_path) for item in cmd_template]
        code, out, err = await run_subprocess(cmd)
    else:
        # Default capture mode is direct emulator screenshot via adb serial.
        adb_exe = str(config.get("adb_exe", "adb"))
        adb_server_port = str(config.get("adb_server_port", "5037"))
        adb_serial = str(config.get("adb_serial", "")).strip()
        cmd = [adb_exe, "-P", adb_server_port]
        if adb_serial:
            cmd += ["-s", adb_serial]
        cmd += ["exec-out", "screencap", "-p"]
        code, out_b, err_b = await run_subprocess_bytes(cmd)
        out = out_b.decode("utf-8", errors="replace").strip()
        err = err_b.decode("utf-8", errors="replace").strip()
        if code == 0:
            with open(output_path, "wb") as f:
                f.write(out_b)
    if code != 0:
        raise RuntimeError(f"screenshot_cmd failed ({code}): {err or out}")
    if not os.path.exists(output_path):
        raise RuntimeError(f"screenshot file not produced: {output_path}")
    return output_path


async def wait_for_template(
    config: dict[str, Any],
    template_path: str,
    threshold: float,
    timeout_s: float,
    poll_s: float,
    capture_path: str,
) -> tuple[bool, float]:
    deadline = time.time() + timeout_s
    best = 0.0
    while time.time() < deadline:
        shot = await capture_frame(config, capture_path)
        ok, score = match_template_path(shot, template_path, threshold)
        best = max(best, score)
        if ok:
            return True, score
        await asyncio.sleep(poll_s)
    return False, best


async def tap_named(controller: Controller, taps: dict[str, Any], name: str, metrics: Metrics) -> None:
    point = taps.get(name)
    if not isinstance(point, dict):
        raise ValueError(f"Missing tap target '{name}' in anomaly config")
    payload = {
        "cmd": "tap",
        "x": int(point["x"]),
        "y": int(point["y"]),
        "down_ms": int(point.get("down_ms", 70)),
    }
    await controller.send(payload)
    metrics.mark_sent()


async def run_anomaly_cycle(controller: Controller, args: argparse.Namespace, metrics: Metrics) -> dict[str, Any]:
    config = load_json_object(args.config)
    if not config.get("adb_exe"):
        config["adb_exe"] = getattr(args, "emuinput_adb_exe", "adb")
    if not config.get("adb_server_port"):
        config["adb_server_port"] = getattr(args, "emuinput_adb_server_port", 5037)
    if not config.get("adb_serial"):
        config["adb_serial"] = getattr(args, "screenshot_serial", "") or getattr(args, "emuinput_serial", "")

    taps = config.get("tap_targets", {})
    templates = config.get("templates", {})
    if not isinstance(taps, dict) or not isinstance(templates, dict):
        raise ValueError("anomaly config requires tap_targets and templates objects")

    threshold = float(config.get("threshold", args.threshold))
    poll_s = float(config.get("poll_seconds", args.poll_seconds))
    capture_path = str(config.get("capture_path", args.capture_path))

    arrival_timeout = float(args.arrival_timeout)
    stable_seconds = float(args.stable_seconds)

    await tap_named(controller, taps, "open_overview", metrics)
    await asyncio.sleep(args.tap_delay)

    ok, score = await wait_for_template(
        config,
        str(templates["overview_open"]),
        threshold,
        args.short_timeout,
        poll_s,
        capture_path,
    )
    if not ok:
        return {"ok": False, "step": "overview_open", "score": score}

    await tap_named(controller, taps, "select_anomaly", metrics)
    await asyncio.sleep(args.tap_delay)
    ok, score = await wait_for_template(
        config,
        str(templates["warp_menu_open"]),
        threshold,
        args.short_timeout,
        poll_s,
        capture_path,
    )
    if not ok:
        return {"ok": False, "step": "warp_menu_open", "score": score}

    await tap_named(controller, taps, "warp", metrics)
    await asyncio.sleep(args.tap_delay)
    ok, score = await wait_for_template(
        config,
        str(templates["arrival_ack"]),
        threshold,
        arrival_timeout,
        poll_s,
        capture_path,
    )
    if not ok:
        return {"ok": False, "step": "arrival_ack", "score": score, "reason": "warp_failed_or_no_change"}

    ok, score = await wait_for_template(
        config,
        str(templates["target_all_visible"]),
        threshold,
        args.short_timeout,
        poll_s,
        capture_path,
    )
    if not ok:
        return {"ok": False, "step": "target_all_visible", "score": score}

    await tap_named(controller, taps, "lock_all", metrics)
    await asyncio.sleep(args.tap_delay)

    clear_start: float | None = None
    while True:
        await tap_named(controller, taps, "select_npc", metrics)
        await asyncio.sleep(args.tap_delay)
        await tap_named(controller, taps, "focus_fire", metrics)

        shot = await capture_frame(config, capture_path)
        lockable, lock_score = match_template_path(shot, str(templates["lockable_targets_present"]), threshold)
        npcs, npcs_score = match_template_path(shot, str(templates["npcs_in_overview_present"]), threshold)

        if lockable or npcs:
            clear_start = None
        else:
            if clear_start is None:
                clear_start = time.time()
            if time.time() - clear_start >= stable_seconds:
                return {
                    "ok": True,
                    "step": "site_complete",
                    "lock_score": lock_score,
                    "npcs_score": npcs_score,
                }
        await asyncio.sleep(args.combat_poll_seconds)


async def command_anomaly_scan(args: argparse.Namespace) -> None:
    config = load_json_object(args.config)
    if not config.get("adb_exe"):
        config["adb_exe"] = getattr(args, "emuinput_adb_exe", "adb")
    if not config.get("adb_server_port"):
        config["adb_server_port"] = getattr(args, "emuinput_adb_server_port", 5037)
    if not config.get("adb_serial"):
        config["adb_serial"] = getattr(args, "screenshot_serial", "") or getattr(args, "emuinput_serial", "")

    threshold = float(config.get("threshold", args.threshold))
    capture_path = str(config.get("capture_path", args.capture_path))
    templates = config.get("templates", {})
    if not isinstance(templates, dict):
        raise ValueError("anomaly config requires templates object")

    shot = await capture_frame(config, capture_path)
    scores: dict[str, dict[str, Any]] = {}
    for name, template_path in templates.items():
        ok, score = match_template_path(shot, str(template_path), threshold)
        scores[str(name)] = {"matched": ok, "score": round(float(score), 4)}

    print(
        json.dumps(
            {
                "mode": "anomaly-scan",
                "capture_path": capture_path,
                "adb_serial": config.get("adb_serial", ""),
                "threshold": threshold,
                "scores": scores,
            },
            indent=2,
        ),
    )


def action_key_from_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def payload_from_action_key(key: str) -> dict[str, Any]:
    out = json.loads(key)
    if not isinstance(out, dict):
        raise ValueError("Action key did not decode to a dict payload")
    return out


def state_bucket(obs: dict[str, Any]) -> str:
    zone = str(obs.get("zone", "unknown"))
    nearby = int(obs.get("nearby", 0))
    health = int(obs.get("health", 100))
    health_bucket = "low" if health < 30 else "mid" if health < 70 else "high"
    crowd_bucket = "crowded" if nearby >= 5 else "sparse"
    return f"{zone}|{crowd_bucket}|{health_bucket}"


def pick_random_action(rng: random.Random, bot_id: int, tick: int) -> dict[str, Any]:
    roll = rng.random()
    if roll < 0.7:
        return {
            "cmd": "move",
            "bot": bot_id,
            "direction": rng.choice(DIRECTIONS),
            "steps": rng.randint(1, 3),
            "tick": tick,
        }
    if roll < 0.9:
        return {
            "cmd": "interact",
            "bot": bot_id,
            "target": f"poi_{rng.randint(1, 8)}",
            "tick": tick,
        }
    return {
        "cmd": "say",
        "bot": bot_id,
        "message": f"ambient_{rng.randint(1000, 9999)}",
        "tick": tick,
    }


async def run_script(controller: Controller, commands: Iterable[dict[str, Any]], tick_delay: float, metrics: Metrics) -> None:
    for item in commands:
        await controller.send(item)
        metrics.mark_sent()
        await asyncio.sleep(tick_delay)


async def run_random_bot(controller: Controller, bot_id: int, ticks: int, tick_delay: float, rng: random.Random, metrics: Metrics) -> None:
    for tick in range(ticks):
        payload = pick_random_action(rng, bot_id, tick)
        await controller.send(payload)
        metrics.mark_sent()
        await asyncio.sleep(tick_delay)


async def run_policy_bot(
    controller: Controller,
    bot_id: int,
    ticks: int,
    tick_delay: float,
    rng: random.Random,
    metrics: Metrics,
    policy: dict[str, dict[str, Any]],
    default_state: str,
    epsilon: float,
) -> None:
    for tick in range(ticks):
        explore = rng.random() < epsilon
        bucket = default_state
        policy_entry = policy.get(bucket)
        if explore or not policy_entry or not policy_entry.get("actions"):
            payload = pick_random_action(rng, bot_id, tick)
        else:
            actions = policy_entry["actions"]
            weights = policy_entry.get("weights", [1] * len(actions))
            key = rng.choices(actions, weights=weights, k=1)[0]
            payload = payload_from_action_key(key)
            payload["bot"] = bot_id
            payload["tick"] = tick

        await controller.send(payload)
        metrics.mark_sent()
        await asyncio.sleep(tick_delay)


def learn_policy_from_jsonl(path: str) -> dict[str, dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            obs = record.get("obs", {})
            action = record.get("action", {})
            if not isinstance(obs, dict) or not isinstance(action, dict):
                continue
            bucket = state_bucket(obs)
            counts[bucket][action_key_from_payload(action)] += 1

    policy: dict[str, dict[str, Any]] = {}
    for bucket, action_counter in counts.items():
        ranked = action_counter.most_common()
        policy[bucket] = {
            "actions": [k for k, _ in ranked],
            "weights": [v for _, v in ranked],
            "samples": sum(action_counter.values()),
        }
    return policy


def resolve_controller(args: argparse.Namespace) -> Controller:
    transport = getattr(args, "transport", "print")
    if transport == "websocket":
        if not getattr(args, "endpoint", ""):
            raise ValueError("--endpoint is required for websocket transport")
        return WebSocketController(args.endpoint)
    if transport == "uinput":
        return UInputController(getattr(args, "uinput_profile", ""))
    if transport == "emuinput":
        return EmuInputController(
            emuinput_root=getattr(args, "emuinput_root", ""),
            serial=getattr(args, "emuinput_serial", "127.0.0.1:5555"),
            host_port=int(getattr(args, "emuinput_host_port", 27183)),
            adb_exe=getattr(args, "emuinput_adb_exe", "adb"),
            adb_server_port=int(getattr(args, "emuinput_adb_server_port", 5037)),
            bin_dir=getattr(args, "emuinput_bin_dir", ""),
            autofix=bool(getattr(args, "emuinput_autofix", True)),
        )
    return PrintController()


async def command_simulate(args: argparse.Namespace) -> None:
    controller = PrintController()
    metrics = Metrics()
    rng = random.Random(args.seed)
    await run_random_bot(controller, bot_id=0, ticks=args.ticks, tick_delay=args.tick_delay, rng=rng, metrics=metrics)
    print(json.dumps({"mode": "simulate", **metrics.snapshot()}))


async def command_script(args: argparse.Namespace) -> None:
    controller = resolve_controller(args)
    metrics = Metrics()
    script = load_script(args.file)
    try:
        await controller.start()
        await run_script(controller, script, tick_delay=args.tick_delay, metrics=metrics)
    finally:
        await controller.close()
    print(json.dumps({"mode": "script", **metrics.snapshot()}))


async def command_swarm(args: argparse.Namespace) -> None:
    controller = resolve_controller(args)
    metrics = Metrics()
    seed = args.seed if args.seed is not None else int(time.time())
    await controller.start()
    tasks = []
    for bot_id in range(args.bots):
        rng = random.Random(seed + bot_id)
        tasks.append(run_random_bot(controller, bot_id, args.ticks, args.tick_delay, rng, metrics))

    try:
        await asyncio.gather(*tasks)
    finally:
        await controller.close()

    print(json.dumps({"mode": "swarm", "bots": args.bots, "seed": seed, **metrics.snapshot()}))


async def command_play(args: argparse.Namespace) -> None:
    controller = resolve_controller(args)
    metrics = Metrics()
    policy = load_policy(args.policy)
    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    try:
        await controller.start()
        await run_policy_bot(
            controller,
            bot_id=args.bot,
            ticks=args.ticks,
            tick_delay=args.tick_delay,
            rng=rng,
            metrics=metrics,
            policy=policy,
            default_state=args.state,
            epsilon=args.epsilon,
        )
    finally:
        await controller.close()

    print(json.dumps({"mode": "play", "seed": seed, "state": args.state, "epsilon": args.epsilon, **metrics.snapshot()}))


async def command_anomaly(args: argparse.Namespace) -> None:
    controller = resolve_controller(args)
    metrics = Metrics()
    cycles: list[dict[str, Any]] = []

    try:
        await controller.start()
        for idx in range(args.cycles):
            result = await run_anomaly_cycle(controller, args, metrics)
            cycles.append({"cycle": idx + 1, **result})
            if not result.get("ok"):
                await asyncio.sleep(args.retry_delay)
    finally:
        await controller.close()

    success = sum(1 for item in cycles if item.get("ok"))
    print(json.dumps({"mode": "anomaly", "cycles": args.cycles, "successful_cycles": success, **metrics.snapshot(), "results": cycles}, indent=2))


def command_learn(args: argparse.Namespace) -> None:
    policy = learn_policy_from_jsonl(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2, sort_keys=True)
    print(json.dumps({"mode": "learn", "states": len(policy), "output": args.output}))


async def command_doctor(args: argparse.Namespace) -> None:
    checks: list[dict[str, Any]] = []

    try:
        import websockets  # noqa: F401

        checks.append({"check": "python_dependency:websockets", "ok": True})
    except ImportError:
        checks.append({"check": "python_dependency:websockets", "ok": False, "detail": "Install with: pip install websockets"})

    if args.endpoint:
        ok, detail = await probe_websocket(args.endpoint, args.timeout)
        checks.append({"check": f"websocket_endpoint:{args.endpoint}", "ok": ok, "detail": detail})

    if args.check_uinput:
        ok, detail = probe_uinput()
        checks.append({"check": "uinput_device", "ok": ok, "detail": detail})

    if args.check_emuinput:
        ok, detail = probe_emuinput_root(args.emuinput_root)
        checks.append({"check": "emuinput_root", "ok": ok, "detail": detail})

    if args.check_vision:
        ok, detail = probe_vision()
        checks.append({"check": "pixel_recognition", "ok": ok, "detail": detail})

    if args.telemetry:
        try:
            policy = learn_policy_from_jsonl(args.telemetry)
            checks.append({"check": f"telemetry_parse:{args.telemetry}", "ok": True, "states": len(policy)})
        except Exception as exc:  # noqa: BLE001
            checks.append({"check": f"telemetry_parse:{args.telemetry}", "ok": False, "detail": str(exc)})

    all_ok = all(item.get("ok", False) for item in checks)
    print(json.dumps({"mode": "doctor", "ok": all_ok, "checks": checks}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Game automation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="Dry-run one random bot")
    p_sim.add_argument("--ticks", type=int, default=100)
    p_sim.add_argument("--tick-delay", type=float, default=0.05)
    p_sim.add_argument("--seed", type=int, default=1234)
    p_sim.set_defaults(func=command_simulate)

    p_script = sub.add_parser("script", help="Run scripted command file")
    p_script.add_argument("--file", required=True)
    p_script.add_argument("--tick-delay", type=float, default=0.05)
    p_script.add_argument("--transport", choices=["print", "websocket", "uinput", "emuinput"], default="print")
    p_script.add_argument("--endpoint", default="", help="ws://host:port/path (used by websocket transport)")
    p_script.add_argument("--uinput-profile", default="", help="JSON mapping action names to uinput command templates")
    p_script.add_argument("--emuinput-root", default="", help="Path to emuinput project root")
    p_script.add_argument("--emuinput-serial", default="127.0.0.1:5555")
    p_script.add_argument("--emuinput-host-port", type=int, default=27183)
    p_script.add_argument("--emuinput-adb-exe", default="adb")
    p_script.add_argument("--emuinput-adb-server-port", type=int, default=5037)
    p_script.add_argument("--emuinput-bin-dir", default="")
    p_script.add_argument("--emuinput-autofix", dest="emuinput_autofix", action="store_true", default=True)
    p_script.add_argument("--no-emuinput-autofix", dest="emuinput_autofix", action="store_false")
    p_script.set_defaults(func=command_script)

    p_swarm = sub.add_parser("swarm", help="Run multiple bots")
    p_swarm.add_argument("--transport", choices=["print", "websocket", "uinput", "emuinput"], default="print")
    p_swarm.add_argument("--endpoint", default="", help="ws://host:port/path (used by websocket transport)")
    p_swarm.add_argument("--uinput-profile", default="", help="JSON mapping action names to uinput command templates")
    p_swarm.add_argument("--emuinput-root", default="", help="Path to emuinput project root")
    p_swarm.add_argument("--emuinput-serial", default="127.0.0.1:5555")
    p_swarm.add_argument("--emuinput-host-port", type=int, default=27183)
    p_swarm.add_argument("--emuinput-adb-exe", default="adb")
    p_swarm.add_argument("--emuinput-adb-server-port", type=int, default=5037)
    p_swarm.add_argument("--emuinput-bin-dir", default="")
    p_swarm.add_argument("--emuinput-autofix", dest="emuinput_autofix", action="store_true", default=True)
    p_swarm.add_argument("--no-emuinput-autofix", dest="emuinput_autofix", action="store_false")
    p_swarm.add_argument("--bots", type=int, default=5)
    p_swarm.add_argument("--ticks", type=int, default=100)
    p_swarm.add_argument("--tick-delay", type=float, default=0.05)
    p_swarm.add_argument("--seed", type=int)
    p_swarm.set_defaults(func=command_swarm)

    p_learn = sub.add_parser("learn", help="Learn a simple policy from JSONL telemetry")
    p_learn.add_argument("--input", required=True, help="JSONL file with {'obs': {...}, 'action': {...}}")
    p_learn.add_argument("--output", default="learned_policy.json")
    p_learn.set_defaults(func=command_learn)

    p_play = sub.add_parser("play", help="Run one policy-driven bot")
    p_play.add_argument("--policy", required=True, help="policy JSON from the learn command")
    p_play.add_argument("--state", default="unknown|sparse|high", help="default state bucket")
    p_play.add_argument("--bot", type=int, default=0)
    p_play.add_argument("--ticks", type=int, default=100)
    p_play.add_argument("--tick-delay", type=float, default=0.05)
    p_play.add_argument("--epsilon", type=float, default=0.15, help="exploration rate (0.0-1.0)")
    p_play.add_argument("--transport", choices=["print", "websocket", "uinput", "emuinput"], default="print")
    p_play.add_argument("--endpoint", default="", help="ws://host:port/path (used by websocket transport)")
    p_play.add_argument("--uinput-profile", default="", help="JSON mapping action names to uinput command templates")
    p_play.add_argument("--emuinput-root", default="", help="Path to emuinput project root")
    p_play.add_argument("--emuinput-serial", default="127.0.0.1:5555")
    p_play.add_argument("--emuinput-host-port", type=int, default=27183)
    p_play.add_argument("--emuinput-adb-exe", default="adb")
    p_play.add_argument("--emuinput-adb-server-port", type=int, default=5037)
    p_play.add_argument("--emuinput-bin-dir", default="")
    p_play.add_argument("--emuinput-autofix", dest="emuinput_autofix", action="store_true", default=True)
    p_play.add_argument("--no-emuinput-autofix", dest="emuinput_autofix", action="store_false")
    p_play.add_argument("--seed", type=int)
    p_play.set_defaults(func=command_play)

    p_anomaly = sub.add_parser("anomaly", help="Run anomaly loop using template-based pixel recognition")
    p_anomaly.add_argument("--config", required=True, help="JSON config with screenshot command, templates, and tap coordinates")
    p_anomaly.add_argument("--cycles", type=int, default=1, help="How many anomaly runs to attempt")
    p_anomaly.add_argument("--transport", choices=["print", "websocket", "uinput", "emuinput"], default="print")
    p_anomaly.add_argument("--endpoint", default="", help="ws://host:port/path (used by websocket transport)")
    p_anomaly.add_argument("--uinput-profile", default="", help="JSON mapping action names to uinput command templates")
    p_anomaly.add_argument("--emuinput-root", default="", help="Path to emuinput project root")
    p_anomaly.add_argument("--emuinput-serial", default="127.0.0.1:5555")
    p_anomaly.add_argument("--emuinput-host-port", type=int, default=27183)
    p_anomaly.add_argument("--emuinput-adb-exe", default="adb")
    p_anomaly.add_argument("--emuinput-adb-server-port", type=int, default=5037)
    p_anomaly.add_argument("--emuinput-bin-dir", default="")
    p_anomaly.add_argument("--emuinput-autofix", dest="emuinput_autofix", action="store_true", default=True)
    p_anomaly.add_argument("--no-emuinput-autofix", dest="emuinput_autofix", action="store_false")
    p_anomaly.add_argument("--screenshot-serial", default="", help="ADB serial used specifically for screenshot capture")
    p_anomaly.add_argument("--threshold", type=float, default=0.86, help="Template match threshold")
    p_anomaly.add_argument("--poll-seconds", type=float, default=0.5)
    p_anomaly.add_argument("--combat-poll-seconds", type=float, default=0.9)
    p_anomaly.add_argument("--tap-delay", type=float, default=0.35)
    p_anomaly.add_argument("--short-timeout", type=float, default=8.0)
    p_anomaly.add_argument("--arrival-timeout", type=float, default=60.0)
    p_anomaly.add_argument("--stable-seconds", type=float, default=7.0)
    p_anomaly.add_argument("--retry-delay", type=float, default=1.0)
    p_anomaly.add_argument("--capture-path", default="/tmp/anomaly_frame.png")
    p_anomaly.set_defaults(func=command_anomaly)

    p_anomaly_scan = sub.add_parser("anomaly-scan", help="Capture one emulator frame and report template match scores")
    p_anomaly_scan.add_argument("--config", required=True, help="JSON config with templates and screenshot settings")
    p_anomaly_scan.add_argument("--emuinput-serial", default="127.0.0.1:5555")
    p_anomaly_scan.add_argument("--emuinput-adb-exe", default="adb")
    p_anomaly_scan.add_argument("--emuinput-adb-server-port", type=int, default=5037)
    p_anomaly_scan.add_argument("--screenshot-serial", default="", help="ADB serial used specifically for screenshot capture")
    p_anomaly_scan.add_argument("--threshold", type=float, default=0.86)
    p_anomaly_scan.add_argument("--capture-path", default="/tmp/anomaly_frame.png")
    p_anomaly_scan.set_defaults(func=command_anomaly_scan)

    p_psdiag = sub.add_parser("powershell-diagnostics", help="Run a unified PowerShell diagnostics battery for emulator and emuinput readiness")
    p_psdiag.add_argument("--serial", default="127.0.0.1:5555", help="Target emulator serial")
    p_psdiag.add_argument("--adb-exe", default="adb", help="Preferred adb executable name/path")
    p_psdiag.add_argument("--emuinput-root", default="", help="Path to emuinput project root")
    p_psdiag.add_argument("--emuinput-bin-dir", default="", help="Path to emuinput bin/android directory")
    p_psdiag.add_argument("--timeout", type=float, default=25.0, help="Timeout per PowerShell diagnostic command")
    p_psdiag.add_argument("--output", default="", help="Optional JSON file to write diagnostics results")
    p_psdiag.add_argument("--print-only", action="store_true", help="Only print the command battery; do not execute PowerShell")
    p_psdiag.set_defaults(func=command_powershell_diagnostics)

    p_doctor = sub.add_parser("doctor", help="Pre-flight checks for live environment readiness")
    p_doctor.add_argument("--endpoint", default="", help="ws://host:port/path endpoint to probe")
    p_doctor.add_argument("--check-uinput", action="store_true", help="Check for /dev/uinput availability")
    p_doctor.add_argument("--check-emuinput", action="store_true", help="Check emuinput project path")
    p_doctor.add_argument("--check-vision", action="store_true", help="Check OpenCV dependency for anomaly mode")
    p_doctor.add_argument("--emuinput-root", default="", help="Path to emuinput project root")
    p_doctor.add_argument("--telemetry", default="", help="Optional telemetry JSONL to validate")
    p_doctor.add_argument("--timeout", type=float, default=3.0, help="Endpoint probe timeout in seconds")
    p_doctor.set_defaults(func=command_doctor)

    return parser


async def main_async() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    if asyncio.iscoroutine(result):
        await result


if __name__ == "__main__":
    asyncio.run(main_async())
