# Taxvasion Game Automation Harness

This repository includes a generic automation harness you can use to:

- script deterministic play sessions for testing,
- run random-bot traffic to stress movement/interaction systems,
- generate repeatable world population behavior for smoke testing,
- and learn a lightweight behavior policy from gameplay telemetry.
- run a deterministic anomaly-farming loop using pixel/template recognition.

## What you get

- `automation_harness.py` — single-file Python runner with:
  - pluggable game controller interfaces,
  - scripted command playback,
  - random world-population bot loops,
  - metrics logging (counts + timestamps),
  - optional websocket transport for game servers,
  - host-side **uinput transport** for emulator input injection,
  - template/pixel recognition helpers for UI state detection,
  - telemetry-driven policy learning (`learn`) and policy playback (`play`).

## Quick start

### 1) Dry-run without a game server

```bash
python3 automation_harness.py simulate --ticks 50 --seed 42
```

### 2) Scripted test run

```bash
python3 automation_harness.py script --file sample_script.json
```

### 3) Random population bot swarm

```bash
python3 automation_harness.py swarm --bots 10 --ticks 200
```

### 4) Learn from telemetry, then replay policy

```bash
python3 automation_harness.py learn --input sample_telemetry.jsonl --output learned_policy.json
python3 automation_harness.py play --policy learned_policy.json --ticks 20 --state 'town|sparse|high'
```

### 5) Run against emulator using emuinput (recommended)

```bash
python3 automation_harness.py script \
  --file sample_script.json \
  --transport emuinput \
  --emuinput-root /path/to/emuinput \
  --emuinput-bin-dir /path/to/emuinput/bin/android \
  --emuinput-adb-server-port 5037
```

`emuinput` is a host-side controller that starts a small on-device daemon (`uinputd`) and injects real input events via Android's input stack.

What we now reuse from your `prototype.py`/emuinput backend:
- daemon lifecycle via `ensure_daemon()`
- startup warmup: every `script`/`swarm`/`play` run explicitly re-confirms `uinputd` before sending actions
- auto-recovery: if a send fails because daemon/socket dropped, harness re-warms and retries once
- adb candidate fallback + optional adb autofix sequence (`kill-server` -> `start-server` -> `connect` -> `get-state`) during backend init
- native uinput socket injection (`tap`, `drag`, `type_text`, `key_tap`)
- keyboard keycode conventions for movement/interaction (W/A/S/D + F)

You can disable autofix retries if needed:

```bash
python3 automation_harness.py script \
  --file sample_script.json \
  --transport emuinput \
  --emuinput-root /path/to/emuinput \
  --no-emuinput-autofix
```

### 6) Launch beginner GUI (run + test recorder)

```bash
python3 automation_gui.py
```

GUI features:
- run any harness mode without memorizing CLI flags,
- choose transport (`print` / `websocket` / `uinput` / `emuinput`),
- record key presses and mouse drags/taps into script JSON for testing,
- save recorded actions and execute them in `script` mode.

## How learning works

The learning mode is intentionally simple and fast:

1. You provide JSONL records with `obs` and `action` keys.
2. The harness buckets each observation into a coarse state key (`zone|crowd|health`).
3. It counts action frequency per state.
4. At runtime, `play` samples actions by those learned frequencies.

## Telemetry format (`sample_telemetry.jsonl`)

```json
{"obs":{"zone":"town","nearby":1,"health":92},"action":{"cmd":"move","direction":"north","steps":1}}
{"obs":{"zone":"town","nearby":1,"health":88},"action":{"cmd":"interact","target":"npc_vendor"}}
{"obs":{"zone":"field","nearby":6,"health":54},"action":{"cmd":"move","direction":"west","steps":2}}
```

## Script format (`sample_script.json`)

```json
[
  {"cmd": "move", "direction": "north", "steps": 3},
  {"cmd": "move", "direction": "east", "steps": 2},
  {"cmd": "interact", "target": "npc_vendor"},
  {"cmd": "wait", "ticks": 5},
  {"cmd": "say", "message": "hello world"}
]
```

## UInput action mapping

`uinput` transport maps each action to a command template from JSON profile.

- `move` actions use directional keys: `move_north`, `move_south`, `move_east`, `move_west`
- other commands use exact keys: `interact`, `say`, `tap`, `swipe`, `text`
- templates can be either a shell string or token list and support `{payload_key}` substitutions

Example profile file: `sample_uinput_profile.json`.

## Beginner first-day to-do list (copy/paste step-by-step)

Use this exact order when you sit down at your PC. Do not skip ahead.

### Step 1: Fill in your values once

Replace these placeholders in the commands below:

- `YOUR_SERIAL` (example: `127.0.0.1:7555`)
- `YOUR_EMUINPUT_ROOT` (example: `C:\\tools\\emuinput`)
- `YOUR_EMUINPUT_BIN` (example: `C:\\tools\\emuinput\\bin\\android`)

### Step 2: Confirm commands are available

```bash
python3 automation_harness.py --help
python3 automation_harness.py doctor --help
python3 automation_harness.py powershell-diagnostics --help
python3 automation_harness.py anomaly-scan --help
python3 automation_harness.py anomaly --help
```

### Step 3: Run Windows diagnostics (best first check)

```bash
# Preview what will run
python3 automation_harness.py powershell-diagnostics \
  --serial YOUR_SERIAL \
  --emuinput-root "YOUR_EMUINPUT_ROOT" \
  --emuinput-bin-dir "YOUR_EMUINPUT_BIN" \
  --print-only

# Run and save report
python3 automation_harness.py powershell-diagnostics \
  --serial YOUR_SERIAL \
  --emuinput-root "YOUR_EMUINPUT_ROOT" \
  --emuinput-bin-dir "YOUR_EMUINPUT_BIN" \
  --output powershell_diagnostics.json
```

### Step 4: Run preflight checks

```bash
python3 automation_harness.py doctor --check-emuinput --emuinput-root "YOUR_EMUINPUT_ROOT"
python3 automation_harness.py doctor --check-vision
python3 automation_harness.py doctor --check-uinput --telemetry sample_telemetry.jsonl
```

### Step 5: Verify button/template detection

```bash
python3 automation_harness.py anomaly-scan \
  --config sample_anomaly_config.json \
  --screenshot-serial YOUR_SERIAL
```

If scores are low, fix template images before running full anomaly mode.

### Step 6: Run one safe anomaly cycle

```bash
python3 automation_harness.py anomaly \
  --config sample_anomaly_config.json \
  --transport emuinput \
  --emuinput-root "YOUR_EMUINPUT_ROOT" \
  --emuinput-bin-dir "YOUR_EMUINPUT_BIN" \
  --screenshot-serial YOUR_SERIAL \
  --cycles 1
```

### Step 7: If it fails, debug in this order

1. Re-run `powershell-diagnostics` and inspect `powershell_diagnostics.json`.
2. Re-run `anomaly-scan` and confirm template scores.
3. Re-check `sample_anomaly_config.json` tap coordinates and template file paths.
4. Re-run with `--cycles 1` until stable; only then increase cycles.

### Step 8: Optional beginner GUI

```bash
python3 automation_gui.py
```

## Live readiness checklist

Before running live, verify:

1. command templates in profile actually trigger expected emulator inputs,
2. `/dev/uinput` is available and writable in your host runtime,
3. your anomaly-complete detector waits for a stability window (avoid leaving between waves),
4. you have bot caps/rate limits and a kill switch.

Preflight command:

```bash
python3 automation_harness.py doctor --check-uinput --telemetry sample_telemetry.jsonl

# include pixel recognition dependency check
python3 automation_harness.py doctor --check-vision
```

### Unified PowerShell diagnostics (Windows / MuMu)

When you are unsure what is missing on your Windows host, run this one battery. It inventories emuinput paths, checks adb candidates, validates serial/device state, and tests emulator screencap in one report.

```bash
# print the planned PowerShell commands only
python3 automation_harness.py powershell-diagnostics \
  --serial 127.0.0.1:7555 \
  --emuinput-root C:\path\to\emuinput \
  --emuinput-bin-dir C:\path\to\emuinput\bin\android \
  --print-only

# execute diagnostics and save JSON output
python3 automation_harness.py powershell-diagnostics \
  --serial 127.0.0.1:7555 \
  --emuinput-root C:\path\to\emuinput \
  --emuinput-bin-dir C:\path\to\emuinput\bin\android \
  --output powershell_diagnostics.json
```


## Emuinput integration (beginner-friendly)

Your automation has two parts:

1. **Decision logic (this repo)** — decides *what to do next* (open overview, warp, lock all, etc.).
2. **Input backend (emuinput)** — performs *the actual taps/keys* in the emulator.

In practice, `--transport emuinput` uses classes from the emuinput project (`emuinput.adb.Adb`, `emuinput.controller.EmuController`) and calls methods like `tap`, `drag`, and `type_text`.

You can verify the emuinput path with:

```bash
python3 automation_harness.py doctor --check-emuinput --emuinput-root /path/to/emuinput
```

If you prefer command templates instead of the emuinput Python API, `--transport uinput` with `sample_uinput_profile.json` still works.

## Pixel recognition anomaly mode (what you asked for)

For your loop (open overview -> select anomaly -> warp -> lock all -> select NPC -> focus fire), use the new `anomaly` mode.

It uses template matching on screenshots, so it can find UI elements by image similarity instead of hard-coding every state transition.

### Requirements

- `opencv-python` for template matching.
- A screenshot command in config (`screenshot_cmd`) that saves a frame to `{output}`.
- Template images cropped from your recording for each state cue.

Install OpenCV:

```bash
python3 -m pip install opencv-python
```

### Config file

Use `sample_anomaly_config.json` as the starting point. It includes:

- `tap_targets`: where to tap for each required action.
- `templates`: image paths used to detect UI state.
- `adb_serial`: emulator serial to capture from (important for multi-emulator).
- Optional `screenshot_cmd`: override capture command if you need a custom MuMu pipeline.

By default, anomaly mode now captures screenshots directly from the targeted emulator with:

`adb -P <port> -s <serial> exec-out screencap -p`

so it does **not** depend on desktop/window focus and can run multiple emulators in parallel.

### Run one cycle

```bash
python3 automation_harness.py anomaly \
  --config sample_anomaly_config.json \
  --transport emuinput \
  --emuinput-root /path/to/emuinput \
  --screenshot-serial 127.0.0.1:7555 \
  --cycles 1
```

### Verify button detection quickly

Use one-frame scan to report template match scores for all configured buttons:

```bash
python3 automation_harness.py anomaly-scan \
  --config sample_anomaly_config.json \
  --screenshot-serial 127.0.0.1:7555
```

This helps validate whether overview/warp/target templates are currently being recognized.

### Built-in behavior

- If warp/arrival is not confirmed within 60 seconds, it marks that cycle failed and moves on.
- Site clear is only accepted after 7 seconds of:
  - no lockable targets template,
  - no NPCs-in-overview template.

This matches your rule for moving to the next anomaly.
