# PalmPay — Agent Preferences (standing rules)

## Python / packages
- NEVER install packages globally or with system python (`python3` = 3.14, no deps).
- ALWAYS use a virtual environment: `.venv/` at project root (already gitignored).
  - Interpreter: Python 3.11 via `mise exec python@3.11 -- python -V` (system has no 3.11; mediapipe needs ≤3.12).
  - Create if missing: `mise exec python@3.11 -- python -m venv .venv`
  - Install deps ONLY inside it: `.venv/bin/pip install -r requirements.txt`
- Run everything through `.venv/bin/python` (tests, scripts, server).

## Node
- Use version-manager-managed node only — never install node packages/system-wide.
- This machine manages node via **mise** (`node -v` shim works; currently v26.x). User calls this "nvm" generically — same intent: version manager, no global pollution.
- `node --check file.js` for JS syntax verification.

## Server lifecycle (bash tool quirk)
- Kill may hang past timeout: `kill -9 $(pgrep -f "...app.py")`, ignore timeout error.
- Relaunch detached: `setsid .venv/bin/python app.py > /tmp/opencode/palmpay_server.log 2>&1 < /dev/null & disown; exit 0`

## Project discipline
- Tests must stay green after every change: `ACTIVE_MODEL="classical" .venv/bin/python -m unittest discover tests` (113 tests; 4 C++ tests via `ctest --test-dir cpp/build`).
- Never store raw palm images; thresholds calibrated empirically, never guessed (see PROJECT_CONTEXT.md §6).
