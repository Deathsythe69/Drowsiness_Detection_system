# Project Rules & Conventions
## Drowsiness & Attention Detection System

These rules apply to all contributors (human or AI-assisted, e.g., Claude Code/Cursor) working on this codebase.

---

## 1. Code Style

- **Python version:** 3.10+ features are allowed (match statements, `|` union types).
- **Formatter:** `black` (line length 100) — run before every commit.
- **Linter:** `ruff` — must pass with zero errors before merging.
- **Type hints:** required on all public function signatures. Use `mypy` in CI if configured.
- **Docstrings:** Google-style docstrings on every module, class, and non-trivial function.
- **Naming:**
  - `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants (e.g., `EAR_THRESHOLD`).
  - Files/modules named after their single responsibility (`features.py`, not `utils.py` grab-bags).

## 2. Architecture Boundaries

- **`core/`** modules must never import from `ui/` — keeps detection logic UI-agnostic and independently testable.
- **`storage/`** is the only layer allowed to touch the database directly; `core/` and `ui/` interact with data via `storage/models.py` objects, never raw SQL.
- No hardcoded thresholds in `core/` logic — all tunables must be read from `config.yaml` via a single `Config` loader class.
- Frame buffers/webcam handles must be owned exclusively by `core/capture.py`; no other module opens `cv2.VideoCapture` directly.
- **Driver-Only Alerting:** Drowsiness and fatigue alerts must evaluate only the primary driver face landmarks inside the active ROI. Secondary faces (passengers) must be tagged `is_primary=False` and excluded from alert evaluation.
- **Vehicle Motion Gating:** Auditory alerts must be suppressed when the vehicle is stationary/parked if `require_motion_for_alert` is enabled.

## 3. Threading & Performance Rules

- Any OpenCV/MediaPipe processing must run off the Qt main thread. Never call blocking CV operations from a UI event handler.
- UI updates from worker threads must go through Qt signals (`pyqtSignal`) — never mutate widgets directly from a non-main thread.
- Target: sustain ≥15 FPS on a mid-range CPU. Profile with `cProfile` before merging any change that touches the per-frame processing path (`features.py`, `classifier.py`).

## 4. Privacy Rules (Non-Negotiable)

- **No raw video frames or images are persisted to disk or database**, ever, by default. Only derived numeric features (EAR, MAR, timestamps, event types) may be logged.
- Any future feature involving screenshots, video recording, or cloud sync must be **opt-in**, clearly disclosed in the UI, and gated behind an explicit consent dialog before implementation begins.
- No telemetry or network calls of any kind in the core detection pipeline without separate, explicit approval.

## 5. Testing Rules

- Every new function in `core/features.py` and `core/classifier.py` requires a corresponding `pytest` unit test before merge.
- No PR/commit that changes threshold math (`EAR`/`MAR` formulas, FSM transition logic) may be merged without an accompanying test demonstrating the new behavior.
- Manual test checklist (lighting, glasses, distance — see Design.md §8) must be re-run before any tagged release.

## 6. Git & Commit Conventions

- Branch naming: `feature/<short-desc>`, `fix/<short-desc>`, `chore/<short-desc>`.
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- No direct commits to `main`; all changes via PR, even for solo development — keeps a clean history and enables easy rollback.
- Squash-merge PRs with a descriptive summary.

## 7. AI-Assisted Development Rules

(Relevant given Claude Code / Cursor / Gemini extension usage in this workflow.)

- AI-generated code must be reviewed line-by-line before commit — no blind acceptance of suggestions, especially in `core/classifier.py` where subtle threshold bugs directly affect false-positive/negative rates.
- Any AI-suggested dependency addition must be checked against `requirements.txt` for version conflicts and license compatibility before inclusion.
- Prompts used to generate significant chunks of logic (e.g., FSM design, EAR formula derivation) should be noted in the PR description for traceability.

## 8. Documentation Rules (Continuous Synchronization)

- **Mandatory Invariant:** After each and every chat/task session, contributors and AI assistants MUST review, improve, and synchronize `memory.md` and all related documentation in `md_files/` with the latest code changes, decisions, and system state.
- Any change to thresholds, formulas, or the FSM must be reflected back into `Design.md` — docs and code must never drift apart.
- `config.yaml` keys must always match what's documented in `Design.md` and `README.md`; no undocumented config options.
- `README.md` must always include an up-to-date "Quick Start" (install → run → calibrate) section and accurate module listings.
- `memory.md` serves as the living technical journal, recording ADRs, milestone completions, configuration schema, and architectural memory across conversations.

## 9. Remote Admin Panel Security Rules

- **PIN Authentication Required:** The remote admin web panel MUST always require PIN authentication before granting access to any endpoint. Unauthenticated requests to protected endpoints MUST return HTTP 401.
- **Audit Logging Mandatory:** ALL remote buzzer override actions (mute/unmute) MUST be logged to SQLite with event type `admin_remote_override` or `admin_remote_unmute` and the remote client IP address in the metadata field.
- **LAN & Wi-Fi Scope:** The Flask web server binds to `0.0.0.0` for local Wi-Fi, Ethernet, and Mobile Hotspot accessibility. It MUST NOT be exposed to the public internet without additional transport security (e.g. reverse proxy TLS or VPN).
- **Dynamic QR Code:** The `/qr` endpoint and `WiFiAccessDialog` MUST encode the full reachable Wi-Fi/LAN URL without exposing sensitive server parameters.
- **No Sensitive Data Exposure:** The `/status` and `/log` endpoints MUST NOT expose raw video frames, file system paths, or database connection strings. Only derived numeric metrics and event summaries are permitted.
- **Session Security:** Server-side session cookies with `os.urandom(24)` secret keys. Sessions are cleared on logout and do not persist across Flask restarts.
- **Werkzeug Logging Suppressed:** The remote admin server suppresses `werkzeug` request logging to avoid polluting the main application console.

## 10. Release Checklist

1. All tests passing (`pytest`).
2. `black` + `ruff` clean.
3. Manual test matrix re-verified.
4. `config.yaml` defaults sane for a first-time user (no debug values left in).
5. PyInstaller build tested on at least one clean machine/VM before distribution.
6. Version bumped in `main.py` / about screen.
