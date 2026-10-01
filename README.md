# ADELTE AI v1 — v2.1.0

Single-file server: `adelte.py` + UI `adelte.html`.

## Run

```bash
pip install -r requirements.txt
python adelte.py
# open http://localhost:8000
```

Options: `python adelte.py --port 9000`, `--no-ai`, `--db /tmp/x.db`.

## Keys (already wired)

All APIs from `new_api` are already imported into the local `.env`
(which is git-ignored and never pushed):

- Groq, Gemini, OpenRouter
- OpenAI pool (`OPENAI_API_KEY` + `OPENAI_API_KEY_POOL`, auto-rotates on 401/429)
- Together, Cerebras, Mistral (chain slots ready)
- Clarifai PAT, Google Custom Search, Telegram bot token
- Spice.ai key, Databricks token, Azure SQL connection string

Check live status: `GET /api/integrations/status` (masked, no secrets leak).
Prove they WORK: `GET /api/integrations/test` actually pings each provider.

## Cargolis Desktop — Siri outside the browser

`cargolis.py` is a system-wide voice companion. Tray icon, global hotkey
`Ctrl+Alt+A`, talks back aloud, screenshots your screen, searches the web —
no browser needed.

```bash
pip install -r requirements-desktop.txt   # mic, voice, hotkey, tray, eyes
python adelte.py                          # terminal 1: the brain
python cargolis.py                        # terminal 2: the voice
```

Works with zero extras too (typed console). Point at a hosted brain with
`ADELTE_URL=https://your-service.onrender.com python cargolis.py` — note PC
commands (`lock`, volume…) then act on the server, not your PC, so keep the
local server for full Siri-style control of your own machine.

## Models

- `adelte-search` — 16 engines (incl. Google CSE), reads pages, answers with sources
- `adelte-coder-3high` — strongest coding chain: OpenAI → Groq → OpenRouter → Gemini → Cerebras → Mistral → Together → Clarifai → Horde
- `adelte-minimax` — deepest reasoning chain
- `adelte-commander` — local system commands only
- `adelte-cargolis` — desktop/screenshot assistant

## Push to GitHub

```bash
git remote add origin https://github.com/ADELTE-Boncoeur/adelte_ai_v1.git
git push -u origin main
```

`.env` and `new_api` are ignored by git and will not upload.
Rotate any key you pasted in chat, since chat history is not a safe store.
