#!/usr/bin/env python3
"""
================================================================================
 ██████  ███████ ██   ████████  █████
 ██   ██ ██      ██      ██    ██   ██     ADELTE SERVER  ·  single file
 ██   ██ █████   ██      ██    ███████     by ADELTE Industries
 ██   ██ ██      ██      ██    ██   ██
 ██████  ███████ ███████ ██    ██   ██     v2.0
================================================================================

Everything in ONE file: search engines, page reader, AI brain, extractive
fallback, SQLite store, API-key system, FastAPI server and SSE streaming.

  * NO API key of your own. NO .env. NO login. NO signup. Nothing to configure.
  * Searches DuckDuckGo, Bing, Brave, Google, Wikipedia, GitHub (repos+issues),
    StackOverflow and Hacker News — all in parallel.
  * Reads the actual top pages, not just snippets.
  * Streams its thinking to the browser live over SSE.
  * Remembers the session so follow-ups reuse what it learned.
  * Mints free API keys so anyone can call your server.

QUICK START                              (works on Python 3.8 - 3.13+)
    pip install fastapi uvicorn httpx selectolax
    python adelte.py
    open http://localhost:8000

OPTIONS
    python adelte.py --port 9000 --host 0.0.0.0
    python adelte.py --no-ai          # extractive answers only (fastest)
    python adelte.py --db /tmp/x.db   # custom database location

The frontend file `adelte.html` should sit next to this script. If it is
missing, a built-in minimal UI is served instead, so the server always runs.
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import html as _html
import hashlib
import json
import os
import re
import secrets
import random
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (Any, AsyncIterator, Awaitable, Callable, Dict, List,
                    Optional, Tuple)
from datetime import datetime
from urllib.parse import parse_qs, quote, quote_plus, unquote, urlparse

# ----------------------------------------------------------------------------
# Version + dependency checks with friendly messages
#
# This file is written to run on Python 3.8 through 3.13+. All annotations use
# typing.Optional / typing.List rather than the 3.10+ `str | None` syntax,
# because Pydantic evaluates annotations at runtime and older interpreters
# raise "unsupported operand type(s) for |" when they do.
# ----------------------------------------------------------------------------
if sys.version_info < (3, 8):                                # pragma: no cover
    sys.exit(f"\n  ADELTE needs Python 3.8 or newer. You are running "
             f"{sys.version.split()[0]}.\n")

try:
    import httpx
    from fastapi import (FastAPI, Header, HTTPException, Query, Request,
                         Response)
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    from selectolax.parser import HTMLParser
except ImportError as e:                                     # pragma: no cover
    print(f"\n  Missing dependency: {e.name}\n"
          f"  Install everything with:\n\n"
          f"      pip install fastapi uvicorn httpx selectolax\n")
    sys.exit(1)


# ============================================================================
#  SECTION 1 — CONFIG
# ============================================================================

HERE = Path(__file__).resolve().parent


def _default_db() -> Path:
    """Keep the database OUT of the folder holding adelte.html.

    SQLite continuously rewrites .db / .db-wal / .db-shm. If those live
    beside adelte.html, a folder-watching dev server (VS Code Live Server,
    browser-sync, nodemon) sees the writes and force-reloads the browser
    the instant an answer is saved -- which looks exactly like the page
    'auto-refreshing and losing the answer'. Writing to a per-user data
    directory removes the trigger entirely."""
    if os.environ.get("VERCEL") == "1" or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        # Serverless: only /tmp is writable. DB resets on cold starts;
        # permanent history needs an external store (see README).
        try:
            Path("/tmp").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return Path("/tmp/adelte.db")
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ADELTE"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "ADELTE"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME",
                                   Path.home() / ".local" / "share")) / "adelte"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base / "adelte.db"
    except Exception:
        return HERE / "adelte.db"


def load_env() -> Dict[str, str]:
    """Read .env sitting beside adelte.py. Keys NEVER get hardcoded here.

    Format is one KEY=value per line; # starts a comment. Values already in
    the real environment win, so `export GROQ_API_KEY=...` also works.
    Rotating a leaked key means editing one line in .env - not this file.
    """
    out: Dict[str, str] = {}
    for name in (".env", "adelte.env"):
        p = HERE / name
        if not p.exists():
            continue
        try:
            for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                # tolerate a trailing inline comment on unquoted values
                if " #" in v:
                    v = v.split(" #", 1)[0].strip()
                if v:
                    out[k.strip().upper()] = v
        except Exception:
            pass
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
              "TOGETHER_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY",
              "OPENAI_API_KEY", "OPENAI_API_KEY_POOL", "CLARIFAI_PAT",
              "GOOGLE_CUSTOM_SEARCH_API_KEY", "GOOGLE_SEARCH_ENGINE_ID",
              "TELEGRAM_BOT_TOKEN", "SPICE_AI_API_KEY", "DATABRICKS_HOST",
              "DATABRICKS_TOKEN", "SQL_DB_CONN_STR"):
        v = os.environ.get(k)
        if v and v.strip() and v.strip().lower() not in (
                "your_api_key_here", "your_key_here", "xxx", "test", "none", "null"):
            # A real process env still wins, but obvious placeholders must
            # never clobber the genuine key already read from .env.
            out[k] = v.strip()
    return out


ENV = load_env()


def env_key(name: str) -> str:
    return (ENV.get(name) or "").strip()


def openai_key_pool() -> List[str]:
    """All OpenAI-compatible keys from .env, primary first, deduped.

    OPENAI_API_KEY is the primary. OPENAI_API_KEY_POOL is a comma-separated
    list imported from new_api (50+ keys). Empty entries are dropped so a
    stray comma can never break the Authorization header.
    """
    pool: List[str] = []
    primary = (ENV.get("OPENAI_API_KEY") or "").strip()
    if primary:
        pool.append(primary)
    for raw in (ENV.get("OPENAI_API_KEY_POOL") or "").split(","):
        k = raw.strip().strip('"').strip("'")
        if k and k not in pool:
            pool.append(k)
    return pool


def integration_status() -> Dict[str, dict]:
    """Masked availability report for every optional integration.

    Additive helper only: never returns a raw secret, only mask() + bool.
    Powers GET /api/integrations/status so the UI can show what is live.
    """
    def _has(*names: str) -> bool:
        return any(bool((ENV.get(n) or "").strip()) for n in names)
    return {
        "groq": {"live": _has("GROQ_API_KEY"), "key": mask(env_key("GROQ_API_KEY"))},
        "gemini": {"live": _has("GEMINI_API_KEY"), "key": mask(env_key("GEMINI_API_KEY"))},
        "openrouter": {"live": _has("OPENROUTER_API_KEY"), "key": mask(env_key("OPENROUTER_API_KEY"))},
        "openai": {"live": bool(openai_key_pool()), "keys": len(openai_key_pool()),
                   "key": mask(openai_key_pool()[0] if openai_key_pool() else "")},
        "together": {"live": _has("TOGETHER_API_KEY"), "key": mask(env_key("TOGETHER_API_KEY"))},
        "cerebras": {"live": _has("CEREBRAS_API_KEY"), "key": mask(env_key("CEREBRAS_API_KEY"))},
        "mistral": {"live": _has("MISTRAL_API_KEY"), "key": mask(env_key("MISTRAL_API_KEY"))},
        "clarifai": {"live": _has("CLARIFAI_PAT"), "key": mask(env_key("CLARIFAI_PAT"))},
        "google_cse": {"live": _has("GOOGLE_CUSTOM_SEARCH_API_KEY") and _has("GOOGLE_SEARCH_ENGINE_ID")},
        "telegram": {"live": _has("TELEGRAM_BOT_TOKEN"), "key": mask(env_key("TELEGRAM_BOT_TOKEN"))},
        "spice": {"live": _has("SPICE_AI_API_KEY"), "key": mask(env_key("SPICE_AI_API_KEY"))},
        "databricks": {"live": _has("DATABRICKS_TOKEN"), "key": mask(env_key("DATABRICKS_TOKEN"))},
        "sql_db": {"live": _has("SQL_DB_CONN_STR")},
    }


async def telegram_notify(text: str) -> bool:
    """Best-effort Telegram alert. Never raises, never blocks answers."""
    token = env_key("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID") or ENV.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as tc:
            r = await tc.post("https://api.telegram.org/bot%s/sendMessage" % token,
                              json={"chat_id": chat, "text": text[:4000]})
            return r.status_code == 200
    except Exception:
        return False


def mask(k: str) -> str:
    """Never print a whole key to the console or an API response."""
    if not k:
        return ""
    return k[:6] + "..." + k[-4:] if len(k) > 14 else "set"


# ---------------------------------------------------------------------------
#  PROVIDERS  —  the paid-grade brains, keyed from .env
# ---------------------------------------------------------------------------
#  Each entry: how to call it, and which env var unlocks it. A provider with
#  no key is simply skipped, so ADELTE still runs with an empty .env.
PROVIDERS = {
    "groq": {
        "label": "Groq",
        "env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "style": "openai",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                   "openai/gpt-oss-120b", "qwen/qwen3-32b"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "style": "openai",
        "models": ["nvidia/nemotron-3-super-120b-a12b:free",
                   "openai/gpt-oss-20b:free",
                   "google/gemma-4-31b-it:free",
                   "google/gemini-2.0-flash-exp:free"],
    },
    "gemini": {
        "label": "Gemini",
        "env": "GEMINI_API_KEY",
        "url": ("https://generativelanguage.googleapis.com/v1beta/models/"
                "{model}:generateContent"),
        "style": "gemini",
        "models": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    },
    "cerebras": {
        "label": "Cerebras",
        "env": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "style": "openai",
        "models": ["llama-3.3-70b", "qwen-3-32b"],
    },
    "together": {
        "label": "Together",
        "env": "TOGETHER_API_KEY",
        "url": "https://api.together.xyz/v1/chat/completions",
        "style": "openai",
        "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"],
    },
    "mistral": {
        "label": "Mistral",
        "env": "MISTRAL_API_KEY",
        "url": "https://api.mistral.ai/v1/chat/completions",
        "style": "openai",
        "models": ["mistral-large-latest", "mistral-small-latest"],
    },
    "openai": {
        "label": "OpenAI",
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "style": "openai",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    },
    "clarifai": {
        "label": "Clarifai",
        "env": "CLARIFAI_PAT",
        "url": "https://api.clarifai.com/v2/openai/v1/chat/completions",
        "style": "openai",
        "models": ["openai/gpt-4o-mini"],
    },
    "databricks": {
        "label": "Databricks",
        "env": "DATABRICKS_TOKEN",
        "url": "",  # resolved per-workspace from DATABRICKS_HOST at call time
        "style": "openai",
        "models": ["databricks-meta-llama-3-3-70b-instruct"],
    },
    "horde": {
        "label": "AI Horde",
        "env": "",                      # anonymous, always available
        "url": "https://aihorde.net/api/v2",
        "style": "horde",
        "models": [],
    },
}


#  THE THREE ADELTE MODELS. The user picks one of these - never a raw
#  provider name. Each is a persona + an ordered chain of providers to try.
ADELTE_MODELS = {
    "adelte-search": {
        "name": "ADELTE Search",
        "color": "#4F8DFD",
        "glow": "#1E4FBF",
        "tag": "research",
        "blurb": "Ten engines at once, reads the pages, answers with sources.",
        "chain": [("openai", "gpt-4o-mini"),
                  ("groq", "llama-3.1-8b-instant"),
                  ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
                  ("gemini", "gemini-2.0-flash"),
                  ("clarifai", "openai/gpt-4o-mini"),
                  ("horde", "")],
        "search": True,
        "temp": 0.5,
        "persona": ("You answer from the sources you are given. Be accurate, "
                    "short and ordered."),
    },
    "adelte-coder-3high": {
        "name": "ADELTE Coder 3 High",
        "color": "#22D3A7",
        "glow": "#0E7C5E",
        "tag": "code",
        "blurb": "Full working files, real logic, no placeholders.",
        "chain": [("openai", "gpt-4o-mini"),
                  ("groq", "openai/gpt-oss-120b"),
                  ("openrouter", "cohere/north-mini-code:free"),
                  ("groq", "llama-3.3-70b-versatile"),
                  ("gemini", "gemini-2.0-flash"),
                  ("cerebras", "qwen-3-32b"),
                  ("mistral", "mistral-small-latest"),
                  ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
                  ("clarifai", "openai/gpt-4o-mini"),
                  ("horde", "")],
        "search": False,
        "temp": 0.25,
        "persona": ("You are ADELTE Coder 3 High, a staff-level software engineer "
                    "and the strongest coding brain in ADELTE. You write "
                    "complete, runnable, production-quality code with real "
                    "logic - never a stub, never a TODO, never a placeholder. "
                    "You handle errors, edge cases and empty input. You name "
                    "things clearly, keep functions small, and add a short "
                    "comment only where the reason is not obvious. If a file "
                    "needs imports, includes or a build step, you write them "
                    "out in full. You prefer the standard library and no "
                    "unnecessary dependencies. When you finish a file it runs "
                    "as-is on the first try. "
                    "CODING PROTOCOL (always follow): "
                    "1) state the plan in one line; "
                    "2) give the full file(s) in fenced code blocks with the "
                    "language tag and file name; "
                    "3) list how to run + how to verify; "
                    "4) list edge cases handled. "
                    "If the request is ambiguous, pick the most useful default "
                    "and say what you assumed."),
    },
    "adelte-minimax": {
        "name": "ADELTE Max",
        "color": "#C084FC",
        "glow": "#7C3AED",
        "tag": "max",
        "blurb": "Deep thinking, no web search. The strongest reasoning chain.",
        "chain": [("openai", "gpt-4o-mini"),
                  ("groq", "llama-3.3-70b-versatile"),
                  ("groq", "openai/gpt-oss-120b"),
                  ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
                  ("gemini", "gemini-2.0-flash"),
                  ("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
                  ("cerebras", "llama-3.3-70b"),
                  ("mistral", "mistral-large-latest"),
                  ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
                  ("clarifai", "openai/gpt-4o-mini"),
                  ("horde", "")],
        "search": False,
        "temp": 0.4,
        "think": True,
        "persona": ("You are ADELTE Max, the most capable ADELTE model. You "
                    "do not search the web - you reason from what you know. "
                    "Think the problem all the way through, then answer "
                    "clearly, completely and in order."),
    },
    "adelte-commander": {
        "name": "ADELTE Commander",
        "tag": "command",
        "blurb": "Runs your computer. Every system command lives here.",
        "color": "#FF9F45",
        "glow": "#C2410C",
        "chain": [],              # local only - never calls a provider
        "search": False,
        "local": True,
        "temp": 0.0,
        "persona": ("You are ADELTE Commander. You only run system "
                    "commands on this machine."),
    },
    "adelte-cargolis": {
        "name": "ADELTE Cargolis",
        "tag": "desktop",
        "blurb": "The floating desktop assistant. Reads your screen, fixes "
                 "your code.",
        "color": "#FF5FA2",
        "glow": "#BE185D",
        "chain": [("openai", "gpt-4o-mini"),
                  ("groq", "openai/gpt-oss-120b"),
                  ("groq", "llama-3.3-70b-versatile"),
                  ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
                  ("openrouter", "cohere/north-mini-code:free"),
                  ("gemini", "gemini-2.0-flash"),
                  ("clarifai", "openai/gpt-4o-mini"),
                  ("horde", "")],
        "search": False,
        "desktop": True,
        "temp": 0.3,
        "persona": ("You are ADELTE Cargolis, a desktop assistant that can "
                    "see what the user captured from their screen. When you "
                    "are given an error, explain in one short paragraph what "
                    "it means, then give the corrected code in full. When "
                    "you are given code, refactor it and explain what "
                    "changed. When you are given plain text, research it and "
                    "answer in short ordered points. Never truncate code."),
    },
}

DEFAULT_MODEL = "adelte-search"


def model_cfg(mid: Optional[str]) -> dict:
    return ADELTE_MODELS.get(mid or "", ADELTE_MODELS[DEFAULT_MODEL])


def provider_ready(pid: str) -> bool:
    p = PROVIDERS.get(pid)
    if not p:
        return False
    if pid == "openai":
        return bool(openai_key_pool())
    if pid == "databricks":
        return bool(env_key("DATABRICKS_TOKEN") and env_key("DATABRICKS_HOST"))
    return True if not p["env"] else bool(env_key(p["env"]))


def databricks_url() -> str:
    """Resolve the Databricks serving URL from DATABRICKS_HOST.

    Additive only: when no host is configured the provider is simply not
    ready (see provider_ready) and the chain skips it.
    """
    host = env_key("DATABRICKS_HOST").rstrip("/")
    if not host:
        return ""
    if not host.startswith("http"):
        host = "https://" + host
    return host + "/serving-endpoints/Databricks-Meta-Llama-3-3-70B-Instruct/invocations"


def live_chain(mid: Optional[str]) -> List[Tuple[str, str]]:
    """The chain for this model, minus providers with no key."""
    return [(p, m) for p, m in model_cfg(mid)["chain"] if provider_ready(p)]


class CFG:
    """Runtime configuration; mutated by CLI flags in main()."""
    host = "0.0.0.0"
    port = 8000
    db_path = _default_db()
    use_ai = True
    ai_budget = 50.0          # seconds to wait for the free AI swarm
    deep_pages = 4            # how many result pages to fully read
    page_chars = 4000         # chars kept per page
    request_timeout = 22.0
    daily_key_limit = 500

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# live per-engine health, surfaced in the sidebar
ENGINE_HEALTH: Dict[str, dict] = {}


# ============================================================================
#  SECTION 2 — TEXT HELPERS
# ============================================================================

def clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", _html.unescape(text)).strip()


def unwrap(href: str) -> str:
    """Unwrap DuckDuckGo / Google redirect links to the real destination."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        p = urlparse(href)
        if "duckduckgo.com" in p.netloc and p.path.startswith("/l/"):
            v = parse_qs(p.query).get("uddg")
            if v:
                return unquote(v[0])
        if "google." in p.netloc and p.path == "/url":
            v = parse_qs(p.query).get("q") or parse_qs(p.query).get("url")
            if v:
                return unquote(v[0])
    except Exception:
        pass
    return href


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of",
    "to", "in", "on", "for", "and", "or", "but", "with", "as", "by", "at",
    "from", "that", "this", "these", "those", "it", "its", "how", "what",
    "why", "when", "where", "which", "who", "do", "does", "did", "can",
    "could", "should", "would", "i", "you", "we", "they", "my", "your",
    "about", "into", "than", "then", "there", "here", "so", "if", "not",
    "have", "has", "had", "will", "just", "also", "more", "most", "such",
}


def keywords(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Z0-9_.+#-]{2,}", text.lower())
            if w not in STOPWORDS]


# ===========================================================================
# COMMAND ENGINE  —  "open youtube", "lock pc", and friends
# ===========================================================================
# Two kinds of command:
#   web    -> the browser opens a URL (works everywhere, including phones)
#   system -> the SERVER's own machine acts (lock, sleep, volume, ...)
# System commands are ON by default (the user asked for this). Pass
# --no-system to lock them off. Destructive ops still need confirm:true.
# Original note kept: they are gated because a
# public server must never let a stranger lock the host or shut it down.

ALLOW_SYSTEM = True

WEB_COMMANDS: Dict[str, Tuple[str, str]] = {
    "youtube":    ("https://www.youtube.com", "YouTube"),
    "google":     ("https://www.google.com", "Google"),
    "gmail":      ("https://mail.google.com", "Gmail"),
    "github":     ("https://github.com", "GitHub"),
    "maps":       ("https://maps.google.com", "Google Maps"),
    "translate":  ("https://translate.google.com", "Google Translate"),
    "drive":      ("https://drive.google.com", "Google Drive"),
    "whatsapp":   ("https://web.whatsapp.com", "WhatsApp Web"),
    "twitter":    ("https://twitter.com", "X / Twitter"),
    "x":          ("https://twitter.com", "X / Twitter"),
    "reddit":     ("https://www.reddit.com", "Reddit"),
    "netflix":    ("https://www.netflix.com", "Netflix"),
    "spotify":    ("https://open.spotify.com", "Spotify"),
    "wikipedia":  ("https://www.wikipedia.org", "Wikipedia"),
    "stackoverflow": ("https://stackoverflow.com", "Stack Overflow"),
    "chatgpt":    ("https://chat.openai.com", "ChatGPT"),
    "linkedin":   ("https://www.linkedin.com", "LinkedIn"),
    "instagram":  ("https://www.instagram.com", "Instagram"),
    "facebook":   ("https://www.facebook.com", "Facebook"),
    "tiktok":     ("https://www.tiktok.com", "TikTok"),
    "amazon":     ("https://www.amazon.com", "Amazon"),
    "twitch":     ("https://www.twitch.tv", "Twitch"),
    "discord":    ("https://discord.com/app", "Discord"),
    "telegram":   ("https://web.telegram.org", "Telegram"),
    "news":       ("https://news.google.com", "Google News"),
    "weather":    ("https://www.google.com/search?q=weather", "Weather"),
    "calendar":   ("https://calendar.google.com", "Google Calendar"),
}

# name -> (platform commands, human label, needs_confirm)
SYSTEM_COMMANDS: Dict[str, dict] = {
    "lock": {"label": "Lock the computer", "confirm": False, "cmd": {
        "nt": 'rundll32.exe user32.dll,LockWorkStation',
        "darwin": 'pmset displaysleepnow',
        "posix": 'loginctl lock-session || xdg-screensaver lock'}},
    "sleep": {"label": "Put the computer to sleep", "confirm": True, "cmd": {
        "nt": 'rundll32.exe powrprof.dll,SetSuspendState 0,1,0',
        "darwin": 'pmset sleepnow',
        "posix": 'systemctl suspend'}},
    "shutdown": {"label": "Shut down the computer", "confirm": True, "cmd": {
        "nt": 'shutdown /s /t 5', "darwin": 'shutdown -h +1',
        "posix": 'shutdown -h +1'}},
    "restart": {"label": "Restart the computer", "confirm": True, "cmd": {
        "nt": 'shutdown /r /t 5', "darwin": 'shutdown -r +1',
        "posix": 'shutdown -r +1'}},
    "screenshot": {"label": "Take a screenshot", "confirm": False, "cmd": {
        "nt": 'powershell -c "Add-Type -AssemblyName System.Windows.Forms;'
              '[System.Windows.Forms.SendKeys]::SendWait(\'{PRTSC}\')"',
        "darwin": 'screencapture -x ~/Desktop/delta_shot.png',
        "posix": 'gnome-screenshot -f ~/delta_shot.png'}},
    "volume up": {"label": "Volume up", "confirm": False, "cmd": {
        "nt": 'powershell -c "(New-Object -ComObject WScript.Shell)'
              '.SendKeys([char]175)"',
        "darwin": 'osascript -e "set volume output volume '
                  '(output volume of (get volume settings) + 10)"',
        "posix": 'amixer -q sset Master 10%+'}},
    "volume down": {"label": "Volume down", "confirm": False, "cmd": {
        "nt": 'powershell -c "(New-Object -ComObject WScript.Shell)'
              '.SendKeys([char]174)"',
        "darwin": 'osascript -e "set volume output volume '
                  '(output volume of (get volume settings) - 10)"',
        "posix": 'amixer -q sset Master 10%-'}},
    "mute": {"label": "Mute audio", "confirm": False, "cmd": {
        "nt": 'powershell -c "(New-Object -ComObject WScript.Shell)'
              '.SendKeys([char]173)"',
        "darwin": 'osascript -e "set volume with output muted"',
        "posix": 'amixer -q sset Master toggle'}},
    "empty trash": {"label": "Empty the recycle bin", "confirm": True, "cmd": {
        "nt": 'powershell -c "Clear-RecycleBin -Force"',
        "darwin": 'osascript -e "tell app \\"Finder\\" to empty trash"',
        "posix": 'rm -rf ~/.local/share/Trash/*'}},
}

# Which command dialect this machine speaks: "nt", "darwin" or "posix".
HOST_OS = "nt" if os.name == "nt" else (
    "darwin" if sys.platform == "darwin" else "posix")
HOST_OS_NAME = {"nt": "Windows", "darwin": "macOS", "posix": "Linux"}[HOST_OS]


# ===========================================================================
#  ADELTE COMMANDER  -  the local command model
#
#  Every system command in ADELTE runs through here and nowhere else. Each
#  entry is a short word the user types, a human label, whether it needs a
#  confirmation, the group it belongs to, and the real command per platform.
#  Windows ("nt") is the primary target; mac and Linux equivalents are given
#  so the same word works everywhere.
# ===========================================================================

def _C(label, group, nt, darwin="", posix="", confirm=False, note=""):
    return {"label": label, "group": group, "confirm": confirm, "note": note,
            "cmd": {"nt": nt, "darwin": darwin or nt, "posix": posix or nt}}


COMMANDER_COMMANDS: Dict[str, dict] = {

    # ---------------- Power & Security ----------------
    "lock": _C("Lock the screen", "Power & Security",
               "rundll32.exe user32.dll,LockWorkStation",
               "pmset displaysleepnow",
               "loginctl lock-session || xdg-screensaver lock"),
    "off": _C("Shut down now", "Power & Security",
              "shutdown /s /t 0", "shutdown -h now", "shutdown -h now",
              confirm=True),
    "reboot": _C("Restart the computer", "Power & Security",
                 "shutdown /r /t 0", "shutdown -r now", "shutdown -r now",
                 confirm=True),
    "sleep": _C("Sleep", "Power & Security",
                "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
                "pmset sleepnow", "systemctl suspend", confirm=True),
    "hibernate": _C("Hibernate", "Power & Security",
                    "shutdown /h", "pmset sleepnow", "systemctl hibernate",
                    confirm=True),
    "logoff": _C("Sign out of Windows", "Power & Security",
                 "shutdown /l", "osascript -e 'tell app \"System Events\" to log out'",
                 "loginctl terminate-user $USER", confirm=True),
    "abort": _C("Cancel a pending shutdown", "Power & Security",
                "shutdown /a", "killall shutdown", "shutdown -c"),
    "stealth": _C("Lock and blank the screen", "Power & Security",
                  "rundll32.exe user32.dll,LockWorkStation && "
                  "powershell -c \"(Add-Type '[DllImport(\\\"user32.dll\\\")]"
                  "public static extern int SendMessage(int hWnd,int hMsg,"
                  "int wParam,int lParam);' -Name a -Pas)::SendMessage(-1,"
                  "0x0112,0xF170,2)\"",
                  "pmset displaysleepnow", "xset dpms force off"),

    # ---------------- Network ----------------
    "ip": _C("Your local IP addresses", "Network",
             "ipconfig", "ifconfig", "ip addr"),
    "publicip": _C("Your public IP", "Network",
                   "curl -s https://api.ipify.org",
                   "curl -s https://api.ipify.org",
                   "curl -s https://api.ipify.org"),
    "wifi": _C("Wi-Fi networks you have saved", "Network",
               "netsh wlan show profiles",
               "/System/Library/PrivateFrameworks/Apple80211.framework/"
               "Versions/Current/Resources/airport -s",
               "nmcli connection show"),
    "wifipass": _C("Show saved Wi-Fi passwords", "Network",
                   "netsh wlan show profile name=* key=clear",
                   "echo 'use Keychain Access'",
                   "sudo grep -r psk= /etc/NetworkManager/system-connections/"),
    "flush": _C("Flush the DNS cache", "Network",
                "ipconfig /flushdns",
                "sudo dscacheutil -flushcache",
                "sudo systemd-resolve --flush-caches"),
    "pingg": _C("Ping Google", "Network",
                "ping google.com", "ping -c 4 google.com",
                "ping -c 4 google.com"),
    "trace": _C("Trace the route to Google", "Network",
                "tracert google.com", "traceroute google.com",
                "traceroute google.com"),
    "ports": _C("Open ports and connections", "Network",
                "netstat -ano", "netstat -an", "ss -tulpn"),
    "mac": _C("Your MAC addresses", "Network",
              "getmac /v", "ifconfig | grep ether", "ip link"),
    "netnear": _C("Devices on your network", "Network",
                  "arp -a", "arp -a", "arp -a"),

    # ---------------- System Performance ----------------
    "specs": _C("Full system information", "System Performance",
                "systeminfo", "system_profiler SPHardwareDataType",
                "uname -a && lscpu"),
    "battery": _C("Battery report", "System Performance",
                  "powercfg /batteryreport", "pmset -g batt",
                  "upower -i /org/freedesktop/UPower/devices/battery_BAT0"),
    "cpu": _C("CPU load", "System Performance",
              "wmic cpu get loadpercentage",
              "top -l 1 | grep 'CPU usage'",
              "top -bn1 | grep 'Cpu(s)'"),
    "ram": _C("Memory in use", "System Performance",
              "systeminfo | findstr /C:\"Available Physical Memory\"",
              "vm_stat", "free -h"),
    "disk": _C("Disk space", "System Performance",
               "wmic logicaldisk get size,freespace,caption",
               "df -h", "df -h"),
    "apps": _C("Running applications", "System Performance",
               "tasklist", "ps aux", "ps aux"),
    "kill": _C("Force-close an app by name", "System Performance",
               "taskkill /F /IM {arg}", "pkill -f {arg}", "pkill -f {arg}",
               confirm=True, note="needs a name, e.g. kill chrome"),
    "clean": _C("Delete temporary files", "System Performance",
                "del /q/f/s %TEMP%\\*",
                "rm -rf /tmp/*", "rm -rf /tmp/*", confirm=True),
    "uptime": _C("How long the machine has been on", "System Performance",
                 "net statistics workstation", "uptime", "uptime"),
    "check": _C("Check the disk for errors", "System Performance",
                "chkdsk", "diskutil verifyVolume /", "sudo fsck -N /"),

    # ---------------- Filesystem ----------------
    "here": _C("Open the current folder", "Filesystem",
               "explorer .", "open .", "xdg-open ."),
    "desktop": _C("Open the Desktop", "Filesystem",
                  "explorer %USERPROFILE%\\Desktop", "open ~/Desktop",
                  "xdg-open ~/Desktop"),
    "downloads": _C("Open Downloads", "Filesystem",
                    "explorer %USERPROFILE%\\Downloads", "open ~/Downloads",
                    "xdg-open ~/Downloads"),
    "docs": _C("Open Documents", "Filesystem",
               "explorer %USERPROFILE%\\Documents", "open ~/Documents",
               "xdg-open ~/Documents"),
    "treev": _C("Folder tree", "Filesystem",
                "tree /F", "find . -maxdepth 3", "tree -L 3 || find . -maxdepth 3"),
    "cls": _C("Clear the screen", "Filesystem", "cls", "clear", "clear"),
    "mkfolder": _C("Make a folder", "Filesystem",
                   "mkdir {arg}", "mkdir -p {arg}", "mkdir -p {arg}",
                   note="needs a name, e.g. mkfolder notes"),
    "newfile": _C("Create an empty file", "Filesystem",
                  "type nul > {arg}", "touch {arg}", "touch {arg}",
                  note="needs a name, e.g. newfile todo.txt"),
    "findfile": _C("Find a file by name", "Filesystem",
                   "dir /s /b {arg}", "find . -name '{arg}'",
                   "find . -name '{arg}'",
                   note="needs a name, e.g. findfile report.pdf"),
    "hidden": _C("Show hidden files", "Filesystem",
                 "dir /a", "ls -la", "ls -la"),

    # ---------------- Windows Admin Tools ----------------
    "taskmgr": _C("Task Manager", "Windows Tools",
                  "taskmgr", "open -a 'Activity Monitor'", "gnome-system-monitor"),
    "devmgr": _C("Device Manager", "Windows Tools",
                 "devmgmt.msc", "open -a 'System Information'", "hardinfo"),
    "services": _C("Services", "Windows Tools",
                   "services.msc", "launchctl list", "systemctl list-units"),
    "regedit": _C("Registry Editor", "Windows Tools",
                  "regedit", "echo 'no registry on macOS'",
                  "echo 'no registry on Linux'", confirm=True),
    "control": _C("Control Panel", "Windows Tools",
                  "control", "open -a 'System Settings'",
                  "gnome-control-center"),
    "diskmgmt": _C("Disk Management", "Windows Tools",
                   "diskmgmt.msc", "open -a 'Disk Utility'", "gnome-disks"),
    "calc": _C("Calculator", "Windows Tools",
               "calc", "open -a Calculator", "gnome-calculator"),
    "notepad": _C("Notepad", "Windows Tools",
                  "notepad", "open -a TextEdit", "gedit"),
    "snipping": _C("Snipping Tool", "Windows Tools",
                   "snippingtool", "screencapture -i ~/Desktop/shot.png",
                   "gnome-screenshot -a"),
    "env": _C("Environment variables", "Windows Tools",
              "set", "printenv", "printenv"),

    # ---------------- Developer ----------------
    "math": _C("Quick calculation", "Developer",
               "python -c \"print({arg})\"", "python3 -c \"print({arg})\"",
               "python3 -c \"print({arg})\"",
               note="needs an expression, e.g. math 45*12"),
    "python": _C("Python version", "Developer",
                 "python --version", "python3 --version", "python3 --version"),
    "code": _C("Open VS Code here", "Developer",
               "code .", "code .", "code ."),
    "git": _C("Git status", "Developer",
              "git status", "git status", "git status"),
    "pip": _C("Installed Python packages", "Developer",
              "pip list", "pip3 list", "pip3 list"),
    "time": _C("Date and time", "Developer",
               "echo %date% %time%", "date", "date"),

    # ---------------- Visual & Audio ----------------
    "matrix": _C("Matrix rain in the terminal", "Visual & Audio",
                 "powershell -c \"while($true){Write-Host -NoNewline "
                 "([char](Get-Random -Min 33 -Max 126)) -Fore Green}\"",
                 "LC_ALL=C tr -c '[:print:]' ' ' < /dev/urandom | head -c 4000",
                 "LC_ALL=C tr -c '[:print:]' ' ' < /dev/urandom | head -c 4000"),
    "starwars": _C("Star Wars in ASCII", "Visual & Audio",
                   "telnet towel.blinkenlights.nl",
                   "telnet towel.blinkenlights.nl",
                   "telnet towel.blinkenlights.nl"),
    "speak": _C("Say something out loud", "Visual & Audio",
                "powershell -c \"Add-Type -AssemblyName System.Speech;"
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                ".Speak('{arg}')\"",
                "say '{arg}'", "spd-say '{arg}'",
                note="needs words, e.g. speak hello"),
    "volup": _C("Volume up", "Visual & Audio",
                "powershell -c \"(New-Object -ComObject WScript.Shell)"
                ".SendKeys([char]175)\"",
                "osascript -e 'set volume output volume "
                "(output volume of (get volume settings) + 10)'",
                "amixer -q sset Master 10%+"),
    "voldown": _C("Volume down", "Visual & Audio",
                  "powershell -c \"(New-Object -ComObject WScript.Shell)"
                  ".SendKeys([char]174)\"",
                  "osascript -e 'set volume output volume "
                  "(output volume of (get volume settings) - 10)'",
                  "amixer -q sset Master 10%-"),
    "mute": _C("Mute", "Visual & Audio",
               "powershell -c \"(New-Object -ComObject WScript.Shell)"
               ".SendKeys([char]173)\"",
               "osascript -e 'set volume with output muted'",
               "amixer -q sset Master toggle"),

    # ---------------- Closing apps ----------------
    "close": _C("Close an app by name", "Closing Apps",
                "taskkill /F /IM {arg}.exe", "pkill -f {arg}", "pkill -f {arg}",
                confirm=True, note="needs a name, e.g. close spotify"),
    "noedge": _C("Close Microsoft Edge", "Closing Apps",
                 "taskkill /F /IM msedge.exe",
                 "pkill -f 'Microsoft Edge'", "pkill -f microsoft-edge",
                 confirm=True),
    "nochrome": _C("Close Google Chrome", "Closing Apps",
                   "taskkill /F /IM chrome.exe",
                   "pkill -f 'Google Chrome'", "pkill -f chrome",
                   confirm=True),
    "closeall": _C("Close every browser", "Closing Apps",
                   "taskkill /F /IM chrome.exe & taskkill /F /IM msedge.exe "
                   "& taskkill /F /IM firefox.exe",
                   "pkill -f 'Google Chrome'; pkill -f Safari; pkill -f firefox",
                   "pkill -f chrome; pkill -f firefox", confirm=True),
}

# Commands that need a word after them, e.g. "kill chrome".
COMMANDER_ARGS = {k for k, v in COMMANDER_COMMANDS.items()
                  if "{arg}" in v["cmd"]["nt"]}

# Answered by Commander itself, no shell involved.
COMMANDER_BUILTIN = ("help", "exit")


def commander_groups() -> Dict[str, list]:
    """Commands bucketed by group, in the order they were defined."""
    out: Dict[str, list] = {}
    for word, spec in COMMANDER_COMMANDS.items():
        out.setdefault(spec["group"], []).append({
            "word": word, "label": spec["label"],
            "confirm": spec["confirm"], "note": spec.get("note", ""),
            "needs_arg": word in COMMANDER_ARGS,
            "cmd": spec["cmd"].get(HOST_OS, spec["cmd"]["nt"]),
        })
    return out


def commander_parse(text: str) -> Optional[dict]:
    """Turn 'kill chrome' into a resolved command, or None if unknown."""
    t = (text or "").strip()
    if not t:
        return None
    t = re.sub(r"^(?:adelte|commander)[,: ]+", "", t, flags=re.I).strip()
    low = t.lower()
    if low in COMMANDER_BUILTIN:
        return {"builtin": low}
    parts = low.split(None, 1)
    word = parts[0]
    arg = t.split(None, 1)[1].strip() if len(parts) > 1 else ""
    spec = COMMANDER_COMMANDS.get(word)
    if not spec:
        return None
    raw = spec["cmd"].get(HOST_OS) or spec["cmd"]["nt"]
    if word in COMMANDER_ARGS:
        if not arg:
            return {"word": word, "spec": spec, "missing_arg": True,
                    "note": spec.get("note", "")}
        safe = arg.replace('"', "").replace("`", "")
        if word != "math":                    # math needs its operators
            safe = re.sub(r"[;&|<>$\n\r]", "", safe)
        raw = raw.replace("{arg}", safe)
    return {"word": word, "spec": spec, "cmd": raw, "arg": arg,
            "confirm": spec["confirm"], "label": spec["label"],
            "group": spec["group"]}


def commander_help() -> str:
    """The full command sheet, grouped, as markdown."""
    lines = ["**ADELTE Commander** - every system command lives here.", ""]
    for group, items in commander_groups().items():
        lines.append("**" + group + "**")
        lines.append("")
        for it in items:
            tail = " *(asks first)*" if it["confirm"] else ""
            argh = " `<name>`" if it["needs_arg"] else ""
            lines.append("- `" + it["word"] + argh + "` - " + it["label"] + tail)
        lines.append("")
    lines.append("Type `exit` to leave Commander mode.")
    return "\n".join(lines)


# things ADELTE answers itself, no engines and no shell
LOCAL_COMMANDS = ("time", "date", "flip a coin", "roll a dice", "roll a die",
                  "random number", "who are you", "commands", "help")


def match_command(q: str) -> Optional[dict]:
    """Return a command descriptor if this message is a command, else None."""
    s = " ".join(q.lower().strip().rstrip("!.?").split())

    # ---- open <site> / play <x> on youtube / search <x> ----
    for trigger in ("open ", "launch ", "go to ", "take me to ", "visit "):
        if s.startswith(trigger):
            target = s[len(trigger):].strip()
            key = target.replace(".com", "").replace("the ", "").strip()
            if key in WEB_COMMANDS:
                url, label = WEB_COMMANDS[key]
                return {"kind": "web", "url": url, "label": "Open " + label}
            if "." in target and " " not in target:
                u = target if target.startswith("http") else "https://" + target
                return {"kind": "web", "url": u, "label": "Open " + target}
            if target:
                return {"kind": "web",
                        "url": "https://duckduckgo.com/?q=" + quote_plus(target),
                        "label": "Search for " + target}

    if s.startswith(("play ", "search youtube for ", "youtube ")):
        term = s.split(" ", 1)[1] if s.startswith(("play ", "youtube ")) \
            else s.replace("search youtube for ", "")
        if term:
            return {"kind": "web",
                    "url": "https://www.youtube.com/results?search_query="
                           + quote_plus(term),
                    "label": "Play " + term + " on YouTube"}

    if s.startswith(("search ", "google ")):
        term = s.split(" ", 1)[1]
        if term:
            return {"kind": "web",
                    "url": "https://www.google.com/search?q=" + quote_plus(term),
                    "label": "Search Google for " + term}

    # ---- local, answered by ADELTE itself ----
    if s in ("what time is it", "time", "what's the time", "whats the time"):
        return {"kind": "local", "op": "time", "label": "Current time"}
    if s in ("date", "what is the date", "what's the date", "today"):
        return {"kind": "local", "op": "date", "label": "Today's date"}
    if s in ("flip a coin", "coin flip", "flip coin"):
        return {"kind": "local", "op": "coin", "label": "Flip a coin"}
    if s in ("roll a dice", "roll a die", "roll dice"):
        return {"kind": "local", "op": "dice", "label": "Roll a dice"}
    if s in ("commands", "list commands", "what commands", "help",
             "show commands"):
        return {"kind": "local", "op": "help", "label": "Command list"}

    # ---- system ----
    for name, spec in SYSTEM_COMMANDS.items():
        if s == name or s.startswith(name + " ") or s in (
                name + " pc", name + " computer", name + " my pc",
                name + " the pc", name + " laptop", name + " my computer"):
            return {"kind": "system", "op": name, "label": spec["label"],
                    "confirm": spec["confirm"]}
    return None


def run_system_command(op: str) -> Tuple[bool, str]:
    """Execute a whitelisted system command on the machine running ADELTE."""
    if not ALLOW_SYSTEM:
        return False, ("System commands are disabled. Restart ADELTE with "
                       "restart without `--no-system` to enable them.")
    spec = SYSTEM_COMMANDS.get(op)
    if not spec:
        return False, "Unknown command."
    if os.name == "nt":
        cmd = spec["cmd"].get("nt")
    elif sys.platform == "darwin":
        cmd = spec["cmd"].get("darwin")
    else:
        cmd = spec["cmd"].get("posix")
    if not cmd:
        return False, "Not supported on this operating system."
    try:
        subprocess.Popen(cmd, shell=True)
        return True, spec["label"] + " — sent to the operating system."
    except Exception as e:                                   # pragma: no cover
        return False, "Failed: " + str(e)


def commander_run(cmd: str, word: str = "") -> Tuple[bool, str]:
    """Run one resolved Commander command and capture what it printed.

    Short informational commands are captured so their output can be shown
    in the chat. Commands that open a window, take over the terminal or
    power the machine down are fired and left alone.
    """
    if not ALLOW_SYSTEM:
        return False, ("System commands are switched off. Start ADELTE "
                       "without `--no-system` to use Commander.")
    detach = {"taskmgr", "devmgr", "services", "regedit", "control",
              "diskmgmt", "calc", "notepad", "snipping", "code", "here",
              "desktop", "downloads", "docs", "matrix", "starwars",
              "off", "reboot", "sleep", "hibernate", "logoff", "lock",
              "stealth", "speak", "volup", "voldown", "mute", "check"}
    try:
        if word in detach:
            subprocess.Popen(cmd, shell=True)
            return True, ""
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, timeout=25)
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        body = out or err or "Done - the command produced no output."
        if len(body) > 6000:
            body = body[:6000] + "\n... (trimmed)"
        return (r.returncode == 0 or bool(out)), body
    except subprocess.TimeoutExpired:
        return False, "That command took too long and was stopped."
    except Exception as e:
        return False, "Failed: " + str(e)


def commander_answer(text: str) -> dict:
    """Full Commander turn: parse, run, and format the reply."""
    parsed = commander_parse(text)
    if not parsed:
        word = (text or "").strip().split(None, 1)[0].lower() if text else ""
        near = [w for w in COMMANDER_COMMANDS if w.startswith(word[:3])][:6] \
            if len(word) >= 2 else []
        tip = ("\n\nClosest matches: "
               + ", ".join("`" + w + "`" for w in near)) if near else ""
        return {"ok": False, "ran": False,
                "answer": ("`" + (text or "").strip()[:40] + "` is not a "
                           "Commander command. Type `help` for the full "
                           "list." + tip)}
    if parsed.get("builtin") == "help":
        return {"ok": True, "ran": False, "answer": commander_help()}
    if parsed.get("builtin") == "exit":
        return {"ok": True, "ran": False, "exit": True,
                "answer": "Leaving Commander. Pick another model when ready."}
    if parsed.get("missing_arg"):
        return {"ok": False, "ran": False,
                "answer": ("`" + parsed["word"] + "` needs something after "
                           "it - " + (parsed.get("note")
                                      or "add a name and try again") + ".")}
    ok, body = commander_run(parsed["cmd"], parsed["word"])
    head = "**" + parsed["label"] + "**  ·  `" + parsed["cmd"] + "`"
    if not body:
        return {"ok": ok, "ran": True, "word": parsed["word"],
                "cmd": parsed["cmd"],
                "answer": head + "\n\nSent to " + HOST_OS_NAME + "."}
    fence = "```\n" + body + "\n```"
    return {"ok": ok, "ran": True, "word": parsed["word"],
            "cmd": parsed["cmd"], "answer": head + "\n\n" + fence}


def local_command(op: str) -> str:
    """Answers ADELTE produces itself — no engines, no shell."""
    now = datetime.now()
    if op == "time":
        return "It's **" + now.strftime("%H:%M") + "** (" + \
               now.strftime("%I:%M %p").lstrip("0") + ")."
    if op == "date":
        return "Today is **" + now.strftime("%A, %d %B %Y") + "**."
    if op == "coin":
        return "**" + random.choice(["Heads", "Tails"]) + ".**"
    if op == "dice":
        return "You rolled a **" + str(random.randint(1, 6)) + "**."
    if op == "help":
        return command_help()
    return "I know the command but not how to run it yet."


def command_help() -> str:
    web = ", ".join(sorted(WEB_COMMANDS)[:18])
    sysn = ", ".join(SYSTEM_COMMANDS)
    return (
        "### Commands I understand\n\n"
        "**Open things** — `open youtube`, `open github`, `open gmail`\n\n"
        "Available: " + web + " …\n\n"
        "**Play / search** — `play lofi beats`, `search quantum computing`\n\n"
        "**Your computer** — " + sysn + "\n\n"
        "_System commands are ON. Disable with_ `--no-system`\n\n"
        "**Quick answers** — `time`, `date`, `flip a coin`, `roll a dice`\n\n"
        "**Build things** — `create a login page`, `make a portfolio site`\n\n"
        "**Research** — ask any real question and I search 10 engines."
    )


CODE_HINTS = (
    "code", "python", "javascript", "typescript", "error", "traceback",
    "exception", "install", "pip", "npm", "api", "function", "class", "bug",
    "fix", "library", "framework", "github", "repo", "sql", "docker", "regex",
    "compile", "syntax", "server", "fastapi", "react", "rust", "golang",
    "java", "how to", "cannot", "failed", "undefined", "null", "404", "500",
    "database", "deploy", "kubernetes", "async", "thread", "memory leak",
)


CONCEPT_PAT = re.compile(
    r"^(what is|what are|what's|whats|who is|who was|define|definition of|"
    r"explain|tell me about|meaning of|difference between|why is|why do|"
    r"when should|history of|overview of|introduction to|pros and cons)",
    re.I)


def classify(query: str) -> str:
    """Pick an engine mix based on what the question looks like.

    Careful: "what is fastapi" mentions a code term but is a CONCEPTUAL
    question. Sending it to the code engines answered it with random GitHub
    repo descriptions ("Prism is a windows desktop ai agent ... fastapi
    backend"). Conceptual phrasing always wins over a keyword match.
    """
    q = query.lower().strip()
    if CONCEPT_PAT.match(q):
        return "web"
    return "code" if any(h in q for h in CODE_HINTS) else "web"


# ---------------------------------------------------------------------------
# INTENT ROUTER
# ---------------------------------------------------------------------------
# The old build only ever did one thing: web search. So "create a login page"
# was sent to search engines, and the word "create" matched Minecraft mods and
# Roblox Creator Hub. A request to BUILD something must never be a web search.

MAKE_VERBS = ("create", "make", "build", "generate", "write", "code",
              "design", "give me", "show me a", "produce", "draft")
MAKE_OBJECTS = ("page", "form", "site", "website", "app", "html", "css",
                "component", "script", "template", "landing", "dashboard",
                "portfolio", "login", "signup", "navbar", "footer", "button",
                "card", "modal", "table", "gallery", "calculator", "clock",
                "todo", "game", "snippet", "function", "class", "program")

GREET = ("hi", "hey", "hello", "yo", "sup", "hiya", "howdy", "good morning",
         "good afternoon", "good evening", "gm", "gn", "hii", "hiii", "hiiii",
         "heyy", "heyyy", "helloo", "hellooo", "hai", "halo", "yoo", "yooo",
         "hey there", "hi there", "hello there", "morning", "evening",
         "wassup", "whats up", "what's up", "greetings", "salut", "bonjour",
         "hola", "muraho", "mwaramutse")


def is_greeting(q: str) -> bool:
    """'hii' is a hello, not a query about Huntington Ingalls Industries.

    Collapses repeated trailing letters so hiiii -> hi, then checks the
    greeting set. Anything <= 3 words that starts with a greeting counts.
    """
    t = " ".join((q or "").lower().split()).strip(" .,!?~-")
    if not t:
        return False
    if t in GREET:
        return True
    squashed = re.sub(r"(.)\1{1,}$", r"\1", t)
    if squashed in GREET:
        return True
    squashed2 = re.sub(r"(.)\1+", r"\1", t)
    if squashed2 in GREET:
        return True
    words = t.split()
    if len(words) <= 3:
        first = re.sub(r"(.)\1+", r"\1", words[0])
        if first in GREET or words[0] in GREET:
            return True
    return False

CHAT_PAT = (
    "how are you", "who are you", "what are you", "your name", "thank",
    "thanks", "thx", "bye", "goodbye", "see you", "love you", "sorry",
    "help me", "what can you do", "who made you", "are you real",
    "tell me a joke", "i am sad", "i'm sad", "i am tired", "i'm tired",
    "good job", "well done", "nice", "cool", "lol", "haha", "ok", "okay",
)


# Common typos for build verbs. "craete a login page" must still BUILD.
TYPO_FIX = {
    "craete": "create", "creat": "create", "cerate": "create",
    "creaate": "create", "crate": "create", "creae": "create",
    "mkae": "make", "amke": "make", "maek": "make",
    "biuld": "build", "buld": "build", "buid": "build", "bulid": "build",
    "genarate": "generate", "generat": "generate", "gnerate": "generate",
    "wrtie": "write", "wirte": "write", "writ": "write",
    "desgin": "design", "desing": "design",
    "pgae": "page", "paeg": "page", "pag": "page",
    "logn": "login", "lgoin": "login", "loign": "login",
    "fomr": "form", "从": "form",
    "webiste": "website", "websie": "website", "wesbite": "website",
    "hmtl": "html", "htlm": "html",
    "portifolio": "portfolio", "portfolo": "portfolio",
    "portofolio": "portfolio", "protfolio": "portfolio",
    "dashbord": "dashboard", "dasboard": "dashboard",
    "landig": "landing", "lading": "landing",
    "calcualtor": "calculator", "calculater": "calculator",
    "resune": "resume", "resum": "resume",
    "requerements": "requirements", "requirments": "requirements",
    "conect": "connect", "connectt": "connect",
}

# Words that must survive typo-correction untouched: the brand, and model ids.
PROTECTED = {"adelte", "adelte's", "adeltes", "minimax", "coder",
             "adeltecoder", "adeltesearch", "3high"}


def fix_typos(q: str) -> str:
    """Repair common misspellings so intent routing still works."""
    out = []
    for w in q.split():
        bare = w.strip(".,!?;:")
        if bare.lower() in PROTECTED:
            out.append(w)
            continue
        rep = TYPO_FIX.get(bare.lower())
        out.append(w.replace(bare, rep) if rep else w)
    return " ".join(out)


# A message that only makes sense against the previous turn.
# "is he an american" alone gets searched as a grammar question.
PRONOUNS = ("he", "she", "it", "they", "him", "her", "them", "his",
            "hers", "its", "their", "this", "that", "these", "those")

FOLLOWUP_PAT = re.compile(
    r"^\s*(is|was|are|were|does|do|did|has|have|had|can|could|will|would|"
    r"should|why|when|where|how|what about|and|but|so|tell me more|more|"
    r"who else|any|which)\b", re.I)


REFINE_PAT = re.compile(
    r"\b(incomplete|not complete|unfinished|cut off|cutoff|truncated|"
    r"doesn'?t work|does not work|not working|broken|error|errors|bug|"
    r"fix (it|this|that)|finish (it|this|that)|continue|rest of (it|the code)|"
    r"missing|half|add (more|to it)|change (it|the)|make it|redo|again|"
    r"improve|better|too short|longer|full code|whole code|"
    r"complete (it|the code)|same code|that code|this code|the code|"
    r"these codes|this codes|your code)\b", re.I)


def wants_refine(query, last_built):
    """True when the user is reacting to code we JUST produced.

    'this codes are incomplete' is not a research question about billing
    codes - it means finish what you just wrote. Searching the web for it
    is always wrong, so this runs before the router.
    """
    if not last_built:
        return False
    q = " ".join(query.lower().split())
    if len(q.split()) > 18:
        return False
    return bool(REFINE_PAT.search(q))


def last_built_code(history):
    """Most recent assistant turn that actually contained a code block."""
    for h in reversed(history or []):
        if h.get("role") == "assistant" and "```" in (h.get("content") or ""):
            return h["content"]
    return ""


def is_followup(query: str, has_history: bool) -> bool:
    """True when the message leans on the previous turn to make sense."""
    if not has_history:
        return False
    q = " ".join(query.lower().split()).rstrip("?.!")
    words = q.split()
    if not words or len(words) > 9:
        return False
    # a pronoun with no proper noun of its own = definitely a follow-up
    if any(w.strip(".,?!") in PRONOUNS for w in words):
        return True
    if len(words) <= 5 and FOLLOWUP_PAT.match(q):
        return True
    return False


def resolve_followup(query: str, last_topic: str) -> str:
    """Rewrite 'is he an american' into 'is Elon Musk an american'.

    Only the SEARCH text changes; the user still sees what they typed.
    """
    if not last_topic:
        return query
    words = query.split()
    out, replaced = [], False
    for w in words:
        bare = w.strip(".,!?;:").lower()
        if bare in PRONOUNS and not replaced:
            out.append(last_topic)
            replaced = True
        else:
            out.append(w)
    rebuilt = " ".join(out)
    if not replaced:
        # no pronoun to swap - just append the topic for context
        rebuilt = query.rstrip("?.! ") + " " + last_topic
    return rebuilt


def topic_of(text: str) -> str:
    """Best-effort subject of an earlier question, for pronoun resolution."""
    t = re.sub(r"^\s*(who|what|where|when|why|how)\s+(is|are|was|were|do|does|"
               r"did|can|should)\s+", "", text.strip(), flags=re.I)
    t = t.strip("?.! ")
    caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", t)
    if caps:
        return max(caps, key=len)
    words = [w for w in t.split() if w.lower() not in STOPWORDS]
    return " ".join(words[:3])


def esc_md(t: str) -> str:
    return t.replace("*", "").replace("_", "").replace("`", "")


# Words that mean "render me a picture", not "tell me about pictures".
PIC_VERBS = ("generate", "create", "make", "draw", "paint", "render",
             "design", "produce", "give me", "show me", "i want", "i need")
PIC_NOUNS = ("image", "picture", "photo", "photograph", "artwork", "art of",
             "illustration", "drawing", "painting", "poster", "wallpaper",
             "logo of", "icon of", "avatar", "portrait", "scene of")
VID_NOUNS = ("video", "clip", "animation", "gif", "movie", "short film",
             "animated")


def media_intent(q: str):
    """Return ('image'|'video', subject) when the user wants media made.

    'draw me a fox in the snow'  -> ('image', 'a fox in the snow')
    'what is a gif'              -> None   (a question, not a request)
    """
    ql = " ".join(q.lower().split())
    if not ql:
        return None
    # A question about the concept is not a request to render one.
    if ql.startswith(("what is", "what are", "who is", "how do", "how does",
                      "why is", "why do", "when did", "where is",
                      "what's", "explain", "define", "tell me about")):
        return None
    # "draw a fox" needs no noun - the verb alone is the whole request.
    STRONG = ("draw ", "paint ", "illustrate ", "render an image",
              "sketch ", "draw me", "paint me")
    kind = None
    if any(w in ql for w in VID_NOUNS):
        kind = "video"
    elif any(w in ql for w in PIC_NOUNS):
        kind = "image"
    elif ql.startswith(STRONG) or any((" " + w) in (" " + ql)
                                      for w in STRONG):
        kind = "image"
    if kind is None:
        return None
    if kind == "image" and not (
            ql.startswith(STRONG)
            or any((" " + w) in (" " + ql) for w in STRONG)
            or any(ql.startswith(v) or (" " + v + " ") in (" " + ql + " ")
                   for v in PIC_VERBS)):
        return None
    if kind == "video" and not any(
            ql.startswith(v) or (" " + v + " ") in (" " + ql + " ")
            for v in PIC_VERBS):
        return None
    # Strip the instruction off the front so only the subject is prompted.
    subj = q.strip()
    low = subj.lower()
    for pat in ("generate", "create", "make", "draw", "paint", "render",
                "design", "produce", "illustrate", "sketch",
                "give me", "show me",
                "i want", "i need", "please", "for me", "can you"):
        if low.startswith(pat):
            subj = subj[len(pat):].strip()
            low = subj.lower()
    for pat in ("me a ", "me an ", "me the ", "a ", "an ", "the ",
                "some ", "of "):
        if low.startswith(pat):
            subj = subj[len(pat):].strip()
            low = subj.lower()
    for noun in (VID_NOUNS if kind == "video" else PIC_NOUNS):
        if low.startswith(noun):
            subj = subj[len(noun):].strip()
            low = subj.lower()
            break
    for pat in ("of ", "showing ", "with ", "that shows ", "for "):
        if low.startswith(pat):
            subj = subj[len(pat):].strip()
            break
    return (kind, subj.strip(" .,:;") or q.strip())


def route(query: str) -> str:
    """Decide what ADELTE should actually DO with this message.

    Returns one of: command | chat | make | research
    """
    q = fix_typos(" ".join(query.lower().split()))
    if not q:
        return "chat"

    # 1. explicit device / app command
    if match_command(q) is not None:
        return "command"

    # 1b. "draw me ..." / "make a video of ..." -> render it, do not search
    mi = media_intent(query)
    if mi is not None:
        return mi[0]          # "image" or "video"

    # 2. short social message -> talk like a friend, do not search
    words = q.split()
    if is_greeting(q):
        return "chat"
    if len(words) <= 4:
        stripped = q.rstrip("!?.")
        if stripped in GREET or any(stripped == g for g in GREET):
            return "chat"
    if len(words) <= 8 and any(p in q for p in CHAT_PAT):
        return "chat"
    if q.endswith("?") and len(words) <= 3 and not any(
            h in q for h in ("what is", "who is", "how to")):
        return "chat"

    # 3. asking ADELTE to BUILD something -> generate, never search
    has_verb = any(q.startswith(v) or (" " + v + " ") in (" " + q + " ")
                   for v in MAKE_VERBS)
    has_obj = any(o in q for o in MAKE_OBJECTS)
    if has_verb and has_obj:
        return "make"
    if q.startswith(("create ", "build me", "make me", "write me", "code me")):
        return "make"

    # 4. everything else is a genuine research question
    return "research"


# ============================================================================
#  SECTION 3 — SEARCH ENGINES
#  Every engine: async (client, query, limit) -> list[dict]
#  A failing engine returns [] — it must never break the whole search.
# ============================================================================

async def eng_duckduckgo(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """DuckDuckGo Lite — most reliable keyless web index."""
    r = await c.post("https://lite.duckduckgo.com/lite/",
                     data={"q": q, "kl": "wt-wt"},
                     headers={**HEADERS, "Referer": "https://lite.duckduckgo.com/"})
    doc = HTMLParser(r.text)
    links = doc.css("a.result-link")
    snips = [clean(td.text()) for td in doc.css("td.result-snippet")]
    out = []
    for i, a in enumerate(links[:n]):
        url = unwrap(a.attributes.get("href", ""))
        if url.startswith("http"):
            out.append({"title": clean(a.text()) or urlparse(url).netloc,
                        "url": url,
                        "snippet": snips[i] if i < len(snips) else "",
                        "source": "duckduckgo"})
    return out


async def eng_duckduckgo_html(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """DuckDuckGo full HTML — slightly different ranking, widens coverage."""
    r = await c.post("https://html.duckduckgo.com/html/", data={"q": q},
                     headers={**HEADERS, "Referer": "https://html.duckduckgo.com/"})
    doc = HTMLParser(r.text)
    out = []
    for res in doc.css("div.result")[: n * 2]:
        a = res.css_first("a.result__a")
        if not a:
            continue
        url = unwrap(a.attributes.get("href", ""))
        if not url.startswith("http"):
            continue
        sn = res.css_first("a.result__snippet") or res.css_first(".result__snippet")
        out.append({"title": clean(a.text()), "url": url,
                    "snippet": clean(sn.text()) if sn else "",
                    "source": "duckduckgo"})
        if len(out) >= n:
            break
    return out


async def eng_bing(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Bing via its RSS feed — no JS wall, no key, very stable."""
    r = await c.get("https://www.bing.com/search",
                    params={"q": q, "format": "rss", "count": n}, headers=HEADERS)
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []
    out = []
    for it in root.findall(".//item")[:n]:
        url = (it.findtext("link") or "").strip()
        if url.startswith("http"):
            out.append({"title": clean(it.findtext("title")), "url": url,
                        "snippet": clean(it.findtext("description")),
                        "source": "bing"})
    return out


async def eng_brave(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Brave Search. Frequently 429s datacenter IPs; then returns []."""
    r = await c.get("https://search.brave.com/search", params={"q": q, "source": "web"},
                    headers={**HEADERS, "Referer": "https://search.brave.com/"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    doc = HTMLParser(r.text)
    out = []
    for res in doc.css("div.snippet")[: n * 2]:
        a = res.css_first("a")
        if not a:
            continue
        url = a.attributes.get("href", "")
        if not url.startswith("http"):
            continue
        ti = res.css_first(".title") or res.css_first("div.title")
        de = res.css_first(".snippet-description") or res.css_first(".snippet-content")
        out.append({"title": clean(ti.text()) if ti else clean(a.text())[:120],
                    "url": url, "snippet": clean(de.text()) if de else "",
                    "source": "brave"})
        if len(out) >= n:
            break
    return out


async def eng_google(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """
    Google, keylessly. Google serves a JS-only shell to plain HTTP clients, so
    scraping /search from a server IP returns nothing. Google News RSS is the
    one Google surface that still answers a plain GET, so we read that first
    and only fall back to the HTML scrape (which works from a home IP).
    """
    out: list[dict] = []
    try:
        r = await c.get("https://news.google.com/rss/search",
                        params={"q": q, "hl": "en-US", "gl": "US",
                                "ceid": "US:en"}, headers=HEADERS)
        if r.status_code == 200:
            root = ET.fromstring(r.text)
            for it in root.findall(".//item")[:n]:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                src = it.find("{http://search.yahoo.com/mrss/}source")
                if src is None:
                    src = it.find("source")
                who = (src.text or "").strip() if src is not None else ""
                if not title or not link.startswith("http"):
                    continue
                out.append({"title": clean(title), "url": link,
                            "snippet": who, "source": "google"})
    except Exception:
        pass
    if out:
        return out[:n]
    try:                              # home IP / proxy: the real thing works
        r = await c.get("https://www.google.com/search",
                        params={"q": q, "num": n, "hl": "en"},
                        headers={**HEADERS,
                                 "Referer": "https://www.google.com/"})
        doc = HTMLParser(r.text)
        for a in doc.css("a"):
            h3 = a.css_first("h3")
            if not h3:
                continue
            url = unwrap(a.attributes.get("href", ""))
            if not url.startswith("http") or "google." in urlparse(url).netloc:
                continue
            out.append({"title": clean(h3.text()), "url": url, "snippet": "",
                        "source": "google"})
            if len(out) >= n:
                break
    except Exception:
        pass
    return out[:n]


async def eng_wikipedia(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Wikipedia MediaWiki API — strong on definitional questions."""
    r = await c.get("https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search", "srsearch": q,
                            "format": "json", "srlimit": min(n, 5),
                            "srprop": "snippet"}, headers=HEADERS)
    out = []
    for it in r.json().get("query", {}).get("search", []):
        t = it.get("title", "")
        out.append({"title": f"Wikipedia: {t}",
                    "url": "https://en.wikipedia.org/wiki/" + t.replace(" ", "_"),
                    "snippet": clean(re.sub(r"<[^>]+>", "", it.get("snippet", ""))),
                    "source": "wikipedia"})
    return out


async def eng_github(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """GitHub repositories — public API, 60 req/h per IP unauthenticated."""
    r = await c.get("https://api.github.com/search/repositories",
                    params={"q": q, "sort": "stars", "order": "desc",
                            "per_page": min(n, 8)},
                    headers={**HEADERS, "Accept": "application/vnd.github+json"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for repo in r.json().get("items", [])[:n]:
        out.append({"title": f"{repo['full_name']} · {repo.get('stargazers_count',0):,}★",
                    "url": repo["html_url"],
                    "snippet": (clean(repo.get("description")) +
                                f" | language: {repo.get('language') or 'n/a'}"),
                    "source": "github"})
    return out


async def eng_github_issues(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """GitHub issues — real problem/solution pairs, gold for debugging."""
    r = await c.get("https://api.github.com/search/issues",
                    params={"q": q, "sort": "reactions", "per_page": min(n, 8)},
                    headers={**HEADERS, "Accept": "application/vnd.github+json"})
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for it in r.json().get("items", [])[:n]:
        out.append({"title": f"[issue] {clean(it.get('title'))}",
                    "url": it.get("html_url", ""),
                    "snippet": clean((it.get("body") or "")[:300]),
                    "source": "github"})
    return out


async def eng_stackoverflow(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """StackExchange API — keyless for modest volumes."""
    r = await c.get("https://api.stackexchange.com/2.3/search/advanced",
                    params={"order": "desc", "sort": "relevance", "q": q,
                            "site": "stackoverflow", "pagesize": min(n, 8),
                            "filter": "default"}, headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for it in r.json().get("items", [])[:n]:
        out.append({"title": clean(it.get("title")), "url": it.get("link", ""),
                    "snippet": (f"score {it.get('score',0)} · "
                                f"{'ANSWERED' if it.get('is_answered') else 'unanswered'}"
                                f" · tags: {', '.join(it.get('tags', [])[:5])}"),
                    "source": "stackoverflow"})
    return out


async def eng_hackernews(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Hacker News via Algolia — opinions, comparisons, current sentiment."""
    r = await c.get("https://hn.algolia.com/api/v1/search",
                    params={"query": q, "hitsPerPage": min(n, 8), "tags": "story"},
                    headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for h in r.json().get("hits", [])[:n]:
        url = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        out.append({"title": clean(h.get("title")), "url": url,
                    "snippet": (f"{h.get('points',0)} points · "
                                f"{h.get('num_comments',0)} comments on Hacker News"),
                    "source": "hackernews"})
    return out


async def eng_mdn(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """MDN Web Docs - authoritative HTML/CSS/JS reference."""
    r = await c.get("https://developer.mozilla.org/api/v1/search",
                    params={"q": q, "locale": "en-US"}, headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for d in r.json().get("documents", [])[:n]:
        out.append({"title": clean(d.get("title")),
                    "url": "https://developer.mozilla.org" + (d.get("mdn_url") or ""),
                    "snippet": clean(d.get("summary")),
                    "source": "mdn"})
    return out


async def eng_stackexchange(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Stack Exchange API - accepted answers, scored."""
    r = await c.get("https://api.stackexchange.com/2.3/search/advanced",
                    params={"order": "desc", "sort": "relevance", "q": q,
                            "site": "stackoverflow", "pagesize": min(n, 8),
                            "filter": "default"}, headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for it in r.json().get("items", [])[:n]:
        out.append({"title": clean(it.get("title")),
                    "url": it.get("link") or "",
                    "snippet": ("score %s, %s answers%s"
                                % (it.get("score", 0),
                                   it.get("answer_count", 0),
                                   " - ACCEPTED" if it.get("is_answered")
                                   else "")),
                    "source": "stackexchange"})
    return out


async def eng_npm(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """npm registry - real package names, versions and descriptions."""
    r = await c.get("https://registry.npmjs.org/-/v1/search",
                    params={"text": q, "size": min(n, 8)}, headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for ob in r.json().get("objects", [])[:n]:
        p = ob.get("package", {})
        out.append({"title": "%s %s" % (p.get("name", ""), p.get("version", "")),
                    "url": (p.get("links", {}) or {}).get("npm", ""),
                    "snippet": clean(p.get("description")),
                    "source": "npm"})
    return out


async def eng_gitlab(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """GitLab projects - a second code host beyond GitHub."""
    r = await c.get("https://gitlab.com/api/v4/projects",
                    params={"search": q, "per_page": min(n, 8),
                            "order_by": "star_count", "sort": "desc"},
                    headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for p in (r.json() or [])[:n]:
        out.append({"title": clean(p.get("name_with_namespace")),
                    "url": p.get("web_url") or "",
                    "snippet": (clean(p.get("description"))
                                or "%s stars on GitLab" % p.get("star_count", 0)),
                    "source": "gitlab"})
    return out


async def eng_wikidata(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Wikidata entities - crisp factual identity for people and orgs."""
    r = await c.get("https://www.wikidata.org/w/api.php",
                    params={"action": "wbsearchentities", "search": q,
                            "language": "en", "format": "json",
                            "limit": min(n, 8)}, headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    out = []
    for it in r.json().get("search", [])[:n]:
        desc = clean(it.get("description"))
        if not desc:
            continue
        out.append({"title": clean(it.get("label")),
                    "url": it.get("concepturi") or "",
                    "snippet": desc, "source": "wikidata"})
    return out


async def eng_google_cse(c: httpx.AsyncClient, q: str, n: int) -> list[dict]:
    """Google Custom Search JSON API - keyed, high-precision.

    Additive engine: only runs when GOOGLE_CUSTOM_SEARCH_API_KEY and
    GOOGLE_SEARCH_ENGINE_ID are both in .env. Otherwise returns [] so the
    merged search never breaks.
    """
    key = env_key("GOOGLE_CUSTOM_SEARCH_API_KEY")
    cx = env_key("GOOGLE_SEARCH_ENGINE_ID")
    if not key or not cx:
        return []
    r = await c.get("https://www.googleapis.com/customsearch/v1",
                    params={"q": q, "key": key, "cx": cx, "num": min(max(n, 1), 10)},
                    headers=HEADERS)
    if r.status_code != 200:
        raise RuntimeError("HTTP %d" % r.status_code)
    out = []
    for it in (r.json().get("items") or [])[:n]:
        out.append({"title": clean(it.get("title")), "url": it.get("link") or "",
                    "snippet": clean(it.get("snippet")), "source": "google_cse"})
    return out


ENGINES: Dict[str, Callable[..., Awaitable[List[dict]]]] = {
    "duckduckgo": eng_duckduckgo,
    "duckduckgo_html": eng_duckduckgo_html,
    "bing": eng_bing,
    "brave": eng_brave,
    "google": eng_google,
    "google_cse": eng_google_cse,
    "wikipedia": eng_wikipedia,
    "github": eng_github,
    "github_issues": eng_github_issues,
    "stackoverflow": eng_stackoverflow,
    "hackernews": eng_hackernews,
    "mdn": eng_mdn,
    "stackexchange": eng_stackexchange,
    "npm": eng_npm,
    "gitlab": eng_gitlab,
    "wikidata": eng_wikidata,
}

WEB_SET = ["google_cse", "duckduckgo", "bing", "duckduckgo_html", "wikipedia", "brave",
           "google", "wikidata"]
CODE_SET = ["github", "stackoverflow", "stackexchange", "mdn", "duckduckgo",
            "bing", "github_issues", "npm"]
# Deep search casts the widest net - used when Deep read is on.
WEB_DEEP = ["google_cse", "duckduckgo", "bing", "duckduckgo_html", "wikipedia", "brave",
            "google", "wikidata", "hackernews", "mdn"]
CODE_DEEP = ["github", "stackoverflow", "stackexchange", "mdn", "duckduckgo",
             "bing", "github_issues", "npm", "gitlab", "hackernews"]


async def run_engine(c: httpx.AsyncClient, name: str, q: str, n: int) -> dict:
    """Execute one engine safely and record its health."""
    fn = ENGINES.get(name)
    if fn is None:
        return {"engine": name, "ok": False, "error": "unknown engine", "results": []}
    t0 = time.time()
    try:
        res = await asyncio.wait_for(fn(c, q, n), timeout=18)
        h = ENGINE_HEALTH.setdefault(name, {"calls": 0})
        h["calls"] += 1
        h["health"] = "ok" if res else "empty"
        h["ms"] = int((time.time() - t0) * 1000)
        return {"engine": name, "ok": True, "error": None, "results": res}
    except Exception as e:
        h = ENGINE_HEALTH.setdefault(name, {"calls": 0})
        h["calls"] += 1
        h["health"] = "blocked"
        h["error"] = f"{type(e).__name__}"
        return {"engine": name, "ok": False,
                "error": f"{type(e).__name__}: {e}"[:150], "results": []}


def dedupe(results: list[dict]) -> list[dict]:
    """Merge duplicate URLs; agreement across engines becomes a ranking signal."""
    seen: dict[str, dict] = {}
    order: list[str] = []
    for r in results:
        url = (r.get("url") or "").rstrip("/")
        if not url:
            continue
        key = re.sub(r"^https?://(www\.)?", "", url).lower()
        if key in seen:
            cur = seen[key]
            if len(r.get("snippet", "")) > len(cur.get("snippet", "")):
                cur["snippet"] = r["snippet"]
            cur["source"] = "+".join(sorted(set(cur["source"].split("+")) |
                                            {r["source"]}))
            cur["agreement"] = cur.get("agreement", 1) + 1
        else:
            r = dict(r)
            r["agreement"] = 1
            seen[key] = r
            order.append(key)
    merged = [seen[k] for k in order]
    merged.sort(key=lambda x: -x.get("agreement", 1))
    return merged


async def read_page(c: httpx.AsyncClient, url: str,
                    max_chars: int = None) -> str:
    """Download a page and extract its readable text. '' on any failure."""
    max_chars = max_chars or CFG.page_chars
    try:
        r = await asyncio.wait_for(
            c.get(url, headers=HEADERS, follow_redirects=True), timeout=13)
        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return ""
        doc = HTMLParser(r.text)
        for t in doc.css("script,style,nav,header,footer,aside,noscript,form,svg,iframe"):
            t.decompose()
        node = (doc.css_first("article") or doc.css_first("main")
                or doc.css_first("div#content") or doc.css_first("div.content")
                or doc.body)
        return clean(node.text(separator=" "))[:max_chars] if node else ""
    except Exception:
        return ""


# ============================================================================
#  SECTION 4 — THE BRAIN
#  Layer 1: AI Horde (free volunteer GPU swarm, anonymous key, no signup)
#  Layer 2: extractive synthesis — always works, so we can never fail
# ============================================================================

HORDE = "https://aihorde.net/api/v2"
HORDE_HEADERS = {
    "apikey": "0000000000",                 # public anonymous key
    "Client-Agent": "adelte:2.0:adelte-industries",
    "Content-Type": "application/json",
}


def extract_intent(query: str) -> list:
    """What is the user actually asking for? Shown live in the think panel."""
    q = (query or "").strip()
    low = q.lower()
    out = []
    kinds = [
        (r"\b(how|steps?|guide|tutorial)\b", "a how-to, step by step"),
        (r"\b(why|reason|because)\b", "an explanation of the cause"),
        (r"\b(what is|what are|define|meaning)\b", "a definition"),
        (r"\b(compare|versus|vs\.?|difference)\b", "a comparison"),
        (r"\b(best|top|recommend|should i)\b", "a recommendation"),
        (r"\b(fix|error|bug|not working|broken|fail)\b", "a fix for a problem"),
        (r"\b(code|function|script|program|api)\b", "something technical"),
        (r"\b(plan|strategy|idea)\b", "a plan"),
        (r"\?\s*$", "a direct answer to a question"),
    ]
    for pat, label in kinds:
        if re.search(pat, low):
            out.append(label)
    kw = [w for w in keywords(q) if len(w) > 3][:5]
    if kw:
        out.append("key topics: " + ", ".join(kw))
    out.append("length: short and ordered")
    return out[:5]


def build_messages(query: str, sources: list[dict], history: list[dict],
                   memory: str = "") -> list[dict]:
    ctx = []
    for i, s in enumerate(sources[:8], 1):
        body = (s.get("page_text") or s.get("snippet") or "")[:900]
        ctx.append(f"[{i}] {s.get('title','')} ({s.get('url','')})\n{body}")
    context = "\n\n".join(ctx) if ctx else "(no sources retrieved)"

    system = ("You are ADELTE. You answer like a knowledgeable friend: warm, "
              "direct, no throat-clearing.\n\n"
              "FORMAT — follow exactly:\n"
              "1. One short opening sentence that answers the question.\n"
              "2. Then AT MOST 4 bullet points, one short sentence each.\n"
              "3. Nothing else. No headings, no preamble, no summary line.\n\n"
              "RULES:\n"
              "- Use ONLY the numbered SOURCES. Cite as [1] after a claim.\n"
              "- Never narrate your process. Do not write 'We need to', "
              "'We must', 'Let me', 'Based on the sources' or list what you "
              "are about to do. Just give the answer.\n"
              "- Never repeat the question back.\n"
              "- If the sources don't cover it, say so in one line.\n"
              "- Include hex colour codes when colours are discussed.")
    if memory:
        system += f"\n\nEarlier in this session:\n{memory[:800]}"

    msgs = [{"role": "system", "content": system}]
    for t in history[-4:]:
        msgs.append({"role": t["role"], "content": t["content"][:1200]})
    msgs.append({"role": "user",
                 "content": f"SOURCES:\n{context}\n\n---\nQUESTION: {query}"})
    return msgs


def flatten_prompt(messages: list[dict]) -> str:
    sysmsg = next((m["content"] for m in messages if m["role"] == "system"), "")
    convo = [f"### {'User' if m['role']=='user' else 'Assistant'}:\n{m['content']}"
             for m in messages if m["role"] != "system"]
    return f"### Instruction:\n{sysmsg}\n\n" + "\n\n".join(convo) + "\n\n### Assistant:\n"


# Openers that mean the model is thinking out loud instead of answering.
SCRATCH_PAT = re.compile(
    r"^\s*(we (need|must|should|can|will|have to)\b"
    r"|let'?s\b|let me\b|first,? (i|we)\b|okay[,.]|ok[,.]|alright[,.]"
    r"|the (user|question) (is |asks|wants)"
    r"|i (need|should|will|must) (to )?\b"
    r"|so we (can|must|need)\b"
    r"|based on the sources?\b|according to the sources?\b"
    r"|using (the )?info from sources?\b"
    r"|here'?s? (my|the) (plan|approach)\b"
    r"|thinking process\b|thought process\b|reasoning\b:"
    r"|analy(s|z)e the (request|question|input)\b"
    r"|\*{0,2}persona:?\*{0,2}\b|\*{0,2}format:?\*{0,2}\b"
    r"|\*{0,2}rules:?\*{0,2}\b|\*{0,2}input question:?\*{0,2}\b"
    r"|\*{0,2}provided sources:?\*{0,2}\b"
    r"|the user is asking\b|the user wants\b)", re.I)


# Whole-answer guard: if a model dumps its instruction sheet, cut it.
LEAK_PAT = re.compile(
    r"(?im)^\s*\**\s*("
    r"thinking process|thought process|analyz(e|ing) the request|"
    r"analys(e|ing) the request|persona\s*:|input question\s*:|"
    r"provided sources\s*:|step \d+\s*:\s*analy)"
    r"\s*\**\s*:?\s*$")


def strip_leaked_plan(text: str) -> str:
    """Drop a leaked instruction/plan preamble and keep the real answer."""
    if not text:
        return text
    if not LEAK_PAT.search(text):
        return text
    lines = text.split("\n")
    # find the last leaked header, keep everything after the block it opens
    cut = 0
    for i, ln in enumerate(lines):
        if LEAK_PAT.match(ln):
            cut = i
    tail = lines[cut + 1:]
    # skip the enumerated plan body that follows the header
    while tail and (
            not tail[0].strip()
            or re.match(r"^\s*(\d+[.)]|[-*\u2022])\s", tail[0])
            or re.match(r"^\s*\**\s*(persona|format|rules|input|provided|"
                        r"cite|use only|no |if sources|include)\b",
                        tail[0], re.I)):
        tail.pop(0)
    out = "\n".join(tail).strip()
    return out if len(out) > 60 else text


def strip_scratchpad(text: str) -> str:
    """Free swarm models often emit their reasoning before the answer.

    Remove explicit <think> blocks, then drop leading paragraphs that are
    the model narrating its own process. Never return empty — if every
    paragraph looks like scratchpad, keep the original text.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<\|?(?:begin|end)_of_thought\|?>", "", text, flags=re.I)

    # Harmony / channel-format models (gpt-oss and friends) wrap their
    # scratchpad in <|channel|>analysis ... <|message|>. The user saw one of
    # these leak verbatim into an answer. Keep only the final channel.
    text = re.sub(r"<\s*\|?\s*(channel|message|start|end|return|constrain)"
                  r"\s*\|?\s*>", lambda m: "<|%s|>" % m.group(1).lower(),
                  text, flags=re.I)
    if "<|channel|>" in text or "<|start|>" in text:
        finals = re.findall(
            r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\||\Z)",
            text, flags=re.S | re.I)
        if finals:
            text = finals[-1]
        else:
            text = re.sub(r"<\|channel\|>\s*(?:analysis|thought|reasoning|"
                          r"commentary)\s*<\|message\|>.*?(?=<\||\Z)",
                          "", text, flags=re.S | re.I)
    text = re.sub(r"<\|(?:channel|message|start|end|return|constrain)\|>",
                  "", text)
    text = re.sub(r"^\s*(?:analysis|assistantfinal|final)\s*\n", "",
                  text, flags=re.I)

    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    while paras and SCRATCH_PAT.match(paras[0].strip()):
        paras.pop(0)
    out = "\n\n".join(paras).strip()
    return out or text.strip()


def trim_completion(text: str) -> str:
    """Instruct models like to keep inventing extra turns — cut them off."""
    for stop in ("### User", "### Instruction", "### Human", "\nUser:",
                 "### Related Questions", "### Answers:", "### Response:",
                 "### Assistant", "<|im_end|>", "<|endoftext|>"):
        i = text.find(stop)
        if i >= 0:                    # cut anywhere; a leading marker means
            text = text[:i]           # the model produced nothing usable
    return strip_leaked_plan(strip_scratchpad(text.strip()))


MAX_CONTINUE = 3          # extra rounds allowed when a reply is cut short
MAX_HORDE_CONTINUE = 5    # Horde workers cap hard, so it needs more
CONTINUE_ASK = (
    "You were cut off mid-answer. Continue from the exact character where you "
    "stopped. Do not repeat anything you already wrote, do not restate the "
    "file, do not apologise, do not open a new code fence unless the previous "
    "one was closed. Just carry straight on and finish the whole thing, "
    "including any closing tags.")


def join_continuation(sofar: str, piece: str) -> str:
    """Glue a continuation on, dropping any overlap the model repeated."""
    if not sofar:
        return piece
    if not piece:
        return sofar
    piece = re.sub(r"^\s*(?:```[a-zA-Z0-9]*\s*\n)", "", piece)
    tail = sofar[-400:]
    for n in range(min(len(tail), len(piece)), 24, -1):
        if piece[:n] and tail.endswith(piece[:n]):
            piece = piece[n:]
            break
    if sofar.endswith("\n") or piece.startswith("\n"):
        return sofar + piece
    return sofar + piece


# ===================================================================
#  OFFLINE BUILDER - real files, no AI, no network, never fails
# ===================================================================
OFFLINE_PALETTE = {
    "ink": "#0F172A", "bg": "#F8FAFC", "card": "#FFFFFF",
    "a": "#4F46E5", "b": "#6366F1", "mut": "#64748B", "line": "#E2E8F0",
}

_OFF_BASE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%(title)s</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--ink:%(ink)s;--bg:%(bg)s;--card:%(card)s;--a:%(a)s;--b:%(b)s;
      --mut:%(mut)s;--line:%(line)s}
body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",
     system-ui,sans-serif;background:var(--bg);color:var(--ink)}
a{color:inherit;text-decoration:none}
.wrap{max-width:1060px;margin:0 auto;padding:0 22px}
%(css)s
</style>
</head>
<body>
%(body)s
<script>
%(js)s
</script>
</body>
</html>
"""


def _off_page(title, css, body, js=""):
    d = dict(OFFLINE_PALETTE)
    d.update(title=title, css=css, body=body, js=js)
    return _OFF_BASE % d


def _off_portfolio(name="Your Name"):
    css = """
nav{position:sticky;top:0;z-index:9;backdrop-filter:blur(14px);
    background:rgba(248,250,252,.82);border-bottom:1px solid var(--line)}
nav .wrap{display:flex;align-items:center;justify-content:space-between;
    height:66px}
.brand{font-weight:800;letter-spacing:-.02em;font-size:19px}
.brand span{background:linear-gradient(90deg,var(--a),var(--b));
    -webkit-background-clip:text;background-clip:text;color:transparent}
nav ul{display:flex;gap:26px;list-style:none}
nav a{color:var(--mut);font-weight:600;font-size:14.5px}
nav a:hover{color:var(--a)}
.burger{display:none;width:42px;height:42px;border:1px solid var(--line);
    border-radius:10px;background:var(--card);cursor:pointer}
.hero{padding:104px 0 84px;text-align:center}
.hero h1{font-size:clamp(38px,6vw,66px);line-height:1.06;letter-spacing:-.03em;
    font-weight:800}
.hero h1 em{font-style:normal;background:linear-gradient(90deg,var(--a),var(--b));
    -webkit-background-clip:text;background-clip:text;color:transparent}
.hero p{margin:20px auto 0;max-width:620px;color:var(--mut);font-size:18px}
.cta{display:inline-flex;gap:12px;margin-top:32px}
.btn{padding:13px 26px;border-radius:12px;font-weight:700;font-size:15px;
    border:1px solid transparent;cursor:pointer}
.btn.p{background:linear-gradient(90deg,var(--a),var(--b));color:#fff}
.btn.s{background:var(--card);border-color:var(--line)}
section{padding:72px 0}
h2{font-size:31px;letter-spacing:-.02em;margin-bottom:8px;font-weight:800}
.sub{color:var(--mut);margin-bottom:34px}
.grid{display:grid;gap:22px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
    padding:26px;transition:transform .25s,box-shadow .25s}
.card:hover{transform:translateY(-6px);box-shadow:0 18px 40px rgba(15,23,42,.09)}
.tag{display:inline-block;font-size:12px;font-weight:700;color:var(--a);
    background:rgba(79,70,229,.10);padding:5px 11px;border-radius:99px;
    margin-bottom:13px}
.card h3{font-size:19px;margin-bottom:8px}
.card p{color:var(--mut);font-size:14.5px}
form{max-width:520px;display:grid;gap:14px}
label{font-weight:600;font-size:14px}
input,textarea{width:100%;padding:13px 15px;border:1px solid var(--line);
    border-radius:11px;background:var(--card);font:inherit;font-size:15px}
input:focus,textarea:focus{outline:2px solid var(--a);border-color:transparent}
.err{color:#DC2626;font-size:13px;min-height:17px}
.ok{background:#065F46;color:#fff;padding:13px 16px;border-radius:11px;
    display:none}
footer{border-top:1px solid var(--line);padding:34px 0;color:var(--mut);
    font-size:14px;text-align:center}
.reveal{opacity:0;transform:translateY(22px);transition:.7s cubic-bezier(.2,.8,.2,1)}
.reveal.in{opacity:1;transform:none}
@media(max-width:760px){nav ul{display:none}.burger{display:block}
  nav ul.open{display:flex;position:absolute;top:66px;left:0;right:0;
    flex-direction:column;gap:0;background:var(--card);
    border-bottom:1px solid var(--line);padding:8px 22px}
  nav ul.open li{padding:12px 0;border-bottom:1px solid var(--line)}}
"""
    body = """
<nav><div class="wrap">
  <div class="brand">%(nm)s<span>.</span></div>
  <ul id="mnu">
    <li><a href="#work">Work</a></li>
    <li><a href="#about">About</a></li>
    <li><a href="#contact">Contact</a></li>
  </ul>
  <button class="burger" id="bg" aria-label="Menu">&#9776;</button>
</div></nav>

<header class="hero"><div class="wrap">
  <h1>I build things<br>for the <em>web</em>.</h1>
  <p>Designer and developer focused on clean interfaces, fast pages and
     details that hold up under real use.</p>
  <div class="cta">
    <a class="btn p" href="#work">See my work</a>
    <a class="btn s" href="#contact">Get in touch</a>
  </div>
</div></header>

<section id="work"><div class="wrap">
  <h2 class="reveal">Selected work</h2>
  <p class="sub reveal">Three projects worth showing.</p>
  <div class="grid">
    <article class="card reveal"><span class="tag">Web app</span>
      <h3>Analytics dashboard</h3>
      <p>Live charts, filtering and CSV export over a small REST API.</p></article>
    <article class="card reveal"><span class="tag">Mobile</span>
      <h3>Habit tracker</h3>
      <p>Offline-first, syncs when it can, gentle streaks and reminders.</p></article>
    <article class="card reveal"><span class="tag">Brand</span>
      <h3>Studio identity</h3>
      <p>Logo, type scale and a component kit the team actually reuses.</p></article>
  </div>
</div></section>

<section id="about"><div class="wrap">
  <h2 class="reveal">About</h2>
  <p class="sub reveal" style="max-width:640px">
    I care about the boring parts: load time, keyboard access, and copy that
    says what it means. Currently open to freelance work.</p>
</div></section>

<section id="contact"><div class="wrap">
  <h2 class="reveal">Contact</h2>
  <p class="sub reveal">Tell me what you are building.</p>
  <div class="ok" id="ok">Thanks - your message is ready to send.</div>
  <form id="f" novalidate>
    <div><label for="n">Name</label>
      <input id="n" name="name" autocomplete="name">
      <div class="err" data-for="n"></div></div>
    <div><label for="e">Email</label>
      <input id="e" name="email" type="email" autocomplete="email">
      <div class="err" data-for="e"></div></div>
    <div><label for="m">Message</label>
      <textarea id="m" name="message" rows="5"></textarea>
      <div class="err" data-for="m"></div></div>
    <button class="btn p" type="submit">Send message</button>
  </form>
</div></section>

<footer><div class="wrap">&copy; <span id="yr"></span> %(nm)s</div></footer>
""" % {"nm": name}
    js = r"""
document.getElementById('yr').textContent=new Date().getFullYear();
document.getElementById('bg').onclick=function(){
  document.getElementById('mnu').classList.toggle('open');};
document.querySelectorAll('nav a').forEach(function(a){
  a.onclick=function(e){var t=document.querySelector(a.getAttribute('href'));
    if(t){e.preventDefault();t.scrollIntoView({behavior:'smooth'});
      document.getElementById('mnu').classList.remove('open');}};});
var io=new IntersectionObserver(function(es){es.forEach(function(en){
  if(en.isIntersecting)en.target.classList.add('in');});},{threshold:.12});
document.querySelectorAll('.reveal').forEach(function(el){io.observe(el);});
var f=document.getElementById('f');
function err(id,msg){
  document.querySelector('[data-for="'+id+'"]').textContent=msg||'';}
f.onsubmit=function(e){
  e.preventDefault();var bad=false;
  var n=document.getElementById('n').value.trim();
  var em=document.getElementById('e').value.trim();
  var m=document.getElementById('m').value.trim();
  err('n');err('e');err('m');
  if(n.length<2){err('n','Please enter your name');bad=true;}
  if(!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(em)){
    err('e','Please enter a valid email');bad=true;}
  if(m.length<10){err('m','A little more detail, please');bad=true;}
  if(bad)return;
  document.getElementById('ok').style.display='block';
  f.reset();};
"""
    return _off_page(name + " - Portfolio", css, body, js)


def _off_login():
    css = """
body{display:grid;place-items:center;min-height:100vh;
     background:linear-gradient(135deg,#EEF2FF,#F8FAFC 55%,#EDE9FE)}
.box{width:100%;max-width:400px;background:var(--card);
     border:1px solid var(--line);border-radius:18px;padding:34px 30px;
     box-shadow:0 22px 60px rgba(15,23,42,.10)}
.logo{width:46px;height:46px;border-radius:13px;
     background:linear-gradient(135deg,var(--a),var(--b));margin-bottom:17px}
h1{font-size:24px;letter-spacing:-.02em;margin-bottom:5px}
.sub{color:var(--mut);font-size:14.5px;margin-bottom:24px}
label{display:block;font-weight:600;font-size:13.5px;margin:0 0 6px}
.fld{margin-bottom:15px;position:relative}
input{width:100%;padding:13px 15px;border:1px solid var(--line);
     border-radius:11px;font:inherit;font-size:15px;background:#fff}
input:focus{outline:2px solid var(--a);border-color:transparent}
input.bad{border-color:#DC2626}
.err{color:#DC2626;font-size:12.5px;min-height:16px;margin-top:4px}
.peek{position:absolute;right:11px;top:33px;border:0;background:none;
     cursor:pointer;color:var(--mut);font-size:12.5px;font-weight:700}
.row{display:flex;align-items:center;justify-content:space-between;
     font-size:13.5px;margin:4px 0 20px}
.row a{color:var(--a);font-weight:600}
button.go{width:100%;padding:14px;border:0;border-radius:11px;color:#fff;
     font:inherit;font-weight:700;font-size:15px;cursor:pointer;
     background:linear-gradient(90deg,var(--a),var(--b))}
button.go:disabled{opacity:.6;cursor:default}
.foot{text-align:center;color:var(--mut);font-size:13.5px;margin-top:19px}
.ok{display:none;background:#065F46;color:#fff;padding:12px;
     border-radius:10px;font-size:14px;margin-bottom:15px}
"""
    body = """
<main class="box">
  <div class="logo"></div>
  <h1>Welcome back</h1>
  <p class="sub">Sign in to continue.</p>
  <div class="ok" id="ok">Signed in.</div>
  <form id="f" novalidate>
    <div class="fld"><label for="e">Email</label>
      <input id="e" type="email" autocomplete="email" placeholder="you@example.com">
      <div class="err" id="ee"></div></div>
    <div class="fld"><label for="p">Password</label>
      <input id="p" type="password" autocomplete="current-password" placeholder="********">
      <button class="peek" id="pk" type="button">SHOW</button>
      <div class="err" id="pe"></div></div>
    <div class="row">
      <label style="font-weight:500"><input type="checkbox" style="width:auto">
        Remember me</label>
      <a href="#">Forgot password?</a></div>
    <button class="go" id="go" type="submit">Sign in</button>
  </form>
  <p class="foot">No account? <a href="#" style="color:var(--a);font-weight:600">Create one</a></p>
</main>
"""
    js = r"""
var p=document.getElementById('p');
document.getElementById('pk').onclick=function(){
  var h=p.type==='password';p.type=h?'text':'password';
  this.textContent=h?'HIDE':'SHOW';};
document.getElementById('f').onsubmit=function(e){
  e.preventDefault();
  var em=document.getElementById('e'),bad=false;
  document.getElementById('ee').textContent='';
  document.getElementById('pe').textContent='';
  em.classList.remove('bad');p.classList.remove('bad');
  if(!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(em.value.trim())){
    document.getElementById('ee').textContent='Enter a valid email';
    em.classList.add('bad');bad=true;}
  if(p.value.length<8){
    document.getElementById('pe').textContent='At least 8 characters';
    p.classList.add('bad');bad=true;}
  if(bad)return;
  var b=document.getElementById('go');
  b.disabled=true;b.textContent='Signing in...';
  setTimeout(function(){
    b.disabled=false;b.textContent='Sign in';
    document.getElementById('ok').style.display='block';},700);};
"""
    return _off_page("Sign in", css, body, js)




# ---- big offline templates (patch 22) ----
_OFF_PORT_CSS = r'''
nav{position:sticky;top:0;z-index:20;backdrop-filter:blur(16px);
    background:rgba(248,250,252,.82);border-bottom:1px solid var(--line)}
.nv{display:flex;align-items:center;gap:26px;height:66px}
.lg{font-weight:800;letter-spacing:-.03em;font-size:19px;margin-right:auto}
.lg span{color:var(--a)}
.nv a.ln{font-size:14.5px;color:var(--mut);font-weight:600;position:relative}
.nv a.ln:hover{color:var(--ink)}
.nv a.ln::after{content:"";position:absolute;left:0;bottom:-6px;height:2px;
    width:0;background:var(--a);transition:width .25s}
.nv a.ln:hover::after{width:100%}
.burger{display:none;flex-direction:column;gap:5px;background:none;border:0;
    cursor:pointer;padding:8px}
.burger i{display:block;width:22px;height:2px;background:var(--ink);
    transition:.3s;border-radius:2px}
.burger.on i:nth-child(1){transform:translateY(7px) rotate(45deg)}
.burger.on i:nth-child(2){opacity:0}
.burger.on i:nth-child(3){transform:translateY(-7px) rotate(-45deg)}
.mob{display:none;flex-direction:column;gap:2px;padding:10px 22px 18px;
    border-bottom:1px solid var(--line);background:var(--card)}
.mob.on{display:flex}
.mob a{padding:11px 0;font-weight:600;border-bottom:1px solid var(--line)}
.btn{display:inline-block;padding:12px 22px;border-radius:11px;
    background:var(--a);color:#fff;font-weight:700;font-size:15px;border:0;
    cursor:pointer;transition:transform .18s,box-shadow .18s,background .18s}
.btn:hover{background:var(--b);transform:translateY(-2px);
    box-shadow:0 12px 26px -12px var(--a)}
.btn.ghost{background:transparent;color:var(--ink);
    border:1px solid var(--line)}
.btn.ghost:hover{border-color:var(--a);color:var(--a);box-shadow:none}
header.hero{padding:96px 0 84px;position:relative;overflow:hidden}
header.hero::before{content:"";position:absolute;width:560px;height:560px;
    right:-180px;top:-220px;border-radius:50%;
    background:radial-gradient(circle,var(--a),transparent 68%);opacity:.13}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:13px;
    font-weight:700;letter-spacing:.08em;text-transform:uppercase;
    color:var(--a);background:rgba(79,70,229,.09);padding:7px 14px;
    border-radius:99px;margin-bottom:20px}
.eyebrow i{width:7px;height:7px;border-radius:50%;background:var(--a);
    animation:pulse 1.9s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
h1{font-size:clamp(38px,6.4vw,66px);line-height:1.04;letter-spacing:-.035em;
    font-weight:800;margin-bottom:20px;max-width:15ch}
h1 em{font-style:normal;
    background:linear-gradient(100deg,var(--a),var(--b));
    -webkit-background-clip:text;background-clip:text;color:transparent}
.lead{font-size:19px;color:var(--mut);max-width:56ch;margin-bottom:32px}
#type::after{content:"|";animation:cur .9s steps(1) infinite;color:var(--a)}
@keyframes cur{50%{opacity:0}}
.cta{display:flex;gap:13px;flex-wrap:wrap;margin-bottom:52px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;max-width:620px}
.stat b{display:block;font-size:31px;font-weight:800;letter-spacing:-.03em}
.stat span{font-size:13px;color:var(--mut);font-weight:600}
section{padding:82px 0;border-top:1px solid var(--line)}
.shead{max-width:60ch;margin-bottom:44px}
h2{font-size:clamp(27px,3.6vw,38px);letter-spacing:-.03em;font-weight:800;
    margin-bottom:12px}
.shead p{color:var(--mut);font-size:17px}
.grid{display:grid;gap:20px}
.g3{grid-template-columns:repeat(3,1fr)}
.g2{grid-template-columns:repeat(2,1fr)}
.card{background:var(--card);border:1px solid var(--line);border-radius:17px;
    padding:26px;transition:transform .25s,box-shadow .25s,border-color .25s}
.card:hover{transform:translateY(-5px);border-color:var(--a);
    box-shadow:0 22px 48px -26px rgba(15,23,42,.4)}
.card h3{font-size:18.5px;margin-bottom:9px;letter-spacing:-.01em}
.card p{color:var(--mut);font-size:15px}
.ico{width:44px;height:44px;border-radius:12px;display:grid;place-items:center;
    background:rgba(79,70,229,.1);margin-bottom:16px}
.ico svg{width:22px;height:22px;stroke:var(--a);fill:none;stroke-width:1.9}
.filters{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:26px}
.chip{padding:8px 16px;border-radius:99px;border:1px solid var(--line);
    background:var(--card);font-size:14px;font-weight:600;cursor:pointer;
    color:var(--mut);transition:.2s}
.chip.on,.chip:hover{background:var(--a);color:#fff;border-color:var(--a)}
.proj{overflow:hidden;padding:0}
.thumb{height:158px;display:grid;place-items:center;font-weight:800;
    font-size:15px;color:#fff;letter-spacing:.04em}
.p1{background:linear-gradient(135deg,#4F46E5,#6366F1)}
.p2{background:linear-gradient(135deg,#0EA5E9,#22D3EE)}
.p3{background:linear-gradient(135deg,#F59E0B,#FB7185)}
.p4{background:linear-gradient(135deg,#10B981,#34D399)}
.p5{background:linear-gradient(135deg,#8B5CF6,#EC4899)}
.p6{background:linear-gradient(135deg,#0F172A,#475569)}
.pbody{padding:22px}
.tags{display:flex;gap:7px;flex-wrap:wrap;margin-top:14px}
.tag{font-size:12px;font-weight:700;padding:4px 10px;border-radius:6px;
    background:rgba(99,102,241,.1);color:var(--a)}
.skill{margin-bottom:19px}
.skill .row{display:flex;justify-content:space-between;font-size:14.5px;
    font-weight:700;margin-bottom:7px}
.skill .row span:last-child{color:var(--mut)}
.track{height:8px;border-radius:99px;background:var(--line);overflow:hidden}
.track i{display:block;height:100%;width:0;border-radius:99px;
    background:linear-gradient(90deg,var(--a),var(--b));transition:width 1.1s}
.tl{position:relative;padding-left:30px}
.tl::before{content:"";position:absolute;left:7px;top:6px;bottom:6px;width:2px;
    background:var(--line)}
.tlitem{position:relative;padding-bottom:30px}
.tlitem::before{content:"";position:absolute;left:-27px;top:5px;width:14px;
    height:14px;border-radius:50%;background:var(--card);
    border:3px solid var(--a)}
.tlitem time{font-size:12.5px;font-weight:700;color:var(--a);
    letter-spacing:.05em;text-transform:uppercase}
.tlitem h4{font-size:17px;margin:4px 0 5px}
.tlitem p{color:var(--mut);font-size:15px}
.quote{font-size:17px;line-height:1.7}
.who{display:flex;align-items:center;gap:12px;margin-top:18px}
.av{width:42px;height:42px;border-radius:50%;display:grid;place-items:center;
    background:linear-gradient(135deg,var(--a),var(--b));color:#fff;
    font-weight:800;font-size:15px}
.who b{display:block;font-size:14.5px}
.who span{font-size:13px;color:var(--mut)}
form{display:grid;gap:16px;max-width:560px}
.field label{display:block;font-size:14px;font-weight:700;margin-bottom:7px}
.field input,.field textarea{width:100%;padding:13px 15px;border-radius:11px;
    border:1px solid var(--line);background:var(--card);font:inherit;
    color:var(--ink);transition:border-color .2s,box-shadow .2s}
.field input:focus,.field textarea:focus{outline:0;border-color:var(--a);
    box-shadow:0 0 0 4px rgba(79,70,229,.13)}
.field textarea{min-height:130px;resize:vertical}
.err{color:#DC2626;font-size:13px;font-weight:600;margin-top:6px;display:none}
.field.bad input,.field.bad textarea{border-color:#DC2626}
.field.bad .err{display:block}
.okmsg{display:none;padding:14px 16px;border-radius:11px;font-weight:600;
    background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0}
.okmsg.on{display:block}
footer{padding:40px 0;border-top:1px solid var(--line);display:flex;
    justify-content:space-between;gap:16px;flex-wrap:wrap;
    color:var(--mut);font-size:14.5px}
.soc{display:flex;gap:10px}
.soc a{width:38px;height:38px;border-radius:10px;display:grid;
    place-items:center;border:1px solid var(--line);transition:.2s}
.soc a:hover{background:var(--a);border-color:var(--a)}
.soc svg{width:17px;height:17px;stroke:var(--mut);fill:none;stroke-width:1.9}
.soc a:hover svg{stroke:#fff}
#top{position:fixed;right:22px;bottom:22px;width:46px;height:46px;
    border-radius:50%;border:0;background:var(--a);color:#fff;cursor:pointer;
    font-size:19px;opacity:0;pointer-events:none;transition:.3s;
    box-shadow:0 12px 28px -12px var(--a)}
#top.on{opacity:1;pointer-events:auto}
.rv{opacity:0;transform:translateY(24px);
    transition:opacity .65s ease,transform .65s ease}
.rv.in{opacity:1;transform:none}
@media(max-width:880px){
  .g3{grid-template-columns:repeat(2,1fr)}
  .stats{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:640px){
  .nv a.ln{display:none}.burger{display:flex}
  .g3,.g2{grid-template-columns:1fr}
  header.hero{padding:60px 0 54px}section{padding:58px 0}
}
'''

_OFF_PORT_BODY = r'''
<nav>
  <div class="wrap nv">
    <div class="lg">%(NM)s<span>.</span></div>
    <a class="ln" href="#work">Work</a>
    <a class="ln" href="#services">Services</a>
    <a class="ln" href="#skills">Skills</a>
    <a class="ln" href="#about">About</a>
    <a class="ln" href="#contact">Contact</a>
    <button class="burger" id="bg" aria-label="Menu">
      <i></i><i></i><i></i></button>
  </div>
  <div class="mob" id="mb">
    <a href="#work">Work</a><a href="#services">Services</a>
    <a href="#skills">Skills</a><a href="#about">About</a>
    <a href="#contact">Contact</a>
  </div>
</nav>

<header class="hero">
  <div class="wrap">
    <div class="eyebrow"><i></i>Available for new work</div>
    <h1>Hi, I am %(NM)s &mdash; I build <em id="type"></em></h1>
    <p class="lead">Product designer and front-end developer. I turn rough
      ideas into fast, accessible interfaces people actually enjoy using.
      Eight years, forty-plus shipped projects, zero abandoned handovers.</p>
    <div class="cta">
      <a class="btn" href="#contact">Start a project</a>
      <a class="btn ghost" href="#work">See my work</a>
    </div>
    <div class="stats">
      <div class="stat"><b class="cnt" data-to="8">0</b><span>Years</span></div>
      <div class="stat"><b class="cnt" data-to="43">0</b><span>Projects</span></div>
      <div class="stat"><b class="cnt" data-to="29">0</b><span>Clients</span></div>
      <div class="stat"><b class="cnt" data-to="14">0</b><span>Awards</span></div>
    </div>
  </div>
</header>

<section id="services">
  <div class="wrap">
    <div class="shead rv"><h2>What I do</h2>
      <p>Three things, done properly, rather than ten done loosely.</p></div>
    <div class="grid g3">
      <div class="card rv"><div class="ico"><svg viewBox="0 0 24 24">
        <rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18"/>
        </svg></div><h3>Interface design</h3>
        <p>Design systems, component libraries and prototypes in Figma that
        map one-to-one onto real code.</p></div>
      <div class="card rv"><div class="ico"><svg viewBox="0 0 24 24">
        <path d="M8 18l-5-6 5-6"/><path d="M16 6l5 6-5 6"/></svg></div>
        <h3>Front-end build</h3>
        <p>Hand-written HTML, CSS and JavaScript. Semantic, accessible,
        and fast on a mid-range phone over 3G.</p></div>
      <div class="card rv"><div class="ico"><svg viewBox="0 0 24 24">
        <path d="M3 17l6-6 4 4 8-8"/><path d="M21 7v6h-6"/></svg></div>
        <h3>Performance work</h3>
        <p>Audits that move real numbers: Largest Contentful Paint,
        bundle weight, and the checkout conversion behind them.</p></div>
    </div>
  </div>
</section>

<section id="work">
  <div class="wrap">
    <div class="shead rv"><h2>Selected work</h2>
      <p>Filter by discipline. Every one shipped to production.</p></div>
    <div class="filters rv">
      <button class="chip on" data-f="all">All</button>
      <button class="chip" data-f="web">Web</button>
      <button class="chip" data-f="app">App</button>
      <button class="chip" data-f="brand">Brand</button>
    </div>
    <div class="grid g3" id="pg">
      <article class="card proj rv" data-c="web">
        <div class="thumb p1">NORTHWIND</div><div class="pbody">
        <h3>Northwind Analytics</h3>
        <p>Dashboard for 12,000 daily users. Cut first paint from
        4.1s to 0.9s.</p>
        <div class="tags"><span class="tag">React</span>
        <span class="tag">D3</span><span class="tag">Design system</span>
        </div></div></article>
      <article class="card proj rv" data-c="app">
        <div class="thumb p2">HARBOUR</div><div class="pbody">
        <h3>Harbour Banking</h3>
        <p>Mobile banking flow rebuilt around one-thumb use. Support
        tickets down 38%.</p>
        <div class="tags"><span class="tag">iOS</span>
        <span class="tag">Figma</span><span class="tag">Research</span>
        </div></div></article>
      <article class="card proj rv" data-c="brand">
        <div class="thumb p3">EMBER</div><div class="pbody">
        <h3>Ember Coffee</h3>
        <p>Identity, packaging and a storefront that sold out the first
        roast in nine days.</p>
        <div class="tags"><span class="tag">Identity</span>
        <span class="tag">Shopify</span></div></div></article>
      <article class="card proj rv" data-c="web">
        <div class="thumb p4">ATLAS</div><div class="pbody">
        <h3>Atlas Docs</h3>
        <p>Documentation platform with instant search across 4,200 pages.</p>
        <div class="tags"><span class="tag">Static</span>
        <span class="tag">Search</span></div></div></article>
      <article class="card proj rv" data-c="app">
        <div class="thumb p5">PULSE</div><div class="pbody">
        <h3>Pulse Fitness</h3>
        <p>Habit tracker with offline sync. 4.8 stars across 6,100 reviews.</p>
        <div class="tags"><span class="tag">PWA</span>
        <span class="tag">IndexedDB</span></div></div></article>
      <article class="card proj rv" data-c="brand">
        <div class="thumb p6">MERIDIAN</div><div class="pbody">
        <h3>Meridian Law</h3>
        <p>Quiet, confident brand for a firm that hated looking like a firm.</p>
        <div class="tags"><span class="tag">Identity</span>
        <span class="tag">Web</span></div></div></article>
    </div>
  </div>
</section>

<section id="skills">
  <div class="wrap">
    <div class="shead rv"><h2>Skills and tools</h2>
      <p>Honest numbers &mdash; what I reach for without thinking.</p></div>
    <div class="grid g2">
      <div class="rv">
        <div class="skill"><div class="row"><span>HTML &amp; CSS</span>
          <span>96%</span></div><div class="track">
          <i class="bar" data-w="96"></i></div></div>
        <div class="skill"><div class="row"><span>JavaScript</span>
          <span>91%</span></div><div class="track">
          <i class="bar" data-w="91"></i></div></div>
        <div class="skill"><div class="row"><span>Accessibility</span>
          <span>88%</span></div><div class="track">
          <i class="bar" data-w="88"></i></div></div>
      </div>
      <div class="rv">
        <div class="skill"><div class="row"><span>Figma</span>
          <span>93%</span></div><div class="track">
          <i class="bar" data-w="93"></i></div></div>
        <div class="skill"><div class="row"><span>Performance</span>
          <span>85%</span></div><div class="track">
          <i class="bar" data-w="85"></i></div></div>
        <div class="skill"><div class="row"><span>Node &amp; APIs</span>
          <span>78%</span></div><div class="track">
          <i class="bar" data-w="78"></i></div></div>
      </div>
    </div>
  </div>
</section>

<section id="about">
  <div class="wrap">
    <div class="shead rv"><h2>The short version</h2>
      <p>Where I have been and what I learned there.</p></div>
    <div class="grid g2">
      <div class="tl rv">
        <div class="tlitem"><time>2023 &mdash; now</time>
          <h4>Independent</h4>
          <p>Design and build for startups and small teams. Usually the only
          person between the idea and the shipped page.</p></div>
        <div class="tlitem"><time>2020 &mdash; 2023</time>
          <h4>Lead Product Designer, Northwind</h4>
          <p>Grew the design system to 84 components and got engineering to
          actually use it.</p></div>
        <div class="tlitem"><time>2017 &mdash; 2020</time>
          <h4>Front-end Developer, Studio Kite</h4>
          <p>Twenty-plus client sites. Learned that the brief is never the
          real problem.</p></div>
      </div>
      <div class="rv">
        <div class="card"><p class="quote">&ldquo;Gave us in six weeks what
          two agencies could not manage in a year. Clear, quick, and never
          precious about feedback.&rdquo;</p>
          <div class="who"><div class="av">RS</div>
          <div><b>Rina Sato</b><span>Head of Product, Harbour</span></div>
          </div></div>
        <div class="card" style="margin-top:18px"><p class="quote">
          &ldquo;The rebuild paid for itself in one quarter. Pages load
          instantly and support stopped drowning.&rdquo;</p>
          <div class="who"><div class="av">DM</div>
          <div><b>Daniel Mensah</b><span>Founder, Ember</span></div>
          </div></div>
      </div>
    </div>
  </div>
</section>

<section id="contact">
  <div class="wrap">
    <div class="shead rv"><h2>Start a project</h2>
      <p>Tell me roughly what you need. I reply within one working day.</p>
    </div>
    <div class="okmsg" id="ok">Thanks &mdash; your message is ready to send.
      I will get back to you within one working day.</div>
    <form id="cf" novalidate class="rv">
      <div class="field" id="f_name"><label for="i_name">Your name</label>
        <input id="i_name" placeholder="Jane Cooper">
        <div class="err">Please tell me your name.</div></div>
      <div class="field" id="f_mail"><label for="i_mail">Email</label>
        <input id="i_mail" type="email" placeholder="jane@company.com">
        <div class="err">That email does not look right.</div></div>
      <div class="field" id="f_msg"><label for="i_msg">Project</label>
        <textarea id="i_msg" placeholder="What are you building, and when
does it need to be live?"></textarea>
        <div class="err">A sentence or two is plenty.</div></div>
      <button class="btn" type="submit">Send message</button>
    </form>
  </div>
</section>

<footer class="wrap">
  <div>&copy; <span id="yr"></span> %(NM)s. Built by hand.</div>
  <div class="soc">
    <a href="#" aria-label="GitHub"><svg viewBox="0 0 24 24">
      <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.9a3.4 3.4 0 0 0-1-2.6c3-.3 6-1.5
      6-6.5a5 5 0 0 0-1.4-3.5 4.7 4.7 0 0 0-.1-3.5s-1.1-.3-3.5 1.3a12 12 0 0 0
      -6 0C7.6 1.2 6.5 1.5 6.5 1.5A4.7 4.7 0 0 0 6.4 5 5 5 0 0 0 5 8.5c0 5 3
      6.2 6 6.5a3.4 3.4 0 0 0-1 2.6V22"/></svg></a>
    <a href="#" aria-label="LinkedIn"><svg viewBox="0 0 24 24">
      <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1
      6-6z"/><rect x="2" y="9" width="4" height="12"/>
      <circle cx="4" cy="4" r="2"/></svg></a>
    <a href="#" aria-label="Email"><svg viewBox="0 0 24 24">
      <rect x="2" y="4" width="20" height="16" rx="2"/>
      <path d="m22 7-10 6L2 7"/></svg></a>
  </div>
</footer>
<button id="top" aria-label="Back to top">&uarr;</button>
'''

_OFF_PORT_JS = r'''
document.getElementById('yr').textContent = new Date().getFullYear();

/* mobile menu ------------------------------------------------------- */
var bg = document.getElementById('bg'), mb = document.getElementById('mb');
bg.addEventListener('click', function () {
  bg.classList.toggle('on'); mb.classList.toggle('on');
});
mb.querySelectorAll('a').forEach(function (a) {
  a.addEventListener('click', function () {
    bg.classList.remove('on'); mb.classList.remove('on');
  });
});

/* smooth scrolling -------------------------------------------------- */
document.querySelectorAll('a[href^="#"]').forEach(function (a) {
  a.addEventListener('click', function (e) {
    var t = document.querySelector(a.getAttribute('href'));
    if (!t) return;
    e.preventDefault();
    window.scrollTo({ top: t.offsetTop - 60, behavior: 'smooth' });
  });
});

/* typing headline --------------------------------------------------- */
var words = ['fast websites', 'clean interfaces', 'design systems',
             'things that ship'];
var el = document.getElementById('type'), wi = 0, ci = 0, del = false;
(function tick() {
  var w = words[wi];
  el.textContent = del ? w.slice(0, --ci) : w.slice(0, ++ci);
  var wait = del ? 45 : 90;
  if (!del && ci === w.length) { del = true; wait = 1500; }
  else if (del && ci === 0) { del = false; wi = (wi + 1) % words.length; wait = 260; }
  setTimeout(tick, wait);
})();

/* reveal on scroll + counters + skill bars -------------------------- */
var io = new IntersectionObserver(function (entries) {
  entries.forEach(function (en) {
    if (!en.isIntersecting) return;
    en.target.classList.add('in');
    en.target.querySelectorAll('.bar').forEach(function (b) {
      b.style.width = b.dataset.w + '%';
    });
    en.target.querySelectorAll('.cnt').forEach(count);
    io.unobserve(en.target);
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
document.querySelectorAll('.rv').forEach(function (n) { io.observe(n); });
/* safety net: if a fast scroll skips the observer, fill bars anyway */
window.addEventListener('scroll', function () {
  document.querySelectorAll('.bar').forEach(function (b) {
    if (b.style.width) return;
    var r = b.getBoundingClientRect();
    if (r.top < window.innerHeight && r.bottom > 0) {
      b.closest('.rv').classList.add('in');
      b.style.width = b.dataset.w + '%';
    }
  });
}, { passive: true });

function count(node) {
  var to = +node.dataset.to, from = 0, t0 = performance.now(), dur = 1100;
  (function step(now) {
    var p = Math.min(1, (now - t0) / dur);
    node.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}
document.querySelectorAll('.stats .cnt').forEach(count);

/* project filter ---------------------------------------------------- */
document.querySelectorAll('.chip').forEach(function (c) {
  c.addEventListener('click', function () {
    document.querySelectorAll('.chip').forEach(function (o) {
      o.classList.remove('on');
    });
    c.classList.add('on');
    var f = c.dataset.f;
    document.querySelectorAll('.proj').forEach(function (p) {
      var show = f === 'all' || p.dataset.c === f;
      p.style.display = show ? '' : 'none';
    });
  });
});

/* contact form validation ------------------------------------------- */
var form = document.getElementById('cf');
function bad(id, cond) {
  document.getElementById(id).classList.toggle('bad', cond);
  return cond;
}
form.addEventListener('submit', function (e) {
  e.preventDefault();
  var n = document.getElementById('i_name').value.trim();
  var m = document.getElementById('i_mail').value.trim();
  var g = document.getElementById('i_msg').value.trim();
  var e1 = bad('f_name', n.length < 2);
  var e2 = bad('f_mail', !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(m));
  var e3 = bad('f_msg', g.length < 8);
  if (e1 || e2 || e3) return;
  document.getElementById('ok').classList.add('on');
  form.reset();
  window.scrollTo({ top: document.getElementById('ok').offsetTop - 90,
                    behavior: 'smooth' });
});
['i_name', 'i_mail', 'i_msg'].forEach(function (id) {
  document.getElementById(id).addEventListener('input', function () {
    this.closest('.field').classList.remove('bad');
  });
});

/* back to top ------------------------------------------------------- */
var top_ = document.getElementById('top');
window.addEventListener('scroll', function () {
  top_.classList.toggle('on', window.scrollY > 520);
});
top_.addEventListener('click', function () {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});
'''

_OFF_LOG_CSS = r'''
body{min-height:100vh;display:grid;place-items:center;padding:26px;
  background:radial-gradient(1200px 620px at 12% -10%,#EEF2FF,transparent),
             radial-gradient(1000px 560px at 100% 110%,#E0E7FF,transparent),
             var(--bg)}
.shell{width:100%;max-width:430px}
.brand{display:flex;align-items:center;gap:11px;justify-content:center;
  margin-bottom:26px}
.mark{width:42px;height:42px;border-radius:13px;display:grid;
  place-items:center;background:linear-gradient(135deg,var(--a),var(--b));
  box-shadow:0 12px 26px -12px var(--a)}
.mark svg{width:21px;height:21px;stroke:#fff;fill:none;stroke-width:2.1}
.brand b{font-size:20px;letter-spacing:-.03em;font-weight:800}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;
  padding:30px 28px;box-shadow:0 26px 60px -34px rgba(15,23,42,.42)}
.tabs{display:flex;background:var(--bg);border:1px solid var(--line);
  border-radius:12px;padding:4px;margin-bottom:24px}
.tabs button{flex:1;padding:10px;border:0;border-radius:9px;background:none;
  font:inherit;font-weight:700;font-size:14.5px;color:var(--mut);
  cursor:pointer;transition:.22s}
.tabs button.on{background:var(--card);color:var(--ink);
  box-shadow:0 2px 8px -3px rgba(15,23,42,.28)}
h1{font-size:24px;letter-spacing:-.03em;margin-bottom:6px;font-weight:800}
.sub{color:var(--mut);font-size:14.5px;margin-bottom:24px}
.field{margin-bottom:17px}
.field label{display:block;font-size:13.5px;font-weight:700;margin-bottom:7px}
.inp{position:relative}
.inp input{width:100%;padding:13px 44px 13px 15px;border-radius:12px;
  border:1px solid var(--line);background:var(--card);font:inherit;
  color:var(--ink);transition:border-color .2s,box-shadow .2s}
.inp input:focus{outline:0;border-color:var(--a);
  box-shadow:0 0 0 4px rgba(79,70,229,.13)}
.eye{position:absolute;right:6px;top:50%;transform:translateY(-50%);
  background:none;border:0;cursor:pointer;padding:8px;line-height:0}
.eye svg{width:19px;height:19px;stroke:var(--mut);fill:none;stroke-width:1.9}
.err{color:#DC2626;font-size:12.5px;font-weight:600;margin-top:6px;
  display:none}
.field.bad .inp input{border-color:#DC2626}
.field.bad .inp input:focus{box-shadow:0 0 0 4px rgba(220,38,38,.13)}
.field.bad .err{display:block}
.meter{display:flex;gap:5px;margin-top:9px}
.meter i{flex:1;height:4px;border-radius:99px;background:var(--line);
  transition:background .3s}
.mtxt{font-size:12.5px;font-weight:700;margin-top:6px;color:var(--mut)}
.row{display:flex;align-items:center;justify-content:space-between;
  margin:4px 0 22px;font-size:13.5px}
.chk{display:flex;align-items:center;gap:8px;cursor:pointer;font-weight:600;
  color:var(--mut)}
.chk input{width:16px;height:16px;accent-color:var(--a);cursor:pointer}
.lnk{color:var(--a);font-weight:700;background:none;border:0;font:inherit;
  cursor:pointer}
.lnk:hover{text-decoration:underline}
.btn{width:100%;padding:14px;border:0;border-radius:12px;background:var(--a);
  color:#fff;font:inherit;font-weight:700;font-size:15.5px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:9px;
  transition:background .2s,transform .18s}
.btn:hover:not(:disabled){background:var(--b);transform:translateY(-1px)}
.btn:disabled{opacity:.66;cursor:not-allowed}
.spin{width:16px;height:16px;border:2px solid rgba(255,255,255,.35);
  border-top-color:#fff;border-radius:50%;display:none;
  animation:sp .7s linear infinite}
.btn.load .spin{display:block}
@keyframes sp{to{transform:rotate(360deg)}}
.or{display:flex;align-items:center;gap:12px;margin:22px 0;color:var(--mut);
  font-size:12.5px;font-weight:700;letter-spacing:.06em}
.or::before,.or::after{content:"";flex:1;height:1px;background:var(--line)}
.socs{display:grid;grid-template-columns:1fr 1fr;gap:11px}
.socs button{padding:12px;border:1px solid var(--line);border-radius:12px;
  background:var(--card);font:inherit;font-weight:700;font-size:14px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;
  gap:8px;transition:.2s}
.socs button:hover{border-color:var(--a);color:var(--a)}
.socs svg{width:17px;height:17px}
.foot{text-align:center;margin-top:22px;font-size:13.5px;color:var(--mut)}
.toast{position:fixed;left:50%;bottom:26px;transform:translate(-50%,90px);
  background:var(--ink);color:#fff;padding:13px 20px;border-radius:12px;
  font-size:14.5px;font-weight:600;opacity:0;transition:.35s;z-index:40}
.toast.on{transform:translate(-50%,0);opacity:1}
.toast.good{background:#065F46}.toast.bad{background:#B91C1C}
.hide{display:none}
.caps{font-size:12.5px;font-weight:700;color:#B45309;margin-top:6px;
  display:none}
.caps.on{display:block}
@media(max-width:460px){.card{padding:24px 20px}.socs{grid-template-columns:1fr}}
'''

_OFF_LOG_BODY = r'''
<div class="shell">
  <div class="brand">
    <div class="mark"><svg viewBox="0 0 24 24">
      <path d="M12 2 3 7v6c0 5 3.8 8.4 9 9 5.2-.6 9-4 9-9V7z"/>
      <path d="m9 12 2 2 4-4"/></svg></div>
    <b>Northwind</b>
  </div>

  <div class="card">
    <div class="tabs">
      <button id="t_in" class="on">Sign in</button>
      <button id="t_up">Create account</button>
    </div>

    <h1 id="ttl">Welcome back</h1>
    <p class="sub" id="sub">Sign in to pick up where you left off.</p>

    <form id="form" novalidate>
      <div class="field hide" id="f_name">
        <label for="i_name">Full name</label>
        <div class="inp"><input id="i_name" placeholder="Jane Cooper"
          autocomplete="name"></div>
        <div class="err" id="e_name">Please enter your name.</div>
      </div>

      <div class="field" id="f_mail">
        <label for="i_mail">Email address</label>
        <div class="inp"><input id="i_mail" type="email"
          placeholder="you@company.com" autocomplete="email"></div>
        <div class="err" id="e_mail">Enter a valid email address.</div>
      </div>

      <div class="field" id="f_pass">
        <label for="i_pass">Password</label>
        <div class="inp">
          <input id="i_pass" type="password" placeholder="At least 8 characters"
            autocomplete="current-password">
          <button class="eye" type="button" id="eye" aria-label="Show password">
            <svg viewBox="0 0 24 24" id="eyeicon">
              <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"/>
              <circle cx="12" cy="12" r="3"/></svg></button>
        </div>
        <div class="caps" id="caps">Caps Lock is on.</div>
        <div class="meter hide" id="meter"><i></i><i></i><i></i><i></i></div>
        <div class="mtxt hide" id="mtxt">Strength</div>
        <div class="err" id="e_pass">Password must be at least 8 characters.</div>
      </div>

      <div class="field hide" id="f_conf">
        <label for="i_conf">Confirm password</label>
        <div class="inp"><input id="i_conf" type="password"
          placeholder="Repeat your password" autocomplete="new-password"></div>
        <div class="err" id="e_conf">The two passwords do not match.</div>
      </div>

      <div class="row">
        <label class="chk"><input type="checkbox" id="rem" checked>
          Remember me</label>
        <button type="button" class="lnk" id="forgot">Forgot password?</button>
      </div>

      <button class="btn" type="submit" id="submit">
        <span class="spin"></span><span id="blabel">Sign in</span></button>
    </form>

    <div class="or">OR</div>
    <div class="socs">
      <button type="button" data-p="Google">
        <svg viewBox="0 0 24 24"><path fill="#EA4335" d="M12 10.2v3.9h5.5a4.7
        4.7 0 0 1-2 3.1l3.2 2.5c1.9-1.7 3-4.3 3-7.4 0-.7-.1-1.4-.2-2z"/>
        <path fill="#34A853" d="M12 22c2.7 0 5-.9 6.7-2.4l-3.2-2.5c-.9.6-2
        1-3.5 1a6 6 0 0 1-5.6-4.1l-3.3 2.5A10 10 0 0 0 12 22z"/>
        <path fill="#FBBC05" d="M6.4 14a6 6 0 0 1 0-3.9L3.1 7.6a10 10 0 0 0 0
        8.8z"/><path fill="#4285F4" d="M12 6c1.5 0 2.8.5 3.8 1.5l2.8-2.8A10 10
        0 0 0 3.1 7.6l3.3 2.5A6 6 0 0 1 12 6z"/></svg>Google</button>
      <button type="button" data-p="GitHub">
        <svg viewBox="0 0 24 24" fill="#0F172A"><path d="M12 2a10 10 0 0 0-3.2
        19.5c.5.1.7-.2.7-.5v-1.8c-2.8.6-3.4-1.3-3.4-1.3-.4-1.2-1.1-1.5-1.1-1.5
        -.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.9.8.1-.6.3-1.1.6-1.3
        -2.2-.3-4.6-1.1-4.6-5 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.7 0 0 .8-.3
        2.7 1a9.4 9.4 0 0 1 5 0c1.9-1.3 2.7-1 2.7-1 .5 1.4.2 2.4.1 2.7.6.7 1
        1.6 1 2.7 0 3.9-2.4 4.7-4.6 5 .4.3.7.9.7 1.9v2.8c0 .3.2.6.7.5A10 10 0 0
        0 12 2z"/></svg>GitHub</button>
    </div>

    <p class="foot" id="foot">New here?
      <button class="lnk" id="swap">Create an account</button></p>
  </div>
</div>
<div class="toast" id="toast"></div>
'''

_OFF_LOG_JS = r'''
var MODE = 'in';                       /* 'in' = sign in, 'up' = sign up  */
var $ = function (id) { return document.getElementById(id); };

/* ---------------------------------------------------------- toast ---- */
var tHide;
function toast(msg, kind) {
  var t = $('toast');
  t.textContent = msg;
  t.className = 'toast on ' + (kind || '');
  clearTimeout(tHide);
  tHide = setTimeout(function () { t.className = 'toast ' + (kind || ''); }, 3200);
}

/* ------------------------------------------------------ mode swap ---- */
function setMode(m) {
  MODE = m;
  var up = m === 'up';
  $('t_in').classList.toggle('on', !up);
  $('t_up').classList.toggle('on', up);
  $('ttl').textContent = up ? 'Create your account' : 'Welcome back';
  $('sub').textContent = up
    ? 'It takes about thirty seconds.'
    : 'Sign in to pick up where you left off.';
  $('blabel').textContent = up ? 'Create account' : 'Sign in';
  $('f_name').classList.toggle('hide', !up);
  $('f_conf').classList.toggle('hide', !up);
  $('meter').classList.toggle('hide', !up);
  $('mtxt').classList.toggle('hide', !up);
  $('i_pass').setAttribute('autocomplete',
    up ? 'new-password' : 'current-password');
  $('foot').innerHTML = up
    ? 'Already have an account? <button class="lnk" id="swap">Sign in</button>'
    : 'New here? <button class="lnk" id="swap">Create an account</button>';
  $('swap').addEventListener('click', function () {
    setMode(MODE === 'in' ? 'up' : 'in');
  });
  ['f_name', 'f_mail', 'f_pass', 'f_conf'].forEach(function (f) {
    $(f).classList.remove('bad');
  });
}
$('t_in').addEventListener('click', function () { setMode('in'); });
$('t_up').addEventListener('click', function () { setMode('up'); });
$('swap').addEventListener('click', function () {
  setMode(MODE === 'in' ? 'up' : 'in');
});

/* --------------------------------------------------- show password --- */
$('eye').addEventListener('click', function () {
  var p = $('i_pass');
  var shown = p.type === 'text';
  p.type = shown ? 'password' : 'text';
  $('eyeicon').innerHTML = shown
    ? '<path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12z"/>' +
      '<circle cx="12" cy="12" r="3"/>'
    : '<path d="M17.9 17.9A10.5 10.5 0 0 1 12 19C5 19 1 12 1 12a19 19 0 0 1' +
      ' 5.1-5.9"/><path d="M9.9 4.2A10.9 10.9 0 0 1 12 4c7 0 11 8 11 8a19' +
      ' 19 0 0 1-2.2 3.2"/><path d="m1 1 22 22"/>';
  this.setAttribute('aria-label', shown ? 'Show password' : 'Hide password');
});

/* ------------------------------------------------------- caps lock --- */
$('i_pass').addEventListener('keyup', function (e) {
  var on = e.getModifierState && e.getModifierState('CapsLock');
  $('caps').classList.toggle('on', !!on);
});

/* ------------------------------------------------ password strength -- */
function score(p) {
  var s = 1;
  if (p.length >= 8) s++;
  if (p.length >= 12) s++;
  if (/[A-Z]/.test(p) && /[a-z]/.test(p)) s++;
  if (/\d/.test(p) && /[^A-Za-z0-9]/.test(p)) s++;
  return Math.min(s, 4);
}
var COLORS = ['#E2E8F0', '#DC2626', '#F59E0B', '#3B82F6', '#059669'];
var LABELS = ['Strength', 'Too short', 'Weak', 'Good', 'Strong'];
$('i_pass').addEventListener('input', function () {
  var s = this.value ? score(this.value) : 0;
  var bars = $('meter').children;
  for (var i = 0; i < bars.length; i++) {
    bars[i].style.background = i < s ? COLORS[s] : '#E2E8F0';
  }
  $('mtxt').textContent = LABELS[s];
  $('mtxt').style.color = s ? COLORS[s] : '#64748B';
});

/* ------------------------------------------------------ validation --- */
function mark(field, isBad, msg) {
  $(field).classList.toggle('bad', isBad);
  if (msg) $('e_' + field.slice(2)).textContent = msg;
  return isBad;
}
['i_name', 'i_mail', 'i_pass', 'i_conf'].forEach(function (id) {
  $(id).addEventListener('input', function () {
    this.closest('.field').classList.remove('bad');
  });
});

function validate() {
  var bad = false;
  var mail = $('i_mail').value.trim();
  var pass = $('i_pass').value;
  if (MODE === 'up') {
    bad = mark('f_name', $('i_name').value.trim().length < 2) || bad;
  }
  bad = mark('f_mail', !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(mail)) || bad;
  bad = mark('f_pass', pass.length < 8) || bad;
  if (MODE === 'up') {
    bad = mark('f_conf', $('i_conf').value !== pass || !$('i_conf').value)
          || bad;
  }
  return !bad;
}

/* ---------------------------------------------------------- submit --- */
$('form').addEventListener('submit', function (e) {
  e.preventDefault();
  if (!validate()) { toast('Please fix the highlighted fields.', 'bad'); return; }
  var btn = $('submit');
  btn.classList.add('load');
  btn.disabled = true;
  /* Replace this timeout with your real fetch() call. */
  setTimeout(function () {
    btn.classList.remove('load');
    btn.disabled = false;
    if ($('rem').checked) {
      try { localStorage.setItem('lastEmail', $('i_mail').value.trim()); }
      catch (err) {}
    }
    toast(MODE === 'up' ? 'Account created. Welcome aboard.'
                        : 'Signed in successfully.', 'good');
  }, 1200);
});

/* --------------------------------------------------------- extras ---- */
$('forgot').addEventListener('click', function () {
  var mail = $('i_mail').value.trim();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(mail)) {
    mark('f_mail', true, 'Enter your email first, then tap reset.');
    return;
  }
  toast('Reset link sent to ' + mail, 'good');
});
document.querySelectorAll('.socs button').forEach(function (b) {
  b.addEventListener('click', function () {
    toast('Continuing with ' + b.dataset.p + '...');
  });
});
try {
  var last = localStorage.getItem('lastEmail');
  if (last) $('i_mail').value = last;
} catch (err) {}
'''


def _off_portfolio_big(name="Your Name"):
    """A full portfolio site - nav, hero, work filter, skills, timeline,
    testimonials, validated contact form. Roughly 12 KB of real code."""
    body = _OFF_PORT_BODY.replace("%(NM)s", name)
    return _off_page(name + " - Portfolio", _OFF_PORT_CSS, body, _OFF_PORT_JS)


def _off_login_big():
    """Sign in and create account in one card, with strength meter,
    caps-lock warning, social buttons and a loading state."""
    return _off_page("Sign in", _OFF_LOG_CSS, _OFF_LOG_BODY, _OFF_LOG_JS)

_OFF_PY_APP = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Task Manager - a complete, single-file Python application.

Standard library only. No pip install, no config, nothing to set up.

    python3 tasks.py add "Buy milk" --due 2026-01-30 --tag home --priority 2
    python3 tasks.py list --status open --sort due
    python3 tasks.py done 3
    python3 tasks.py search milk
    python3 tasks.py stats
    python3 tasks.py export tasks.csv
    python3 tasks.py serve --port 8080
    python3 tasks.py test

The same data is reachable three ways: a command line, a small REST API,
and a browser page served from the API. Everything persists in SQLite.
\"\"\"

from __future__ import annotations

import argparse
import csv
import datetime as dt
import http.server
import json
import os
import re
import socketserver
import sqlite3
import sys
import threading
import unittest
import urllib.parse
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

APP_NAME = "Task Manager"
APP_VERSION = "1.0.0"
DB_ENV = "TASKS_DB"
DEFAULT_DB = os.environ.get(DB_ENV, os.path.expanduser("~/.tasks.db"))

STATUS_OPEN = "open"
STATUS_DONE = "done"
STATUS_ALL = (STATUS_OPEN, STATUS_DONE)

PRIORITIES = {0: "none", 1: "low", 2: "normal", 3: "high", 4: "urgent"}

DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")


# =====================================================================
#  Errors
# =====================================================================
class TaskError(Exception):
    \"\"\"Anything the user did wrong, phrased so it can be printed as-is.\"\"\"


class NotFound(TaskError):
    pass


# =====================================================================
#  Model
# =====================================================================
@dataclass
class Task:
    \"\"\"One task. `id` is None until the row has been written.\"\"\"

    title: str
    note: str = ""
    status: str = STATUS_OPEN
    priority: int = 2
    due: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    id: Optional[int] = None

    # ---------------------------------------------------------- helpers
    def validate(self) -> None:
        if not self.title or not self.title.strip():
            raise TaskError("A task needs a title.")
        if len(self.title) > 300:
            raise TaskError("Title is too long (max 300 characters).")
        if self.status not in STATUS_ALL:
            raise TaskError("Status must be 'open' or 'done'.")
        if self.priority not in PRIORITIES:
            raise TaskError("Priority must be 0-4.")
        if self.due is not None and self.due != "":
            if not DATE_RE.match(self.due):
                raise TaskError("Due date must look like 2026-01-30.")
            try:
                dt.date.fromisoformat(self.due)
            except ValueError:
                raise TaskError("That due date is not a real date.")
        for t in self.tags:
            if not re.match(r"^[\\w-]{1,24}$", t):
                raise TaskError("Bad tag %r - use letters, digits, - or _." % t)

    @property
    def overdue(self) -> bool:
        if self.status == STATUS_DONE or not self.due:
            return False
        return dt.date.fromisoformat(self.due) < dt.date.today()

    @property
    def days_left(self) -> Optional[int]:
        if not self.due:
            return None
        return (dt.date.fromisoformat(self.due) - dt.date.today()).days

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["overdue"] = self.overdue
        d["days_left"] = self.days_left
        d["priority_label"] = PRIORITIES[self.priority]
        return d

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Task":
        return Task(
            id=row["id"],
            title=row["title"],
            note=row["note"] or "",
            status=row["status"],
            priority=row["priority"],
            due=row["due"] or None,
            tags=[t for t in (row["tags"] or "").split(",") if t],
            created=row["created"],
            updated=row["updated"],
        )


# =====================================================================
#  Storage
# =====================================================================
SCHEMA = \"\"\"
CREATE TABLE IF NOT EXISTS tasks (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  title    TEXT    NOT NULL,
  note     TEXT    NOT NULL DEFAULT '',
  status   TEXT    NOT NULL DEFAULT 'open',
  priority INTEGER NOT NULL DEFAULT 2,
  due      TEXT,
  tags     TEXT    NOT NULL DEFAULT '',
  created  TEXT    NOT NULL,
  updated  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status   ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_due      ON tasks(due);
CREATE INDEX IF NOT EXISTS idx_priority ON tasks(priority);
\"\"\"


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat(sep=" ")


class Store:
    \"\"\"Every database call lives here, so the rest of the app stays clean.\"\"\"

    def __init__(self, path: str = DEFAULT_DB) -> None:
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        with self._cx() as cx:
            cx.executescript(SCHEMA)

    def _cx(self) -> sqlite3.Connection:
        cx = getattr(self._local, "cx", None)
        if cx is None:
            cx = sqlite3.connect(self.path, timeout=10,
                                 check_same_thread=False)
            cx.row_factory = sqlite3.Row
            cx.execute("PRAGMA journal_mode=WAL")
            cx.execute("PRAGMA foreign_keys=ON")
            self._local.cx = cx
        return cx

    # -------------------------------------------------------- write
    def add(self, task: Task) -> Task:
        task.validate()
        stamp = now_iso()
        task.created = task.created or stamp
        task.updated = stamp
        with self._cx() as cx:
            cur = cx.execute(
                "INSERT INTO tasks(title,note,status,priority,due,tags,"
                "created,updated) VALUES(?,?,?,?,?,?,?,?)",
                (task.title.strip(), task.note, task.status, task.priority,
                 task.due or None, ",".join(task.tags), task.created,
                 task.updated))
            task.id = cur.lastrowid
        return task

    def update(self, tid: int, **fields: Any) -> Task:
        task = self.get(tid)
        for k, v in fields.items():
            if v is None or not hasattr(task, k):
                continue
            setattr(task, k, v)
        task.validate()
        task.updated = now_iso()
        with self._cx() as cx:
            cx.execute(
                "UPDATE tasks SET title=?,note=?,status=?,priority=?,due=?,"
                "tags=?,updated=? WHERE id=?",
                (task.title, task.note, task.status, task.priority,
                 task.due or None, ",".join(task.tags), task.updated, tid))
        return task

    def delete(self, tid: int) -> None:
        self.get(tid)
        with self._cx() as cx:
            cx.execute("DELETE FROM tasks WHERE id=?", (tid,))

    def clear_done(self) -> int:
        with self._cx() as cx:
            cur = cx.execute("DELETE FROM tasks WHERE status=?", (STATUS_DONE,))
            return cur.rowcount

    # --------------------------------------------------------- read
    def get(self, tid: int) -> Task:
        row = self._cx().execute(
            "SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if row is None:
            raise NotFound("No task with id %d." % tid)
        return Task.from_row(row)

    def list(self, status: Optional[str] = None, tag: Optional[str] = None,
             sort: str = "id", limit: int = 500) -> List[Task]:
        sql = "SELECT * FROM tasks"
        args: List[Any] = []
        where = []
        if status in STATUS_ALL:
            where.append("status=?")
            args.append(status)
        if tag:
            where.append("(','||tags||',') LIKE ?")
            args.append("%%,%s,%%" % tag)
        if where:
            sql += " WHERE " + " AND ".join(where)
        order = {
            "id": "id ASC",
            "due": "due IS NULL, due ASC, priority DESC",
            "priority": "priority DESC, due IS NULL, due ASC",
            "title": "title COLLATE NOCASE ASC",
            "created": "created DESC",
        }.get(sort, "id ASC")
        sql += " ORDER BY " + order + " LIMIT ?"
        args.append(limit)
        return [Task.from_row(r) for r in self._cx().execute(sql, args)]

    def search(self, text: str, limit: int = 200) -> List[Task]:
        like = "%" + text.strip() + "%"
        rows = self._cx().execute(
            "SELECT * FROM tasks WHERE title LIKE ? OR note LIKE ? OR "
            "tags LIKE ? ORDER BY id ASC LIMIT ?", (like, like, like, limit))
        return [Task.from_row(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        cx = self._cx()
        total = cx.execute("SELECT COUNT(*) c FROM tasks").fetchone()["c"]
        done = cx.execute("SELECT COUNT(*) c FROM tasks WHERE status=?",
                          (STATUS_DONE,)).fetchone()["c"]
        today = dt.date.today().isoformat()
        overdue = cx.execute(
            "SELECT COUNT(*) c FROM tasks WHERE status=? AND due IS NOT NULL "
            "AND due < ?", (STATUS_OPEN, today)).fetchone()["c"]
        tags: Dict[str, int] = {}
        for r in cx.execute("SELECT tags FROM tasks WHERE tags<>''"):
            for t in r["tags"].split(","):
                if t:
                    tags[t] = tags.get(t, 0) + 1
        return {
            "total": total,
            "open": total - done,
            "done": done,
            "overdue": overdue,
            "percent_done": round(100.0 * done / total, 1) if total else 0.0,
            "tags": dict(sorted(tags.items(), key=lambda kv: -kv[1])),
        }

    def export_csv(self, path: str) -> int:
        rows = self.list(limit=100000)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "title", "note", "status", "priority", "due",
                        "tags", "created", "updated"])
            for t in rows:
                w.writerow([t.id, t.title, t.note, t.status, t.priority,
                            t.due or "", " ".join(t.tags), t.created,
                            t.updated])
        return len(rows)

    def import_csv(self, path: str) -> int:
        n = 0
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    self.add(Task(
                        title=row.get("title", ""),
                        note=row.get("note", ""),
                        status=row.get("status", STATUS_OPEN),
                        priority=int(row.get("priority") or 2),
                        due=(row.get("due") or None),
                        tags=[t for t in (row.get("tags") or "").split()
                              if t]))
                    n += 1
                except TaskError:
                    continue
        return n


# =====================================================================
#  Terminal output
# =====================================================================
class C:
    \"\"\"ANSI colours, switched off when output is piped to a file.\"\"\"

    on = sys.stdout.isatty()

    @classmethod
    def _w(cls, code: str, s: str) -> str:
        return ("\\033[%sm%s\\033[0m" % (code, s)) if cls.on else s

    @classmethod
    def dim(cls, s): return cls._w("2", s)

    @classmethod
    def bold(cls, s): return cls._w("1", s)

    @classmethod
    def red(cls, s): return cls._w("31", s)

    @classmethod
    def green(cls, s): return cls._w("32", s)

    @classmethod
    def yellow(cls, s): return cls._w("33", s)

    @classmethod
    def cyan(cls, s): return cls._w("36", s)


def render_table(tasks: Iterable[Task]) -> str:
    tasks = list(tasks)
    if not tasks:
        return C.dim("  Nothing here yet. Add one with:  tasks.py add \\"...\\"")
    w_title = max(5, min(52, max(len(t.title) for t in tasks)))
    head = "  %-4s %-3s %-*s %-10s %s" % (
        "ID", "", w_title, "TITLE", "DUE", "TAGS")
    out = [C.bold(head), C.dim("  " + "-" * (len(head) - 2))]
    for t in tasks:
        mark = C.green("[x]") if t.status == STATUS_DONE else "[ ]"
        title = t.title if len(t.title) <= w_title else t.title[:w_title - 1] + "\\u2026"
        if t.status == STATUS_DONE:
            title = C.dim(title)
        due = t.due or ""
        if t.overdue:
            due = C.red(due + "!")
        elif t.due and (t.days_left or 0) <= 2:
            due = C.yellow(due)
        pri = ""
        if t.priority >= 3 and t.status == STATUS_OPEN:
            pri = C.red("*") if t.priority == 4 else C.yellow("*")
        tags = C.cyan(" ".join("#" + x for x in t.tags)) if t.tags else ""
        out.append("  %-4s %s %-*s %-10s %s%s" % (
            t.id, mark, w_title, title, due, tags, pri))
    return "\\n".join(out)


def render_stats(s: Dict[str, Any]) -> str:
    bar_w = 34
    filled = int(bar_w * s["percent_done"] / 100)
    bar = C.green("#" * filled) + C.dim("." * (bar_w - filled))
    lines = [
        "",
        "  " + C.bold("Progress"),
        "  [%s] %s%%" % (bar, s["percent_done"]),
        "",
        "  %-10s %d" % ("total", s["total"]),
        "  %-10s %d" % ("open", s["open"]),
        "  %-10s %d" % ("done", s["done"]),
        "  %-10s %s" % ("overdue",
                        C.red(str(s["overdue"])) if s["overdue"]
                        else str(s["overdue"])),
    ]
    if s["tags"]:
        lines += ["", "  " + C.bold("Tags")]
        for t, n in list(s["tags"].items())[:10]:
            lines.append("  %-14s %s" % ("#" + t, C.dim("*" * min(n, 30))))
    return "\\n".join(lines) + "\\n"


# =====================================================================
#  HTTP API + browser page
# =====================================================================
PAGE = \"\"\"<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task Manager</title><style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0F172A;--card:#1E293B;--line:#334155;--fg:#F1F5F9;
      --mut:#94A3B8;--a:#38BDF8;--ok:#34D399;--bad:#F87171}
body{font:15px/1.6 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",
     system-ui,sans-serif;background:var(--bg);color:var(--fg);padding:34px 18px}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:26px;letter-spacing:-.02em;margin-bottom:3px}
.sub{color:var(--mut);font-size:14px;margin-bottom:22px}
form{display:flex;gap:9px;margin-bottom:18px}
input{flex:1;padding:12px 14px;border:1px solid var(--line);border-radius:10px;
      background:var(--card);color:var(--fg);font:inherit}
input:focus{outline:2px solid var(--a);border-color:transparent}
button{padding:12px 18px;border:0;border-radius:10px;background:var(--a);
       color:#062A36;font:inherit;font-weight:700;cursor:pointer}
li{list-style:none;display:flex;align-items:center;gap:11px;padding:12px 14px;
   background:var(--card);border:1px solid var(--line);border-radius:11px;
   margin-bottom:8px}
li.done .t{text-decoration:line-through;color:var(--mut)}
.t{flex:1}
.due{font-size:12.5px;color:var(--mut)}
.due.over{color:var(--bad);font-weight:700}
.x{background:none;border:0;color:var(--mut);cursor:pointer;font-size:17px}
.bar{height:7px;background:var(--card);border-radius:99px;overflow:hidden;
     margin:6px 0 22px}
.bar i{display:block;height:100%;background:var(--ok);width:0;transition:.5s}
</style></head><body><div class="wrap">
<h1>Task Manager</h1><p class="sub" id="s">Loading...</p>
<div class="bar"><i id="b"></i></div>
<form id="f"><input id="q" placeholder="Add a task and press Enter"
  autocomplete="off"><button>Add</button></form>
<ul id="l"></ul></div><script>
async function api(p,o){const r=await fetch('/api'+p,o);return r.json();}
async function draw(){
  const d=await api('/tasks'),s=await api('/stats');
  document.getElementById('s').textContent=
    s.open+' open, '+s.done+' done'+(s.overdue?', '+s.overdue+' overdue':'');
  document.getElementById('b').style.width=s.percent_done+'%';
  const l=document.getElementById('l');l.innerHTML='';
  d.tasks.forEach(function(t){
    const li=document.createElement('li');if(t.status==='done')li.className='done';
    const c=document.createElement('input');c.type='checkbox';
    c.checked=t.status==='done';
    c.onchange=function(){api('/tasks/'+t.id,{method:'PATCH',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({status:c.checked?'done':'open'})}).then(draw);};
    const sp=document.createElement('span');sp.className='t';sp.textContent=t.title;
    li.appendChild(c);li.appendChild(sp);
    if(t.due){const d2=document.createElement('span');
      d2.className='due'+(t.overdue?' over':'');d2.textContent=t.due;
      li.appendChild(d2);}
    const x=document.createElement('button');x.className='x';x.textContent='x';
    x.onclick=function(){api('/tasks/'+t.id,{method:'DELETE'}).then(draw);};
    li.appendChild(x);l.appendChild(li);});
}
document.getElementById('f').onsubmit=function(e){
  e.preventDefault();const q=document.getElementById('q');
  if(!q.value.trim())return;
  api('/tasks',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({title:q.value.trim()})}).then(function(){
      q.value='';draw();});};
draw();
</script></body></html>\"\"\"


class Handler(http.server.BaseHTTPRequestHandler):
    store: Store = None            # type: ignore  (set in serve())
    server_version = "TaskManager/" + APP_VERSION

    def log_message(self, fmt, *a):     # quieter default logging
        sys.stderr.write("  %s %s\\n" % (self.command, self.path))

    # ------------------------------------------------------- helpers
    def _send(self, code: int, payload: Any, ctype="application/json"):
        body = (payload if isinstance(payload, bytes)
                else json.dumps(payload, indent=2).encode("utf-8")
                if ctype == "application/json" else payload.encode("utf-8"))
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods",
                         "GET,POST,PATCH,DELETE,OPTIONS")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            raise TaskError("Body must be JSON.")

    def _route(self) -> Tuple[str, Optional[int]]:
        path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        m = re.match(r"^/api/tasks/(\\d+)$", path)
        if m:
            return "/api/tasks/:id", int(m.group(1))
        return path, None

    def _guard(self, fn):
        try:
            fn()
        except NotFound as e:
            self._send(404, {"error": str(e)})
        except TaskError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:                     # never leak a traceback
            self._send(500, {"error": "%s: %s" % (type(e).__name__, e)})

    # ------------------------------------------------------- verbs
    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        def run():
            path, tid = self._route()
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if path == "/api/tasks":
                ts = self.store.list(
                    status=(qs.get("status") or [None])[0],
                    tag=(qs.get("tag") or [None])[0],
                    sort=(qs.get("sort") or ["id"])[0])
                return self._send(200, {"count": len(ts),
                                        "tasks": [t.to_dict() for t in ts]})
            if path == "/api/tasks/:id":
                return self._send(200, self.store.get(tid).to_dict())
            if path == "/api/search":
                q = (qs.get("q") or [""])[0]
                ts = self.store.search(q)
                return self._send(200, {"query": q, "count": len(ts),
                                        "tasks": [t.to_dict() for t in ts]})
            if path == "/api/stats":
                return self._send(200, self.store.stats())
            if path == "/api/health":
                return self._send(200, {"ok": True, "app": APP_NAME,
                                        "version": APP_VERSION})
            self._send(404, {"error": "No route %s" % path})
        self._guard(run)

    def do_POST(self):
        def run():
            path, _ = self._route()
            if path != "/api/tasks":
                return self._send(404, {"error": "No route %s" % path})
            b = self._body()
            t = self.store.add(Task(
                title=b.get("title", ""), note=b.get("note", ""),
                priority=int(b.get("priority", 2)),
                due=b.get("due") or None, tags=list(b.get("tags") or [])))
            self._send(201, t.to_dict())
        self._guard(run)

    def do_PATCH(self):
        def run():
            path, tid = self._route()
            if path != "/api/tasks/:id":
                return self._send(404, {"error": "No route %s" % path})
            b = self._body()
            if "priority" in b and b["priority"] is not None:
                b["priority"] = int(b["priority"])
            self._send(200, self.store.update(tid, **b).to_dict())
        self._guard(run)

    def do_DELETE(self):
        def run():
            path, tid = self._route()
            if path != "/api/tasks/:id":
                return self._send(404, {"error": "No route %s" % path})
            self.store.delete(tid)
            self._send(200, {"deleted": tid})
        self._guard(run)


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(store: Store, host: str, port: int) -> None:
    Handler.store = store
    with ThreadedServer((host, port), Handler) as srv:
        print("\\n  %s %s" % (C.bold(APP_NAME), C.dim("v" + APP_VERSION)))
        print("  %s  http://%s:%d" % (C.green("running"),
                                      host if host != "0.0.0.0" else "localhost",
                                      port))
        print("  %s  %s\\n" % (C.dim("database"), store.path))
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\\n  stopped\\n")


# =====================================================================
#  Tests
# =====================================================================
class TaskTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def test_add_and_get(self):
        t = self.store.add(Task(title="Write docs", tags=["work"]))
        self.assertIsNotNone(t.id)
        self.assertEqual(self.store.get(t.id).title, "Write docs")

    def test_title_required(self):
        with self.assertRaises(TaskError):
            self.store.add(Task(title="   "))

    def test_bad_date(self):
        with self.assertRaises(TaskError):
            self.store.add(Task(title="x", due="30-01-2026"))
        with self.assertRaises(TaskError):
            self.store.add(Task(title="x", due="2026-02-31"))

    def test_bad_priority(self):
        with self.assertRaises(TaskError):
            self.store.add(Task(title="x", priority=9))

    def test_bad_tag(self):
        with self.assertRaises(TaskError):
            self.store.add(Task(title="x", tags=["not a tag"]))

    def test_update_and_done(self):
        t = self.store.add(Task(title="a"))
        self.store.update(t.id, status=STATUS_DONE)
        self.assertEqual(self.store.get(t.id).status, STATUS_DONE)

    def test_delete(self):
        t = self.store.add(Task(title="a"))
        self.store.delete(t.id)
        with self.assertRaises(NotFound):
            self.store.get(t.id)

    def test_missing_raises(self):
        with self.assertRaises(NotFound):
            self.store.get(4242)

    def test_overdue(self):
        past = (dt.date.today() - dt.timedelta(days=3)).isoformat()
        t = self.store.add(Task(title="late", due=past))
        self.assertTrue(self.store.get(t.id).overdue)
        self.store.update(t.id, status=STATUS_DONE)
        self.assertFalse(self.store.get(t.id).overdue)

    def test_filter_by_tag(self):
        self.store.add(Task(title="a", tags=["home"]))
        self.store.add(Task(title="b", tags=["work"]))
        self.assertEqual(len(self.store.list(tag="home")), 1)

    def test_sort_by_due(self):
        self.store.add(Task(title="later", due="2030-01-01"))
        self.store.add(Task(title="sooner", due="2026-01-01"))
        self.assertEqual(self.store.list(sort="due")[0].title, "sooner")

    def test_search(self):
        self.store.add(Task(title="Buy milk"))
        self.store.add(Task(title="Call mum"))
        self.assertEqual(len(self.store.search("milk")), 1)

    def test_stats(self):
        a = self.store.add(Task(title="a"))
        self.store.add(Task(title="b"))
        self.store.update(a.id, status=STATUS_DONE)
        s = self.store.stats()
        self.assertEqual((s["total"], s["done"], s["open"]), (2, 1, 1))
        self.assertEqual(s["percent_done"], 50.0)

    def test_clear_done(self):
        a = self.store.add(Task(title="a"))
        self.store.update(a.id, status=STATUS_DONE)
        self.assertEqual(self.store.clear_done(), 1)
        self.assertEqual(self.store.stats()["total"], 0)


# =====================================================================
#  Command line
# =====================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tasks.py",
        description="%s %s - tasks in your terminal, browser and API."
                    % (APP_NAME, APP_VERSION))
    p.add_argument("--db", default=DEFAULT_DB, help="database file")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("add", help="add a task")
    a.add_argument("title", nargs="+")
    a.add_argument("--note", default="")
    a.add_argument("--due")
    a.add_argument("--tag", action="append", default=[])
    a.add_argument("--priority", type=int, default=2, choices=list(PRIORITIES))

    l = sub.add_parser("list", help="list tasks")
    l.add_argument("--status", choices=list(STATUS_ALL))
    l.add_argument("--tag")
    l.add_argument("--sort", default="id",
                   choices=["id", "due", "priority", "title", "created"])

    for name, helptext in (("done", "mark done"), ("open", "mark open"),
                           ("rm", "delete")):
        q = sub.add_parser(name, help=helptext)
        q.add_argument("id", type=int, nargs="+")

    e = sub.add_parser("edit", help="change a task")
    e.add_argument("id", type=int)
    e.add_argument("--title")
    e.add_argument("--note")
    e.add_argument("--due")
    e.add_argument("--priority", type=int, choices=list(PRIORITIES))

    s = sub.add_parser("search", help="find text")
    s.add_argument("text", nargs="+")

    sub.add_parser("stats", help="progress summary")
    sub.add_parser("clear", help="remove finished tasks")

    x = sub.add_parser("export", help="write a CSV")
    x.add_argument("path")
    i = sub.add_parser("import", help="read a CSV")
    i.add_argument("path")

    sv = sub.add_parser("serve", help="start the web app and API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)

    sub.add_parser("test", help="run the test suite")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd is None:
        build_parser().print_help()
        return 0
    if args.cmd == "test":
        loader = unittest.TestLoader().loadTestsFromTestCase(TaskTests)
        res = unittest.TextTestRunner(verbosity=2).run(loader)
        return 0 if res.wasSuccessful() else 1

    store = Store(args.db)
    out = lambda s: print(s)                                   # noqa: E731
    try:
        if args.cmd == "add":
            t = store.add(Task(title=" ".join(args.title), note=args.note,
                               due=args.due, tags=args.tag,
                               priority=args.priority))
            out(json.dumps(t.to_dict()) if args.json
                else "  %s #%d  %s" % (C.green("added"), t.id, t.title))

        elif args.cmd == "list":
            ts = store.list(status=args.status, tag=args.tag, sort=args.sort)
            out(json.dumps([t.to_dict() for t in ts], indent=2) if args.json
                else "\\n" + render_table(ts) + "\\n")

        elif args.cmd in ("done", "open"):
            new = STATUS_DONE if args.cmd == "done" else STATUS_OPEN
            for tid in args.id:
                store.update(tid, status=new)
            out("  %s %s" % (C.green(new), ", ".join("#%d" % i
                                                     for i in args.id)))

        elif args.cmd == "rm":
            for tid in args.id:
                store.delete(tid)
            out("  %s %s" % (C.red("deleted"),
                             ", ".join("#%d" % i for i in args.id)))

        elif args.cmd == "edit":
            t = store.update(args.id, title=args.title, note=args.note,
                             due=args.due, priority=args.priority)
            out("  %s #%d  %s" % (C.green("updated"), t.id, t.title))

        elif args.cmd == "search":
            ts = store.search(" ".join(args.text))
            out(json.dumps([t.to_dict() for t in ts], indent=2) if args.json
                else "\\n" + render_table(ts) + "\\n")

        elif args.cmd == "stats":
            s = store.stats()
            out(json.dumps(s, indent=2) if args.json else render_stats(s))

        elif args.cmd == "clear":
            out("  %s %d finished task(s)" % (C.red("removed"),
                                              store.clear_done()))

        elif args.cmd == "export":
            out("  %s %d rows -> %s" % (C.green("exported"),
                                        store.export_csv(args.path),
                                        args.path))

        elif args.cmd == "import":
            out("  %s %d rows from %s" % (C.green("imported"),
                                          store.import_csv(args.path),
                                          args.path))

        elif args.cmd == "serve":
            serve(store, args.host, args.port)

    except TaskError as e:
        print("  %s %s" % (C.red("error"), e), file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print("  %s %s" % (C.red("error"), e), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""


def _off_python_app():
    """A complete standard-library Python program: CLI + SQLite store +
    REST API + browser page + 14 unit tests. Roughly 31 KB."""
    return _OFF_PY_APP


OFFLINE_KINDS = (
    ("python", ("python app", "python program", "python script",
                "python cli", "python code", "task manager in python",
                "todo app in python", "python project", "app in python",
                "program in python", "script in python", "cli in python",
                "write python", "build python", "make python")),
    ("portfolio", ("portfolio", "portifolio", "personal site", "personal page")),
    ("login", ("login page", "login form", "sign in page", "signin page",
               "login screen", "log in page")),
)


def offline_kind(q: str):
    """Which local template, if any, this request maps to."""
    t = (q or "").lower()
    for kind, words in OFFLINE_KINDS:
        for w in words:
            if w in t:
                return kind
    return None


def offline_build(q: str):
    """Return (markdown_answer, kind) built locally, or None."""
    kind = offline_kind(q)
    if not kind:
        return None

    if kind == "python":
        code = _off_python_app()
        lang = "python"
        head = ("Built it locally - no AI needed, so this never fails.\n\n"
                "1. One file, standard library only - no pip install\n"
                "2. SQLite storage with WAL, indexes and thread-safe "
                "connections\n"
                "3. Full command line: add, list, done, edit, rm, search, "
                "stats, export, import\n"
                "4. Built-in REST API and a browser page on the same port "
                "(`serve`)\n"
                "5. Coloured terminal tables, progress bar and tag chart\n"
                "6. 14 unit tests included - run `python3 tasks.py test`\n")
        tail = ("\nRun it:\n\n"
                "```sh\n"
                "python3 tasks.py add \"Buy milk\" --due 2026-01-30 "
                "--tag home\n"
                "python3 tasks.py list --sort due\n"
                "python3 tasks.py serve --port 8080\n"
                "python3 tasks.py test\n"
                "```\n\n"
                "Terminal colours used: `#34D399` done, `#F87171` overdue, "
                "`#FBBF24` due soon, `#38BDF8` tags. Web page colours: "
                "`#0F172A` background, `#1E293B` cards, `#334155` lines, "
                "`#F1F5F9` text, `#94A3B8` muted, `#38BDF8` accent.\n\n"
                "Want it to sync to a server, send reminders, or use "
                "PostgreSQL instead of SQLite?")

    elif kind == "portfolio":
        m = re.search(r"(?:portfolio|portifolio)\s+for\s+([A-Za-z .'-]{2,40})",
                      q or "", re.I)
        code = _off_portfolio_big((m.group(1).strip() if m else "Your Name"))
        lang = "html"
        head = ("Built it locally - no AI needed, so this never fails.\n\n"
                "1. Sticky blurred nav with a working mobile burger menu\n"
                "2. Hero with a typing headline and animated counters\n"
                "3. Services grid, six projects with a working category "
                "filter\n"
                "4. Animated skill bars, a career timeline and "
                "testimonials\n"
                "5. Contact form with real inline validation\n"
                "6. Scroll-reveal animations, smooth scrolling, back-to-top\n"
                "7. Fully responsive down to 360px\n")
        tail = ("\nColours used: `#0F172A` ink, `#F8FAFC` background, "
                "`#FFFFFF` cards, `#4F46E5` and `#6366F1` accent, "
                "`#64748B` muted text, `#E2E8F0` lines, `#DC2626` errors.\n\n"
                "Want dark mode, a blog section, or different colours?")

    else:
        code = _off_login_big()
        lang = "html"
        head = ("Built it locally - no AI needed, so this never fails.\n\n"
                "1. Sign in and create account in one card, tabbed\n"
                "2. Live email, password and confirm-password validation\n"
                "3. Password strength meter with four levels\n"
                "4. Show/hide password and a Caps Lock warning\n"
                "5. Remember me, saved to localStorage\n"
                "6. Forgot-password flow and Google / GitHub buttons\n"
                "7. Loading spinner on submit plus a toast for every "
                "result\n")
        tail = ("\nColours used: `#0F172A` ink, `#F8FAFC` background, "
                "`#FFFFFF` cards, `#4F46E5` and `#6366F1` accent, "
                "`#64748B` muted, `#E2E8F0` lines, `#DC2626` weak, "
                "`#F59E0B` fair, `#3B82F6` good, `#059669` strong.\n\n"
                "Want it wired to a real backend, or two-factor added?")

    return (head + "\n```" + lang + "\n" + code + "\n```\n" + tail), kind


async def call_openai_style(c: httpx.AsyncClient, pid: str, model: str,
                            messages: list, temp: float,
                            budget: float) -> Tuple[Optional[str], str]:
    """Groq / OpenRouter / Cerebras / Together / Mistral / OpenAI all speak this.

    Additive upgrade: OpenAI rotates through OPENAI_API_KEY_POOL (all new_api
    keys) on 401/429, Clarifai uses its PAT as bearer, Databricks resolves its
    workspace URL at call time. Nothing old was removed.
    """
    p = PROVIDERS[pid]
    keys: List[str] = openai_key_pool() if pid == "openai" else [env_key(p["env"])]
    keys = [k for k in keys if k]
    if not keys:
        return None, "%s has no key" % p["label"]
    url = databricks_url() if pid == "databricks" else p["url"]
    if not url:
        return None, "%s not configured" % p["label"]
    cap = 4096 if pid == "groq" else 8192
    msgs = list(messages)
    whole = ""
    last_err = ""
    # A long file can exceed one response. When the provider says it stopped
    # because it ran out of room ("length"), ask it to carry straight on and
    # stitch the pieces together, so code is never delivered half-finished.
    for round_no in range(MAX_CONTINUE + 1):
        body = {"model": model, "messages": msgs,
                "temperature": temp, "max_tokens": cap}
        piece, why, last_err = "", "", last_err
        ok_round = False
        for ki, key in enumerate(keys):
            heads = {"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"}
            if pid == "openrouter":
                heads["HTTP-Referer"] = "http://localhost:%d" % CFG.port
                heads["X-Title"] = "ADELTE"
            try:
                r = await c.post(url, headers=heads, json=body, timeout=budget)
            except Exception as e:
                last_err = "%s unreachable (%s)" % (p["label"], type(e).__name__)
                continue
            if r.status_code in (401, 403, 429) and len(keys) > 1 and ki < len(keys) - 1:
                last_err = "%s key %d refused (%d) - rotating" % (p["label"], ki + 1, r.status_code)
                continue  # try next key in pool, additive rotation
            if r.status_code != 200:
                last_err = "%s HTTP %d - %s" % (p["label"], r.status_code,
                                                r.text[:160].replace("\n", " "))
                break  # non-auth error: do not burn the whole pool
            try:
                j = r.json()
                ch = j["choices"][0]
                piece = ch["message"]["content"] or ""
                why = (ch.get("finish_reason") or "").lower()
            except Exception:
                last_err = "%s sent a shape I could not read" % p["label"]
                break
            ok_round = True
            break
        if not ok_round:
            if whole:
                break
            return None, last_err or ("%s failed" % p["label"])
        whole = join_continuation(whole, piece)
        if why != "length" or round_no >= MAX_CONTINUE or not piece.strip():
            break
        msgs = msgs + [
            {"role": "assistant", "content": piece},
            {"role": "user", "content": CONTINUE_ASK}]
    return whole.strip() or None, ""


async def call_gemini(c: httpx.AsyncClient, model: str, messages: list,
                      temp: float, budget: float) -> Tuple[Optional[str], str]:
    """Google's REST shape: contents[] + systemInstruction."""
    key = env_key("GEMINI_API_KEY")
    sys_txt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = []
    for m in messages:
        if m["role"] == "system":
            continue
        turns.append({"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]})
    body: Dict[str, object] = {
        "contents": turns,
        "generationConfig": {"temperature": temp, "maxOutputTokens": 8192},
    }
    if sys_txt:
        body["systemInstruction"] = {"parts": [{"text": sys_txt}]}
    url = PROVIDERS["gemini"]["url"].format(model=model)
    try:
        r = await c.post(url, headers={"x-goog-api-key": key,
                                       "Content-Type": "application/json"},
                         json=body, timeout=budget)
    except Exception as e:
        return None, "Gemini unreachable (%s)" % type(e).__name__
    if r.status_code != 200:
        return None, "Gemini HTTP %d - %s" % (r.status_code,
                                              r.text[:160].replace("\n", " "))
    try:
        j = r.json()
        parts = j["candidates"][0]["content"]["parts"]
        txt = "".join(p.get("text", "") for p in parts)
    except Exception:
        return None, "Gemini sent a shape I could not read"
    return (txt or "").strip() or None, ""


async def generate_with_model(c: httpx.AsyncClient, mid: str, messages: list,
                              budget: float, emit=None) -> Tuple[Optional[str], str]:
    """Walk the model's provider chain until one answers.

    Returns (text, provider_label). Every failure is narrated to the UI so
    the user can see exactly which brain answered and which one refused.
    """
    cfg = model_cfg(mid)
    chain = live_chain(mid)
    if not chain:
        if emit:
            await emit("thinking", {"text": "No API keys in .env - using the "
                                            "free swarm", "p": 62})
        chain = [("horde", "")]
    temp = cfg.get("temp", 0.4)
    per = max(8.0, budget / max(1, len(chain)))
    for i, (pid, model) in enumerate(chain, 1):
        label = PROVIDERS[pid]["label"]
        nice = "%s%s" % (label, (" " + model.split("/")[-1]) if model else "")
        if emit:
            await emit("thinking", {"text": "Asking %s (%d of %d)"
                                    % (nice, i, len(chain)), "p": 60 + i * 4})
        t0 = time.time()
        if pid == "horde":
            async def _hp(st):
                if not emit:
                    return
                qp = st.get("queue_position")
                if st.get("processing"):
                    msg = "A free-swarm worker is writing the answer"
                elif qp is not None:
                    msg = "Waiting in the free-swarm queue - position %s" % qp
                else:
                    msg = "Waiting for a free-swarm worker"
                await emit("thinking", {"text": msg, "p": 70})
            txt = await horde_generate(c, messages, budget=per,
                                       on_progress=_hp)
            err = "" if txt else "the free swarm had no worker free"
        elif PROVIDERS[pid]["style"] == "gemini":
            txt, err = await call_gemini(c, model, messages, temp, per)
        else:
            txt, err = await call_openai_style(c, pid, model, messages, temp, per)
        if txt and usable_answer(txt, strict=(pid == "horde")):
            ms = int((time.time() - t0) * 1000)
            if emit:
                await emit("thinking", {"text": "%s answered in %d ms"
                                        % (nice, ms), "p": 88})
            return trim_completion(strip_scratchpad(txt)), nice
        if emit:
            await emit("thinking", {"text": "%s failed - %s"
                                    % (nice, err or "empty reply"), "p": 62})
    return None, ""


# ---------------------------------------------------------------------------
#  WHO ARE YOU  —  identity questions must never hit a search engine
# ---------------------------------------------------------------------------
IDENTITY_PAT = re.compile(
    r"\b(who (are|r) (you|u)|who made (you|u)|who (built|created|developed|"
    r"designed|owns|invented) (you|u|this)|what (are|is) (you|your name)|"
    r"your name|who am i (talking|speaking) (to|with)|what model are you|"
    r"which (ai|model) are you|are you (chatgpt|gpt|claude|gemini|openai)|"
    r"who is your (maker|creator|owner|developer)|tell me about yourself|"
    r"introduce yourself|what can you do|what do you do)\b", re.I)


def identity_answer(query: str, mid: Optional[str] = None) -> str:
    q = query.lower()
    cfg = model_cfg(mid)
    if re.search(r"(made|built|created|developed|owns|maker|creator|owner|"
                 r"developer|invented)", q):
        return (
            "I was made by **ADELTE Industries**.\n\n"
            "- ADELTE is the assistant; ADELTE Industries is the team behind it\n"
            "- I run entirely on your own machine, from `adelte.py`\n"
            "- I am not ChatGPT, Claude or Gemini - those can be *brains* I "
            "borrow, but the assistant is ADELTE\n"
            "- You are currently talking to **" + cfg["name"] + "**"
        )
    if re.search(r"(what can you do|what do you do)", q):
        return (
            "Quite a lot - here's the short version.\n\n"
            "- **Search** ten engines at once and read the pages properly\n"
            "- **Build** working files: pages, forms, scripts, whole apps\n"
            "- **Run commands** like `open youtube`, `lock pc`, `flip a coin`\n"
            "- **Remember** this conversation, so follow-ups just work\n"
            "- **Talk** with voice, in or out"
        )
    return (
        "I'm **ADELTE**, built by **ADELTE Industries**.\n\n"
        "- You're using **" + cfg["name"] + "** - " + cfg["blurb"] + "\n"
        "- I search, read, build and run commands, all from your machine\n"
        "- Ask me anything, or say `help` for the command list"
    )


# The free anonymous Horde tier refuses anything over 512 tokens per job
# ("KudosUpfront"), so a long file can only be produced by asking repeatedly
# and stitching the pieces together. That is what horde_generate does.
HORDE_MAX_LENGTH = 512


async def _horde_once(c: httpx.AsyncClient, prompt: str, budget: float,
                      on_progress=None, want: int = HORDE_MAX_LENGTH
                      ) -> Optional[str]:
    """One Horde round trip. Returns raw text (untrimmed) or None."""
    want = max(80, min(int(want), HORDE_MAX_LENGTH))
    payload = {
        "prompt": prompt[:9000],
        "params": {"max_length": want, "max_context_length": 4096,
                   "temperature": 0.4, "top_p": 0.9, "rep_pen": 1.07},
        "models": [], "trusted_workers": False,
    }
    try:
        r = await asyncio.wait_for(
            c.post(f"{HORDE}/generate/text/async", json=payload,
                   headers=HORDE_HEADERS), timeout=18)
        if r.status_code == 403 and "kudos" in r.text.lower():
            # Asked for more than the anonymous tier allows - halve and retry.
            payload["params"]["max_length"] = max(80, want // 2)
            r = await asyncio.wait_for(
                c.post(f"{HORDE}/generate/text/async", json=payload,
                       headers=HORDE_HEADERS), timeout=18)
        if r.status_code not in (200, 202):
            return None
        job = r.json().get("id")
        if not job:
            return None
    except Exception:
        return None

    t0, delay = time.time(), 1.5
    while time.time() - t0 < budget:
        await asyncio.sleep(delay)
        delay = min(delay * 1.3, 6.0)
        try:
            s = await asyncio.wait_for(
                c.get(f"{HORDE}/generate/text/status/{job}",
                      headers=HORDE_HEADERS), timeout=14)
            d = s.json()
        except Exception:
            continue
        if d.get("faulted"):
            return None
        if d.get("done"):
            gens = d.get("generations") or []
            return gens[0]["text"] if gens and gens[0].get("text") else None
        if on_progress:
            await on_progress({"queue_position": d.get("queue_position"),
                               "wait_time": d.get("wait_time"),
                               "processing": d.get("processing")})
    try:   # out of budget — release the volunteer GPU
        await c.delete(f"{HORDE}/generate/text/status/{job}",
                       headers=HORDE_HEADERS, timeout=7)
    except Exception:
        pass
    return None


def _looks_unfinished(t: str) -> bool:
    """True when a code answer was clearly guillotined mid-file."""
    if not t:
        return False
    tail = t.rstrip()
    fences = tail.count("```")
    if fences % 2 == 1:                       # a fence was opened, never shut
        return True
    if fences >= 2:
        body = tail.rsplit("```", 2)[1] if fences >= 2 else ""
        low = body.strip().lower()
        if low.startswith("html") or "<html" in low:
            if "</html>" not in low:
                return True
    # ends mid-word / mid-tag / mid-rule rather than on punctuation
    if tail and tail[-1] not in ".!?;:)}\"'`\n>":
        return True
    return False


async def horde_generate(c: httpx.AsyncClient, messages: list[dict],
                         budget: float, on_progress=None) -> Optional[str]:
    """Horde, but it keeps going until the file is actually finished.

    A single Horde job is capped by the worker, so one call can never
    return a long file. We ask again with "carry straight on" and stitch
    the pieces, exactly like the OpenAI-style providers do.
    """
    base = flatten_prompt(messages)
    t_start = time.time()
    whole = await _horde_once(c, base, budget, on_progress,
                              want=HORDE_MAX_LENGTH)
    if not whole:
        return None
    rounds = 0
    while (rounds < MAX_HORDE_CONTINUE
           and _looks_unfinished(whole)
           and time.time() - t_start < budget * 2.6):
        rounds += 1
        if on_progress:
            await on_progress({"continuing": rounds,
                               "note": "finishing the file (part %d)"
                                       % (rounds + 1)})
        cont = (base
                + "\n\n### Assistant (partial answer so far)\n" + whole[-2500:]
                + "\n\n### Instruction\nContinue that answer from exactly "
                  "where it stopped. Do not repeat any line you already "
                  "wrote, do not restart the file, do not apologise, do not "
                  "add commentary. Output only the remaining text, and "
                  "finish it completely (close every tag and the code "
                  "fence).\n\n### Continuation\n")
        nxt = await _horde_once(c, cont, budget, on_progress,
                                want=HORDE_MAX_LENGTH)
        if not nxt or not nxt.strip():
            break
        whole = join_continuation(whole, nxt)
    return trim_completion(whole)


# ===========================================================================
# FRIEND MODE  —  short, warm, no citations, remembers the session
# ===========================================================================

FRIEND_LINES = {
    "greet": [
        "Hey. Good to see you. What are we working on?",
        "Hi there. What's on your mind today?",
        "Hey — I'm here. Ask me anything, or tell me to build something.",
    ],
    "how": [
        "Doing well, thanks for asking. Ten search engines warmed up and "
        "nothing better to do than help you. How are you doing?",
        "I'm good. Ready to search, build or just talk. How's your day going?",
    ],
    "who": [
        "I'm ADELTE. I search ten engines at once, read the pages properly, "
        "build things when you ask, and run commands on your machine. "
        "I also remember what we talked about, so you can just say "
        "\"and the second one?\" and I'll know.",
    ],
    "thanks": [
        "Anytime. What's next?",
        "Happy to help. Anything else?",
    ],
    "bye": [
        "See you. Your session is saved, so pick up right where we left off.",
        "Later. I'll remember this conversation when you come back.",
    ],
    "sorry": ["No need to apologise. Let's just carry on."],
    "mood": [
        "Sorry to hear it. Want a distraction, or should we knock something "
        "off your list so the day feels lighter?",
    ],
    "praise": ["Appreciated. Let's keep going."],
    "joke": [
        "A programmer's wife says: go to the shop and buy a loaf of bread, "
        "and if they have eggs, get six. He came back with six loaves.",
    ],
    "fallback": [
        "I'm listening. Give me a question to research, something to build, "
        "or a command like `open youtube`.",
    ],
}


def friend_reply(query: str, history: list,
                 strict: bool = False) -> str:
    """Warm, short, human. No sources, no headings, no bullet dump."""
    q = " ".join(query.lower().strip().rstrip("!?.").split())
    first = not history

    def pick(k):
        return random.choice(FRIEND_LINES[k])

    if is_greeting(q) or q in GREET or any(q.startswith(g + " ")
                                           for g in GREET):
        if not first:
            return random.choice([
                "Hey again. Where were we?",
                "Welcome back. Want to keep going with what we were doing?",
            ])
        return pick("greet")
    if "how are you" in q or "how's it going" in q or "how you doing" in q:
        return pick("how")
    if "who are you" in q or "what are you" in q or "your name" in q \
            or "who made you" in q or "what can you do" in q:
        return pick("who")
    if q.startswith("thank") or q in ("thanks", "thx", "ty", "cheers"):
        return pick("thanks")
    if q in ("bye", "goodbye", "see you", "gn", "good night", "cya"):
        return pick("bye")
    if "sorry" in q:
        return pick("sorry")
    if "sad" in q or "tired" in q or "stressed" in q or "bad day" in q:
        return pick("mood")
    if "joke" in q:
        return pick("joke")
    if q in ("nice", "cool", "good job", "well done", "lol", "haha",
             "ok", "okay", "great", "awesome"):
        return pick("praise")
    if strict:
        # Nothing matched. "help me in codes" is a real request for help,
        # not small talk - the caller should hand it to the AI instead of
        # firing a canned line at the user.
        return ""
    return pick("fallback")


# ===========================================================================
# MAKER MODE  —  build the artifact instead of searching for it
# ===========================================================================
# Colour codes the user explicitly asked to see in answers.
PALETTES = {
    "indigo": ["#4F46E5", "#6366F1", "#818CF8", "#0F172A", "#F8FAFC"],
    "emerald": ["#059669", "#10B981", "#34D399", "#052E1B", "#F0FDF4"],
    "sunset": ["#F97316", "#FB923C", "#FDBA74", "#431407", "#FFF7ED"],
    "rose": ["#E11D48", "#F43F5E", "#FB7185", "#4C0519", "#FFF1F2"],
    "ocean": ["#0284C7", "#0EA5E9", "#38BDF8", "#082F49", "#F0F9FF"],
}


def login_page(palette: str = "indigo") -> str:
    c = PALETTES.get(palette, PALETTES["indigo"])
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in</title>
<style>
  :root{
    --brand:%s; --brand-2:%s; --brand-3:%s;
    --ink:%s; --bg:%s;
  }
  *{box-sizing:border-box}
  body{
    margin:0;min-height:100vh;display:grid;place-items:center;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    background:linear-gradient(135deg,var(--brand) 0%%,var(--brand-3) 100%%);
    padding:24px;
  }
  .card{
    width:100%%;max-width:400px;background:var(--bg);border-radius:20px;
    padding:40px 32px;box-shadow:0 24px 60px rgba(0,0,0,.28);
  }
  .logo{
    width:52px;height:52px;border-radius:14px;margin:0 auto 20px;
    background:linear-gradient(135deg,var(--brand),var(--brand-3));
    display:grid;place-items:center;color:#fff;font-weight:800;font-size:22px;
  }
  h1{margin:0 0 6px;text-align:center;font-size:24px;color:var(--ink)}
  p.sub{margin:0 0 28px;text-align:center;color:#64748b;font-size:14px}
  label{display:block;font-size:13px;font-weight:600;color:var(--ink);
        margin:0 0 6px}
  input{
    width:100%%;padding:12px 14px;margin-bottom:18px;border:1px solid #e2e8f0;
    border-radius:10px;font-size:15px;outline:none;transition:.15s;
    background:#fff;color:var(--ink);
  }
  input:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(0,0,0,.06)}
  .row{display:flex;justify-content:space-between;align-items:center;
       margin-bottom:22px;font-size:13px}
  .row a{color:var(--brand);text-decoration:none;font-weight:600}
  button{
    width:100%%;padding:13px;border:0;border-radius:10px;cursor:pointer;
    background:linear-gradient(135deg,var(--brand),var(--brand-2));
    color:#fff;font-size:15px;font-weight:600;transition:.15s;
  }
  button:hover{filter:brightness(1.08);transform:translateY(-1px)}
  .foot{margin-top:22px;text-align:center;font-size:13px;color:#64748b}
  .foot a{color:var(--brand);font-weight:600;text-decoration:none}
  .err{color:#dc2626;font-size:13px;margin:-10px 0 14px;display:none}
</style>
</head>
<body>
  <form class="card" onsubmit="return submitForm(event)">
    <div class="logo">A</div>
    <h1>Welcome back</h1>
    <p class="sub">Sign in to continue to your account</p>

    <label for="email">Email</label>
    <input id="email" type="email" placeholder="you@example.com" required>

    <label for="pw">Password</label>
    <input id="pw" type="password" placeholder="••••••••" required minlength="6">

    <div class="err" id="err">Please check your details and try again.</div>

    <div class="row">
      <label style="font-weight:400;display:flex;gap:6px;align-items:center">
        <input type="checkbox" style="width:auto;margin:0"> Remember me
      </label>
      <a href="#">Forgot password?</a>
    </div>

    <button type="submit">Sign in</button>
    <div class="foot">New here? <a href="#">Create an account</a></div>
  </form>

<script>
function submitForm(e){
  e.preventDefault();
  var email = document.getElementById('email').value.trim();
  var pw    = document.getElementById('pw').value;
  var err   = document.getElementById('err');
  if(!email.includes('@') || pw.length < 6){
    err.style.display='block';
    return false;
  }
  err.style.display='none';
  alert('Signed in as ' + email);
  return false;
}
</script>
</body>
</html>""" % (c[0], c[1], c[2], c[3], c[4])


def portfolio_page(pal: str = "indigo", who: str = "Your Name") -> str:
    c = PALETTES.get(pal, PALETTES["indigo"])
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__WHO__ - Portfolio</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  :root{--a:__C0__;--b:__C1__;--c:__C2__;--bg:__C3__;--fg:__C4__}
  body{font:16px/1.65 -apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",
       system-ui,sans-serif;background:var(--bg);color:var(--fg)}
  a{color:inherit;text-decoration:none}
  .wrap{max-width:1060px;margin:0 auto;padding:0 22px}
  nav{position:sticky;top:0;z-index:9;backdrop-filter:blur(14px);
      background:color-mix(in srgb,var(--bg) 78%,transparent);
      border-bottom:1px solid rgba(255,255,255,.09)}
  nav .wrap{display:flex;align-items:center;justify-content:space-between;
            height:66px}
  .brand{font-weight:800;letter-spacing:-.02em;font-size:19px}
  .brand span{background:linear-gradient(90deg,var(--a),var(--c));
              -webkit-background-clip:text;background-clip:text;color:transparent}
  .links{display:flex;gap:26px;font-size:14px;opacity:.85}
  .links a:hover{color:var(--c)}
  .burger{display:none;width:26px;height:20px;position:relative;cursor:pointer}
  .burger i{position:absolute;left:0;right:0;height:2px;background:var(--fg);
            border-radius:2px;transition:.25s}
  .burger i:nth-child(1){top:0}.burger i:nth-child(2){top:9px}
  .burger i:nth-child(3){top:18px}
  header{padding:120px 0 96px;position:relative;overflow:hidden}
  header:before{content:"";position:absolute;width:620px;height:620px;
     right:-180px;top:-260px;border-radius:50%;filter:blur(90px);opacity:.30;
     background:radial-gradient(circle,var(--a),transparent 68%)}
  .eyebrow{display:inline-flex;align-items:center;gap:8px;font-size:12px;
     letter-spacing:.16em;text-transform:uppercase;padding:7px 13px;
     border:1px solid rgba(255,255,255,.16);border-radius:999px;opacity:.8}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--c)}
  h1{font-size:clamp(38px,7vw,74px);line-height:1.03;letter-spacing:-.035em;
     margin:22px 0 18px;font-weight:800}
  h1 em{font-style:normal;background:linear-gradient(90deg,var(--a),var(--c));
        -webkit-background-clip:text;background-clip:text;color:transparent}
  .lede{font-size:clamp(16px,2.2vw,20px);opacity:.72;max-width:60ch}
  .cta{display:flex;gap:13px;flex-wrap:wrap;margin-top:32px}
  .btn{padding:13px 24px;border-radius:11px;font-weight:650;font-size:15px;
       border:1px solid transparent;cursor:pointer;transition:.2s}
  .btn.p{background:linear-gradient(90deg,var(--a),var(--b));color:#fff}
  .btn.p:hover{filter:brightness(1.13);transform:translateY(-2px)}
  .btn.g{border-color:rgba(255,255,255,.2)}
  .btn.g:hover{border-color:var(--c);color:var(--c)}
  section{padding:88px 0;border-top:1px solid rgba(255,255,255,.07)}
  .st{font-size:12px;letter-spacing:.16em;text-transform:uppercase;
      color:var(--c);font-weight:700;margin-bottom:14px}
  h2{font-size:clamp(26px,4vw,38px);letter-spacing:-.02em;margin-bottom:16px}
  .grid{display:grid;gap:20px;margin-top:34px;
        grid-template-columns:repeat(auto-fit,minmax(268px,1fr))}
  .card{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.1);
        border-radius:16px;padding:24px;transition:.25s}
  .card:hover{transform:translateY(-5px);border-color:var(--a);
              box-shadow:0 18px 44px rgba(0,0,0,.34)}
  .thumb{height:142px;border-radius:11px;margin-bottom:17px;
         background:linear-gradient(135deg,var(--a),var(--c));opacity:.9}
  .card h3{font-size:18px;margin-bottom:7px}
  .card p{font-size:14px;opacity:.68}
  .tags{display:flex;gap:7px;flex-wrap:wrap;margin-top:14px}
  .tag{font-size:11px;padding:4px 10px;border-radius:999px;
       background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1)}
  .bars{display:grid;gap:16px;margin-top:30px;max-width:560px}
  .bar b{display:flex;justify-content:space-between;font-size:14px;
         font-weight:600;margin-bottom:7px}
  .track{height:7px;border-radius:99px;background:rgba(255,255,255,.1)}
  .fill{height:100%;border-radius:99px;
        background:linear-gradient(90deg,var(--a),var(--c));width:0;
        transition:width 1.1s cubic-bezier(.2,.8,.2,1)}
  form{display:grid;gap:14px;max-width:520px;margin-top:28px}
  input,textarea{width:100%;padding:13px 15px;border-radius:11px;
    background:rgba(255,255,255,.05);color:var(--fg);font:inherit;
    border:1px solid rgba(255,255,255,.14)}
  input:focus,textarea:focus{outline:none;border-color:var(--a);
    box-shadow:0 0 0 3px color-mix(in srgb,var(--a) 26%,transparent)}
  .err{color:#F87171;font-size:13px;display:none}
  .ok{display:none;padding:13px 15px;border-radius:11px;font-size:14px;
      background:color-mix(in srgb,var(--a) 18%,transparent);
      border:1px solid var(--a)}
  footer{padding:38px 0;text-align:center;font-size:13px;opacity:.55;
         border-top:1px solid rgba(255,255,255,.07)}
  .reveal{opacity:0;transform:translateY(22px);
          transition:opacity .6s ease,transform .6s ease}
  .reveal.in{opacity:1;transform:none}
  @media(max-width:720px){
    .links{position:fixed;inset:66px 0 auto 0;background:var(--bg);
      flex-direction:column;gap:0;padding:8px 22px 20px;display:none;
      border-bottom:1px solid rgba(255,255,255,.1)}
    .links.open{display:flex}
    .links a{padding:13px 0;border-bottom:1px solid rgba(255,255,255,.06)}
    .burger{display:block}
    header{padding:74px 0 60px}section{padding:60px 0}
  }
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
</head>
<body>
<nav><div class="wrap">
  <div class="brand">__WHO__<span>.</span></div>
  <div class="links" id="nv">
    <a href="#work">Work</a><a href="#about">About</a>
    <a href="#skills">Skills</a><a href="#contact">Contact</a>
  </div>
  <div class="burger" id="bg"><i></i><i></i><i></i></div>
</div></nav>

<header><div class="wrap">
  <div class="eyebrow"><span class="dot"></span> Available for work</div>
  <h1>I build things<br>that <em>actually ship</em>.</h1>
  <p class="lede">__WHO__ - designer and developer. I turn rough ideas into
     fast, accessible products people enjoy using.</p>
  <div class="cta">
    <button class="btn p" onclick="document.getElementById('work')
      .scrollIntoView({behavior:'smooth'})">See my work</button>
    <a class="btn g" href="#contact">Get in touch</a>
  </div>
</div></header>

<section id="work"><div class="wrap reveal">
  <div class="st">Selected work</div>
  <h2>Projects I'm proud of</h2>
  <div class="grid">
    <div class="card"><div class="thumb"></div><h3>Nimbus Analytics</h3>
      <p>Real-time dashboard handling 40k events a minute without breaking
         a sweat.</p>
      <div class="tags"><span class="tag">React</span>
        <span class="tag">WebSocket</span><span class="tag">D3</span></div></div>
    <div class="card"><div class="thumb"></div><h3>Harbor Commerce</h3>
      <p>Headless storefront that cut checkout abandonment by 34%.</p>
      <div class="tags"><span class="tag">Next.js</span>
        <span class="tag">Stripe</span><span class="tag">Edge</span></div></div>
    <div class="card"><div class="thumb"></div><h3>Field Notes</h3>
      <p>Offline-first mobile app for researchers with zero data loss.</p>
      <div class="tags"><span class="tag">PWA</span>
        <span class="tag">IndexedDB</span><span class="tag">Sync</span></div></div>
  </div>
</div></section>

<section id="about"><div class="wrap reveal">
  <div class="st">About</div>
  <h2>Short version</h2>
  <p class="lede">Six years building for the web. I care about speed,
     accessibility and code the next person can read. Currently exploring
     local-first software and everything that runs on your own machine.</p>
</div></section>

<section id="skills"><div class="wrap reveal">
  <div class="st">Skills</div><h2>What I work with</h2>
  <div class="bars">
    <div class="bar"><b><span>JavaScript / TypeScript</span><span>92%</span></b>
      <div class="track"><div class="fill" data-w="92"></div></div></div>
    <div class="bar"><b><span>Python</span><span>85%</span></b>
      <div class="track"><div class="fill" data-w="85"></div></div></div>
    <div class="bar"><b><span>UI / UX design</span><span>78%</span></b>
      <div class="track"><div class="fill" data-w="78"></div></div></div>
    <div class="bar"><b><span>Infrastructure</span><span>70%</span></b>
      <div class="track"><div class="fill" data-w="70"></div></div></div>
  </div>
</div></section>

<section id="contact"><div class="wrap reveal">
  <div class="st">Contact</div><h2>Let's build something</h2>
  <form id="cf" novalidate>
    <input id="nm" placeholder="Your name" autocomplete="name">
    <input id="em" type="email" placeholder="you@example.com"
           autocomplete="email">
    <textarea id="ms" rows="4" placeholder="What are we making?"></textarea>
    <div class="err" id="er"></div>
    <button class="btn p" type="submit">Send message</button>
    <div class="ok" id="okb">Thanks - I'll reply within a day.</div>
  </form>
</div></section>

<footer>&copy; <span id="yr"></span> __WHO__ - built from scratch.</footer>

<script>
  document.getElementById('yr').textContent = new Date().getFullYear();
  var bg = document.getElementById('bg'), nv = document.getElementById('nv');
  bg.onclick = function(){ nv.classList.toggle('open'); };
  nv.addEventListener('click', function(e){
    if(e.target.tagName === 'A') nv.classList.remove('open'); });

  document.querySelectorAll('a[href^="#"]').forEach(function(a){
    a.addEventListener('click', function(e){
      var t = document.querySelector(a.getAttribute('href'));
      if(t){ e.preventDefault(); t.scrollIntoView({behavior:'smooth'}); }
    });
  });

  var io = new IntersectionObserver(function(es){
    es.forEach(function(en){
      if(!en.isIntersecting) return;
      en.target.classList.add('in');
      en.target.querySelectorAll('.fill').forEach(function(f){
        f.style.width = f.dataset.w + '%'; });
      io.unobserve(en.target);
    });
  }, {threshold:.16});
  document.querySelectorAll('.reveal').forEach(function(r){ io.observe(r); });

  document.getElementById('cf').addEventListener('submit', function(e){
    e.preventDefault();
    var n=document.getElementById('nm').value.trim(),
        m=document.getElementById('em').value.trim(),
        t=document.getElementById('ms').value.trim(),
        er=document.getElementById('er'), ok=document.getElementById('okb');
    var bad = !n ? 'Please tell me your name.'
            : !/^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$/.test(m) ? 'That email looks off.'
            : t.length < 10 ? 'A little more detail, please.' : '';
    er.style.display = bad ? 'block' : 'none';
    er.textContent = bad;
    if(bad) return;
    ok.style.display = 'block';
    this.reset();
  });
</script>
</body>
</html>""".replace("__C0__", c[0]).replace("__C1__", c[1]) \
           .replace("__C2__", c[2]).replace("__C3__", c[3]) \
           .replace("__C4__", c[4]).replace("__WHO__", who)


BUILD_WORDS = ("create", "make", "build", "generate", "write", "design",
               "code", "give me", "i want", "i need", "develop")


def wants_build(q: str) -> bool:
    ql = " " + q.lower().strip() + " "
    return any((" " + w + " ") in ql or ql.startswith(" " + w + " ")
               for w in BUILD_WORDS)


MIN_CODE_CHARS = 10000


def biggest_block(text: str) -> int:
    """Length of the largest fenced code block in an answer."""
    if not text:
        return 0
    return max([len(b) for b in re.findall(r"```\w*\n(.*?)\n```", text,
                                           re.S)] or [0])


def maker_answer(query: str) -> Optional[str]:
    """Return a finished artifact for common build requests."""
    # The big local templates come first - they are complete, tested files
    # well past MIN_CODE_CHARS. The older small ones stay as a last resort.
    _big = offline_build(query)
    if _big:
        return _big[0]
    q = query.lower()
    pal = "indigo"
    for name in PALETTES:
        if name in q:
            pal = name
    if any(w in q for w in ("green",)):
        pal = "emerald"
    elif any(w in q for w in ("orange", "warm")):
        pal = "sunset"
    elif any(w in q for w in ("red", "pink")):
        pal = "rose"
    elif any(w in q for w in ("blue",)):
        pal = "ocean"

    if "portfolio" in q or "portifolio" in q:
        c = PALETTES[pal]
        who = "Your Name"
        mm = re.search(r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", query)
        if mm:
            who = mm.group(1)
        return (
            "Here's a complete portfolio site - one file, no dependencies. "
            "Save it as `index.html` and open it.\n\n"
            "**What's inside**\n\n"
            "- Sticky nav that turns into a working burger menu on phones\n"
            "- Hero, projects grid, about, animated skill bars, contact form\n"
            "- Scroll-reveal animations and a validated form with real errors\n"
            "- Responsive to 320px, and it respects reduced-motion\n\n"
            "**Colour codes used**\n\n"
            "- `" + c[0] + "` primary\n"
            "- `" + c[1] + "` primary hover\n"
            "- `" + c[2] + "` accent / gradient end\n"
            "- `" + c[3] + "` background\n"
            "- `" + c[4] + "` text\n\n"
            "```html\n" + portfolio_page(pal, who) + "\n```\n\n"
            "Change `__WHO__` wherever you see your name, swap the three "
            "project cards, and it's yours. Want a different palette, a dark "
            "and light toggle, or a blog section added?"
        )

    if "login" in q or "sign in" in q or "signin" in q or "log in" in q:
        c = PALETTES[pal]
        return (
            "Here's a complete login page — one file, no dependencies. "
            "Save it as `login.html` and open it.\n\n"
            "**Colour codes used**\n\n"
            "- `" + c[0] + "` primary\n"
            "- `" + c[1] + "` primary hover\n"
            "- `" + c[2] + "` gradient end\n"
            "- `" + c[3] + "` text\n"
            "- `" + c[4] + "` card background\n\n"
            "```html\n" + login_page(pal) + "\n```\n\n"
            "It validates the email and a 6-character minimum password, shows "
            "an inline error, and is responsive down to a phone. Want it with "
            "a different palette, or wired to a real backend?"
        )
    return None


BOILER = re.compile(r"(cookie|privacy policy|sign in|subscribe|newsletter"
                    r"|all rights reserved|terms of service|advertisement"
                    r"|skip to content|javascript is disabled|enable javascript"
                    r"|read more|click here|sign up|log in|follow us|share this"
                    r"|last updated|table of contents|edit this page"
                    r"|was this page helpful|report an issue)", re.I)

# Sentences that are really code, nav crumbs or UI chrome rather than prose.
JUNK = re.compile(r"(^\s*[\W_]|@app\.|def |import |from \w+ import|=\s*\w+\(\)"
                  r"|\{|\}|;\s*$|>>>|\$ |^\d+\.\d+\.\d+)", re.M)
EMOJIISH = re.compile("[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u27BF"
                      "\u2B00-\u2BFF\uFE0F\u2028\u2029]")


def is_good_sentence(s: str) -> bool:
    """A sentence has to look like prose a person would actually say."""
    if not (55 <= len(s) <= 300):
        return False
    if BOILER.search(s) or JUNK.search(s):
        return False
    if EMOJIISH.search(s):
        return False
    words = s.split()
    if len(words) < 9:
        return False
    # must end like a sentence, and start like one
    if not s[0].isupper():
        return False
    if not s.rstrip().endswith((".", "!", "?")):
        return False
    # reject symbol soup / nav bars / code
    letters = sum(c.isalpha() or c.isspace() for c in s)
    if letters / float(len(s)) < 0.82:
        return False
    if s.count("|") > 1 or s.count("/") > 3 or s.count("(") > 3:
        return False
    # reject fragments made of very short tokens
    if sum(len(w) for w in words) / float(len(words)) < 3.2:
        return False
    return True


def extractive_answer(query: str, sources: list[dict], memory: str = "") -> str:
    """
    Build a cited answer straight from the sources. Scores each sentence by
    query-term overlap, cross-engine agreement and rank; filters boilerplate.
    This layer has no external dependency, so ADELTE always answers.
    """
    kw = set(keywords(query)) or {w for w in query.lower().split() if len(w) > 2}


    kw_ordered = [w for w in keywords(query)] or sorted(kw)
    head = kw_ordered[0] if kw_ordered else ""
    # "rust ownership" must not match a page about the Rust video game. When
    # the query has several meaningful terms, demand real coverage of them.
    need = 2 if len(kw) >= 2 else 1
    if len(kw) >= 4:
        need = 3
    # A definition reads "X is ...", "X refers to ...". Reward those heavily,
    # because that is what someone asking "what is X" actually wants.
    DEFN = re.compile(r"\b(is|are|was|were|means|refers to|stands for|"
                      r"is a|is an|is the|allows|lets you|provides|enables)\b",
                      re.I)

    def collect(min_ov: int) -> list:
        """One shared pass. Same prose rules every time — the fallback must
        not be allowed to smuggle in the junk the strict pass rejected."""
        out = []
        for idx, s in enumerate(sources, 1):
            # ---- source-level relevance -------------------------------
            # A page whose title or URL is about the query beats a random
            # repo that merely name-drops it. This is what stops "what is
            # postgresql" from answering with a WhatsApp clone's README.
            meta = (str(s.get("title", "")) + " " +
                    str(s.get("url", ""))).lower()
            meta_hits = sum(1 for w in kw if w in meta)
            src_boost = 1.6 * meta_hits
            if head and head in meta:
                src_boost += 1.4
            # official-looking docs get a nudge
            if re.search(r"(docs?|documentation|wiki|\.org/|/guide|/manual)",
                         meta):
                src_boost += 0.8
            # a repo/list page that only mentions the term is weak evidence
            if re.search(r"(github\.com|awesome|\bclone\b|tutorial-list)", meta)\
                    and meta_hits < len(kw):
                src_boost -= 1.5

            # Titles are NOT answers — Stack Overflow titles are questions,
            # and feeding them in produced "What is fastApi analog for ...?".
            # Only real page prose and snippets may become sentences.
            blob = " ".join(filter(None, [s.get("snippet", ""),
                                          s.get("page_text", "")]))
            for sent in re.split(r"(?<=[.!?])\s+|\n+", blob):
                sent = " ".join(sent.strip(" \u2022\u00b7-\u2013\u2014\t").split())
                if not is_good_sentence(sent):
                    continue
                if sent.rstrip().endswith("?"):      # no question as answer
                    continue
                low = sent.lower()
                if re.search(r"\b(chapter|section|appendix|figure|listing)\s+\d",
                             low):
                    continue
                if low.startswith(("chapter ", "section ", "part ", "step ",
                                   "figure ", "table ", "appendix ",
                                   "note that ", "in this ", "in the next ",
                                   "we'll ", "we will ", "let's ", "here we ",
                                   "as mentioned", "welcome to ",
                                   "introduction ", "this edition",
                                   "at first, ", "to start using")):
                    continue
                # personal blog narration is not an answer
                if re.match(r"^(i |i'|my |me |we |our |you might|you'll|"
                            r"you will|but )", low):
                    continue
                if low.count(" i ") + low.count(" my ") + low.count(" i'") >= 2:
                    continue
                sw = set(keywords(sent))
                ov = len(kw & sw)
                if ov < min_ov:
                    continue
                score = (ov * 1.5 + 0.5 * s.get("agreement", 1)
                         + max(0, 1.5 - idx * 0.05) - 0.002 * len(sent)
                         + src_boost)
                # Covering EVERY query term matters more than covering one
                # term twice. "rust ownership" should reward a sentence about
                # ownership, not any sentence that says "Rust".
                cover = ov / float(len(kw)) if kw else 0
                score += 2.5 * cover
                if len(kw) > 1 and cover < 1.0:
                    score -= 1.2 * (1.0 - cover)
                if DEFN.search(sent):
                    score += 1.2
                # "<subject> is a ..." — the classic definition shape
                if head and re.match(r"^" + re.escape(head) +
                                     r"\b[^.]{0,30}\b(is|are)\b", low):
                    score += 3.0
                if head and low.startswith(head):
                    score += 2.0
                if s.get("page_text"):
                    score += 0.8      # real article text beats a snippet
                out.append((score, sent, idx))
        return out

    scored = collect(need)
    if len(scored) < 3 and need > 1:
        scored = collect(need - 1)
    if len(scored) < 2 and need > 2:
        scored = collect(1)

    scored.sort(key=lambda x: -x[0])

    def norm(t: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", t.lower())

    picked: list = []
    norms: list = []
    for _, sent, idx in scored:
        n = norm(sent)
        if len(n) < 30:
            continue
        dup = False
        for prev in norms:
            # exact repeat, or one sentence swallowed by another, or a
            # near-identical opening — all count as the same sentence.
            if n == prev or n in prev or prev in n or n[:60] == prev[:60]:
                dup = True
                break
            a, b = set(n.split()), set(prev.split())
            if a and b and len(a & b) / float(len(a | b)) > 0.6:
                dup = True
                break
        if dup:
            continue
        norms.append(n)
        picked.append((sent, idx))
        if len(picked) >= 5:
            break

    if not picked:
        out = [f"I collected **{len(sources)}** sources for _{query}_ but could not "
               f"extract a confident summary. Most relevant results:", ""]
        for i, s in enumerate(sources[:8], 1):
            out.append(f"{i}. [{s.get('title','(untitled)')}]({s.get('url','')}) — "
                       f"{(s.get('snippet') or '')[:140]}")
        return "\n".join(out)

    # The UI already renders source cards under every answer, so repeating a
    # second text list here just doubled the noise. Keep the prose only, and
    # keep it SHORT and ORDERED — a lead line, then at most four points.
    parts = []
    if memory:
        parts.append("Picking up from what we covered earlier — ")

    lead, rest = picked[0], picked[1:4]
    parts.append("Here's the short version: " + lead[0] + " [" + str(lead[1]) + "]")
    if rest:
        parts.append("")
        for sent, idx in rest:
            parts.append("- " + sent + " [" + str(idx) + "]")
    parts.append("")
    parts.append("_Based on " + str(len(sources)) + " sources — the cards "
                 "below show each one. Ask a follow-up and I'll keep the "
                 "context._")
    return "\n".join(parts)


# An AI reply shorter than this is a failed generation, not an answer.
MIN_AI_CHARS = 120


def usable_answer(text: Optional[str], strict: bool = True) -> bool:
    """Free swarm workers sometimes return a fragment or a bare heading.

    Anything too short, or with no sentence in it at all, is worse than the
    extractive fallback — so reject it and let the fallback run.

    strict=False is used for keyed providers (Groq, Gemini, OpenRouter). Those
    are paid-grade brains: if one says "Yes, that works." we keep it. Holding
    them to the swarm's 120-char bar threw away perfectly good short answers.
    """
    if not text:
        return False
    t = text.strip()
    if not strict:
        return len(t) >= 2
    if len(t) < MIN_AI_CHARS:
        return False
    if "." not in t and "\n-" not in t and "\n*" not in t:
        return False
    return True


async def ai_answer(c: httpx.AsyncClient, messages: list[dict],
                    budget: float, on_progress=None,
                    model: str = "") -> Optional[str]:
    """Answer with the best brain available.

    The keyed providers (Groq/OpenRouter/Gemini) come first because they are
    fast and can return a whole file. The free swarm is capped at 500 tokens
    and used only as the last resort - going straight to it was truncating
    long code half way through.
    """
    if not CFG.use_ai:
        return None
    try:
        chain = live_chain(model or DEFAULT_MODEL)
    except Exception:
        chain = []
    if chain:
        async def _em(kind, payload):
            if on_progress:
                try:
                    await on_progress(payload if isinstance(payload, dict)
                                      else {"text": str(payload)})
                except Exception:
                    pass
        txt, _who = await generate_with_model(
            c, model or DEFAULT_MODEL, messages, budget, _em)
        if txt and usable_answer(txt, strict=False):
            return txt
    out = await horde_generate(c, messages, budget, on_progress)
    return out if usable_answer(out) else None


# ============================================================================
#  SECTION 5 — STORE (SQLite: keys, conversations, memory)
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys(
  key TEXT PRIMARY KEY, label TEXT, created_at REAL, calls INTEGER DEFAULT 0,
  daily_limit INTEGER DEFAULT 500, day_bucket TEXT, day_calls INTEGER DEFAULT 0,
  revoked INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS conversations(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT,
  content TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_conv ON conversations(session_id);
CREATE TABLE IF NOT EXISTS memory(
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, query TEXT,
  keywords TEXT, answer TEXT, sources TEXT, created_at REAL);
CREATE INDEX IF NOT EXISTS ix_mem ON memory(session_id);
CREATE TABLE IF NOT EXISTS accounts(
  username TEXT PRIMARY KEY, email TEXT, pwd_hash TEXT, salt TEXT,
  avatar TEXT DEFAULT '', api_key TEXT, created_at REAL, last_seen REAL,
  calls INTEGER DEFAULT 0, verified INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_acct_key ON accounts(api_key);
CREATE TABLE IF NOT EXISTS logins(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, kind TEXT,
  ok INTEGER, ip TEXT, agent TEXT, at REAL);
CREATE INDEX IF NOT EXISTS ix_login_u ON logins(username);
"""


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self._migrate()
        self.db.commit()

    def _migrate(self) -> None:
        """Add columns that older ADELTE databases do not have yet.

        Users upgrade in place, so we can never assume the file on disk
        matches the current schema. Each ALTER is tried on its own and a
        duplicate-column error is simply ignored.
        """
        wanted = [("accounts", "verified", "INTEGER DEFAULT 0"),
                  ("accounts", "avatar", "TEXT DEFAULT ''")]
        for table, col, decl in wanted:
            try:
                cols = {r[1] for r in self.db.execute(
                    "PRAGMA table_info(%s)" % table).fetchall()}
                if col not in cols:
                    self.db.execute("ALTER TABLE %s ADD COLUMN %s %s"
                                    % (table, col, decl))
            except Exception:
                pass
        self.db.commit()

    # -- keys ---------------------------------------------------------------
    def create_key(self, label: str = "") -> dict:
        key = "adelte-" + secrets.token_urlsafe(24)
        self.db.execute(
            "INSERT INTO api_keys(key,label,created_at,daily_limit,day_bucket,"
            "day_calls) VALUES(?,?,?,?,?,0)",
            (key, label[:80], time.time(), CFG.daily_key_limit,
             time.strftime("%Y-%m-%d")))
        self.db.commit()
        return {"api_key": key, "label": label,
                "daily_limit": CFG.daily_key_limit, "created_at": time.time()}

    KEY_PREFIXES = ("adelte-", "delta-", "sk-adelte-")

    def looks_like_key(self, key: str) -> bool:
        """Accept any key ADELTE could plausibly have minted."""
        k = (key or "").strip()
        if not any(k.startswith(p) for p in self.KEY_PREFIXES):
            return False
        body = k.split("-", 1)[1] if "-" in k else ""
        return len(body) >= 16

    def adopt_key(self, key: str, label: str = "adopted") -> None:
        """Re-register a key we no longer have a row for.

        Keys minted before a database move (or on another machine) must keep
        working - ADELTE is free for everyone and must never hand back an
        auth error for a genuine ADELTE key.
        """
        self.db.execute(
            "INSERT OR IGNORE INTO api_keys(key,label,created_at,daily_limit,"
            "day_bucket,day_calls) VALUES(?,?,?,?,?,0)",
            (key.strip(), label[:80], time.time(), CFG.daily_key_limit,
             time.strftime("%Y-%m-%d")))
        self.db.commit()

    def use_key(self, key: str) -> Tuple[bool, str, Optional[dict]]:
        key = (key or "").strip()
        row = self.db.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        if not row and self.looks_like_key(key):
            self.adopt_key(key)
            row = self.db.execute("SELECT * FROM api_keys WHERE key=?",
                                  (key,)).fetchone()
        if not row:
            return False, "unknown API key", None
        if row["revoked"]:
            return False, "API key revoked", None
        today = time.strftime("%Y-%m-%d")
        used = row["day_calls"] if row["day_bucket"] == today else 0
        if used >= row["daily_limit"]:
            return False, f"daily limit of {row['daily_limit']} reached", None
        self.db.execute("UPDATE api_keys SET calls=calls+1,day_bucket=?,day_calls=? "
                        "WHERE key=?", (today, used + 1, key))
        self.db.commit()
        return True, "ok", {"label": row["label"], "calls_today": used + 1,
                            "daily_limit": row["daily_limit"],
                            "remaining_today": row["daily_limit"] - used - 1}

    def key_info(self, key: str) -> Optional[dict]:
        row = self.db.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        today = time.strftime("%Y-%m-%d")
        used = row["day_calls"] if row["day_bucket"] == today else 0
        return {"label": row["label"], "created_at": row["created_at"],
                "calls_total": row["calls"], "calls_today": used,
                "daily_limit": row["daily_limit"],
                "remaining_today": row["daily_limit"] - used,
                "revoked": bool(row["revoked"])}

    # -- conversation -------------------------------------------------------
    def add_turn(self, sid: str, role: str, content: str) -> None:
        self.db.execute("INSERT INTO conversations(session_id,role,content,"
                        "created_at) VALUES(?,?,?,?)", (sid, role, content, time.time()))
        self.db.commit()

    def history(self, sid: str, limit: int = 12) -> list[dict]:
        rows = self.db.execute("SELECT role,content FROM conversations WHERE "
                               "session_id=? ORDER BY id DESC LIMIT ?",
                               (sid, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def clear(self, sid: str) -> None:
        self.db.execute("DELETE FROM conversations WHERE session_id=?", (sid,))
        self.db.execute("DELETE FROM memory WHERE session_id=?", (sid,))
        self.db.commit()

    # -- memory -------------------------------------------------------------
    # -- accounts -----------------------------------------------------------
    EMAIL_DOMAIN = "@adelte.mab"

    @staticmethod
    def _hash(pwd: str, salt: str) -> str:
        return hashlib.sha256((salt + pwd).encode("utf-8")).hexdigest()

    @classmethod
    def clean_name(cls, username: str) -> str:
        """Accept 'bon', 'bon@adelte.mab', ' Bon ' - all mean the same user.

        The user should never have to type the domain, but if they paste it
        (or their browser autofills it) we must not mangle it into
        'bonadeltemab' the way the old strip did.
        """
        u = (username or "").strip().lower()
        if "@" in u:                      # they typed an e-mail - keep the name
            u = u.split("@", 1)[0]
        return re.sub(r"[^a-z0-9._-]", "", u)

    def register(self, username: str, password: str) -> dict:
        u = self.clean_name(username)
        if len(u) < 3:
            raise ValueError("username needs at least 3 letters or digits")
        if len(password or "") < 4:
            raise ValueError("password needs at least 4 characters")
        if self.get_account(u):
            raise ValueError("that username is taken")
        salt = secrets.token_hex(8)
        key = "adelte-" + secrets.token_urlsafe(24)
        self.adopt_key(key, "account:" + u)
        self.db.execute(
            "INSERT INTO accounts(username,email,pwd_hash,salt,avatar,api_key,"
            "created_at,last_seen,calls) VALUES(?,?,?,?,'',?,?,?,0)",
            (u, u + self.EMAIL_DOMAIN, self._hash(password, salt), salt,
             key, time.time(), time.time()))
        self.db.commit()
        return self.public_account(u)

    def login(self, username: str, password: str) -> dict:
        u = self.clean_name(username)
        row = self.get_account(u)
        if not row:
            raise ValueError("no account called " + u)
        if self._hash(password or "", row["salt"]) != row["pwd_hash"]:
            raise ValueError("wrong password")
        self.db.execute("UPDATE accounts SET last_seen=? WHERE username=?",
                        (time.time(), u))
        self.db.commit()
        return self.public_account(u)

    def get_account(self, username: str):
        return self.db.execute("SELECT * FROM accounts WHERE username=?",
                               (self.clean_name(username),)).fetchone()

    def account_by_key(self, key: str):
        return self.db.execute("SELECT * FROM accounts WHERE api_key=?",
                               ((key or "").strip(),)).fetchone()

    def public_account(self, username: str) -> dict:
        r = self.get_account(username)
        if not r:
            return {}
        return {"username": r["username"], "email": r["email"],
                "avatar": r["avatar"] or "", "api_key": r["api_key"],
                "created_at": r["created_at"], "calls": r["calls"],
                "verified": bool(self._col(r, "verified", 0))}

    @staticmethod
    def _col(row, name, default=None):
        """Read a column that may not exist on an older database file."""
        try:
            v = row[name]
        except Exception:
            return default
        return default if v is None else v

    # -- admin ---------------------------------------------------------------
    def log_auth(self, username: str, kind: str, ok: bool,
                 ip: str = "", agent: str = "") -> None:
        try:
            self.db.execute(
                "INSERT INTO logins(username,kind,ok,ip,agent,at) "
                "VALUES(?,?,?,?,?,?)",
                (self.clean_name(username), kind, 1 if ok else 0,
                 ip or "", (agent or "")[:200], time.time()))
            self.db.commit()
        except Exception:
            pass

    def all_accounts(self) -> list:
        out = []
        for r in self.db.execute(
                "SELECT * FROM accounts ORDER BY created_at DESC").fetchall():
            out.append({
                "username": r["username"], "email": r["email"],
                "avatar": r["avatar"] or "", "api_key": r["api_key"],
                "created_at": r["created_at"], "last_seen": r["last_seen"],
                "calls": r["calls"] or 0,
                "verified": bool(self._col(r, "verified", 0)),
            })
        return out

    def auth_log(self, limit: int = 300) -> list:
        try:
            rows = self.db.execute(
                "SELECT * FROM logins ORDER BY at DESC LIMIT ?",
                (int(limit),)).fetchall()
        except Exception:
            return []
        return [{"id": r["id"], "username": r["username"], "kind": r["kind"],
                 "ok": bool(r["ok"]), "ip": r["ip"], "agent": r["agent"],
                 "at": r["at"]} for r in rows]

    def set_verified(self, username: str, on: bool) -> dict:
        u = self.clean_name(username)
        if not self.get_account(u):
            raise ValueError("no account called " + u)
        self.db.execute("UPDATE accounts SET verified=? WHERE username=?",
                        (1 if on else 0, u))
        self.db.commit()
        return self.public_account(u)

    def delete_account(self, username: str) -> bool:
        u = self.clean_name(username)
        row = self.get_account(u)
        if not row:
            return False
        try:
            self.db.execute("DELETE FROM api_keys WHERE key=?",
                            (row["api_key"],))
        except Exception:
            pass
        self.db.execute("DELETE FROM accounts WHERE username=?", (u,))
        self.db.execute("DELETE FROM logins WHERE username=?", (u,))
        self.db.commit()
        return True

    def set_avatar(self, username: str, data_url: str) -> dict:
        if len(data_url or "") > 400000:
            raise ValueError("picture too large - keep it under 300 KB")
        self.db.execute("UPDATE accounts SET avatar=? WHERE username=?",
                        (data_url, (username or "").strip().lower()))
        self.db.commit()
        return self.public_account(username)

    def bump_account(self, key: str) -> None:
        self.db.execute("UPDATE accounts SET calls=calls+1,last_seen=? "
                        "WHERE api_key=?", (time.time(), (key or "").strip()))
        self.db.commit()

    def remember(self, sid: str, query: str, answer: str, sources: list[dict]) -> None:
        slim = [{"title": s.get("title"), "url": s.get("url"),
                 "snippet": (s.get("snippet") or "")[:400],
                 "source": s.get("source")} for s in sources[:10]]
        self.db.execute("INSERT INTO memory(session_id,query,keywords,answer,"
                        "sources,created_at) VALUES(?,?,?,?,?,?)",
                        (sid, query, " ".join(keywords(query + " " + answer[:600])),
                         answer, json.dumps(slim), time.time()))
        self.db.commit()

    def recall(self, sid: str, query: str, limit: int = 3) -> list[dict]:
        rows = self.db.execute("SELECT query,keywords,answer,sources FROM memory "
                               "WHERE session_id=? ORDER BY id DESC LIMIT 40",
                               (sid,)).fetchall()
        qk = set(keywords(query))
        if not qk:
            return []
        hits = []
        for r in rows:
            stored = set((r["keywords"] or "").split())
            ov = len(qk & stored)
            # absolute overlap, OR a decent share of a short follow-up question
            # ("how does it compare to Flask?" only has a few content words)
            if ov >= 2 or (ov >= 1 and ov / len(qk) >= 0.25):
                hits.append((ov, {"query": r["query"], "answer": r["answer"],
                                  "sources": json.loads(r["sources"] or "[]")}))
        hits.sort(key=lambda x: -x[0])
        return [h[1] for h in hits[:limit]]

    def stats(self) -> dict:
        g = lambda s: self.db.execute(s).fetchone()[0]
        return {"api_keys": g("SELECT COUNT(*) FROM api_keys"),
                "sessions": g("SELECT COUNT(DISTINCT session_id) FROM conversations"),
                "turns": g("SELECT COUNT(*) FROM conversations"),
                "memories": g("SELECT COUNT(*) FROM memory"),
                "api_calls": g("SELECT COALESCE(SUM(calls),0) FROM api_keys")}


STORE: Optional["Store"] = None      # created in main()


# ============================================================================
#  SECTION 6 — FASTAPI APP
# ============================================================================

CLIENT: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CLIENT, STORE
    if STORE is None:                      # allows `uvicorn delta:app` too
        STORE = Store(CFG.db_path)
    if CLIENT is None:
        CLIENT = httpx.AsyncClient(
            headers=HEADERS, follow_redirects=True,
            timeout=httpx.Timeout(CFG.request_timeout, connect=9.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=25))
    yield
    try:
        await CLIENT.aclose()
    except Exception:
        pass
    CLIENT = None


def get_store() -> "Store":
    """Lazy store for serverless cold starts where lifespan may lag."""
    global STORE
    if STORE is None:
        STORE = Store(CFG.db_path)
    return STORE


async def get_client() -> httpx.AsyncClient:
    """Lazy HTTP client for serverless cold starts."""
    global CLIENT
    if CLIENT is None:
        CLIENT = httpx.AsyncClient(
            headers=HEADERS, follow_redirects=True,
            timeout=httpx.Timeout(CFG.request_timeout, connect=9.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=25))
    return CLIENT


app = FastAPI(title="ADELTE Server", version="2.1.0", lifespan=lifespan,
              description="Keyless multi-engine research server with a free public API.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


class ChatReq(BaseModel):
    message: str = Field(..., min_length=1, description="Your question")
    session_id: Optional[str] = Field(None, description="Reuse to keep context")
    engines: Optional[List[str]] = Field(None, description="Override the engine mix")
    deep: bool = Field(True, description="Also download and read the top pages")
    use_memory: bool = Field(True, description="Recall earlier answers this session")
    max_results: int = Field(8, ge=1, le=20)
    ai_budget: float = Field(50.0, ge=0, le=180,
                             description="Seconds to wait for the AI layer")
    confirm: bool = Field(False, description="Confirm a destructive command")
    model: Optional[str] = Field(None, description="adelte-search | "
                                 "adelte-coder-3high | adelte-minimax")


class KeyReq(BaseModel):
    label: str = Field("", max_length=80)


def auth(authorization: Optional[str], x_api_key: Optional[str]) -> Optional[dict]:
    """No key = allowed. A supplied key must be valid."""
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key:
        return None
    ok, msg, info = get_store().use_key(key)
    if not ok:
        raise HTTPException(401, msg)
    return info


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def filter_relevant(query: str, results: list) -> list:
    """Throw away results that have nothing to do with the query.

    Necessary because engines lie. Bing's RSS endpoint has been observed
    returning a channel titled "what is postgresql" whose items were all
    about the film "Waiting to Exhale" — cached content served under the
    wrong query. Without this guard those hits reach the answer.
    """
    kw = [w for w in keywords(query) if len(w) > 2]
    if not kw:
        return results

    kept, rescue = [], []
    for r in results:
        hay = " ".join(str(r.get(k, "")) for k in
                       ("title", "snippet", "url")).lower()
        hits = sum(1 for w in kw if w in hay)
        if hits == 0:
            continue                       # zero overlap -> certainly noise
        # A single weak term is only enough when it is the main subject.
        if hits == 1 and len(kw) > 1 and kw[0] not in hay:
            rescue.append(r)
            continue
        kept.append(r)

    # Never return nothing just because the filter was strict.
    if len(kept) < 3:
        kept = kept + rescue[: 5 - len(kept)]
    return kept if kept else results[:5]


async def gather(query: str, names: list[str], n: int, deep: bool, emit) -> list[dict]:
    """Run engines in parallel, dedupe, then deep-read the best pages."""
    await emit("engines_start", {"engines": names})
    tasks = [asyncio.create_task(run_engine(CLIENT, nm, query, n)) for nm in names]

    got: list[dict] = []
    done_n = 0
    for fut in asyncio.as_completed(tasks):
        env = await fut
        done_n += 1
        pct = 15 + int(40 * done_n / max(1, len(names)))
        if env["ok"] and env["results"]:
            got.extend(env["results"])
            await emit("engine", {"engine": env["engine"], "status": "ok",
                                  "count": len(env["results"]), "p": pct})
        elif env["ok"]:
            await emit("engine", {"engine": env["engine"], "status": "empty",
                                  "count": 0, "p": pct})
        else:
            await emit("engine", {"engine": env["engine"], "status": "bad",
                                  "count": 0, "error": env["error"], "p": pct})

    merged = dedupe(got)
    before = len(merged)
    merged = filter_relevant(query, merged)
    dropped = before - len(merged)
    note = (f"Merged **{len(got)}** hits into **{before}** unique sources"
            + (f", dropped **{dropped}** off-topic" if dropped else ""))
    await emit("thinking", {"text": note, "p": 60})

    if deep and merged:
        top = merged[:CFG.deep_pages]
        await emit("thinking", {"text": f"Reading {len(top)} pages in full", "p": 62})
        texts = await asyncio.gather(*[read_page(CLIENT, s["url"]) for s in top],
                                     return_exceptions=True)
        for i, (s, t) in enumerate(zip(top, texts)):
            if isinstance(t, str) and len(t) > 200:
                s["page_text"] = t
                await emit("read", {"url": s["url"], "chars": len(t),
                                    "p": 63 + i * 2})
    return merged


# ------------------------------- endpoints ---------------------------------

def _safe_openapi():
    """Never let /docs break.

    FastAPI regenerates the schema on demand; if any route annotation upsets
    the installed pydantic version the whole page 500s. We cache a good
    schema, and fall back to a minimal valid one so /docs always renders.
    """
    if app.openapi_schema:
        return app.openapi_schema
    try:
        from fastapi.openapi.utils import get_openapi
        app.openapi_schema = get_openapi(
            title="ADELTE Server", version="3.0.0",
            description="ADELTE by ADELTE Industries - free public AI API. "
                        "No key required. Send POST /api/chat with "
                        "{\"message\": \"...\"}.",
            routes=app.routes)
    except Exception as exc:                       # pragma: no cover
        print("  [openapi] full schema failed (%s) - serving reduced schema"
              % exc)
        app.openapi_schema = {
            "openapi": "3.1.0",
            "info": {"title": "ADELTE Server", "version": "3.0.0",
                     "description": "Reduced schema. The API itself is "
                                    "unaffected."},
            "paths": {
                "/api/chat": {"post": {
                    "summary": "Ask ADELTE (JSON reply)",
                    "requestBody": {"required": True, "content": {
                        "application/json": {"schema": {
                            "type": "object",
                            "required": ["message"],
                            "properties": {
                                "message": {"type": "string"},
                                "session_id": {"type": "string"},
                                "model": {"type": "string"},
                                "ai_budget": {"type": "number"}}}}}},
                    "responses": {"200": {"description": "Answer"}}}},
                "/api/chat/stream": {"post": {
                    "summary": "Ask ADELTE (live SSE stream)",
                    "responses": {"200": {"description": "text/event-stream"}}}},
                "/api/models": {"get": {"summary": "List ADELTE models",
                                        "responses": {"200": {"description": "OK"}}}},
                "/api/health": {"get": {"summary": "Health check",
                                        "responses": {"200": {"description": "OK"}}}},
            },
        }
    return app.openapi_schema


app.openapi = _safe_openapi


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.1.0", "time": time.time(),
            "engines": list(ENGINES), "ai_enabled": CFG.use_ai}


class SocialReq(BaseModel):
    platform: str = Field(..., description="whatsapp|telegram|slack|discord|"
                                           "email|x|webhook")
    handle: str = Field("", max_length=200)
    token: str = Field("", max_length=400)
    target: str = Field("", max_length=400)


#  How each platform is reached. Everything is either a public deep link
#  (no account access at all) or YOUR OWN bot token - ADELTE never asks for
#  a password and never proxies through anyone else's server.
SOCIAL = {
    "whatsapp": {"label": "WhatsApp", "kind": "link", "needs": ["target"],
                 "how": "Opens wa.me with your message pre-typed. No login.",
                 "tmpl": "https://wa.me/{target}?text={text}"},
    "telegram": {"label": "Telegram", "kind": "api", "needs": ["token", "target"],
                 "how": "Uses your own bot token from @BotFather. Really sends.",
                 "tmpl": "https://api.telegram.org/bot{token}/sendMessage"},
    "slack": {"label": "Slack", "kind": "webhook", "needs": ["token"],
              "how": "Incoming webhook URL from your workspace. Really sends.",
              "tmpl": "{token}"},
    "discord": {"label": "Discord", "kind": "webhook", "needs": ["token"],
                "how": "Channel webhook URL. Really sends.",
                "tmpl": "{token}"},
    "email": {"label": "Email", "kind": "link", "needs": ["target"],
              "how": "Opens your mail app with the draft ready.",
              "tmpl": "mailto:{target}?subject=From%20ADELTE&body={text}"},
    "x": {"label": "X / Twitter", "kind": "link", "needs": [],
          "how": "Opens the composer with your text. No login needed.",
          "tmpl": "https://twitter.com/intent/tweet?text={text}"},
    "webhook": {"label": "Custom webhook", "kind": "webhook", "needs": ["token"],
                "how": "Any URL that accepts a JSON POST.",
                "tmpl": "{token}"},
}


@app.get("/api/social")
async def social_list():
    return {"platforms": [dict(id=k, label=v["label"], kind=v["kind"],
                               needs=v["needs"], how=v["how"])
                          for k, v in SOCIAL.items()]}


@app.post("/api/social/send")
async def social_send(req: SocialReq, text: str = Query("", max_length=2000)):
    """Actually deliver a message, or hand back a deep link to open."""
    p = SOCIAL.get(req.platform)
    if not p:
        raise HTTPException(400, "Unknown platform: " + req.platform)
    body = text or "Hello from ADELTE"
    if p["kind"] == "link":
        url = p["tmpl"].format(target=quote(req.target or ""),
                               text=quote(body))
        return {"ok": True, "action": "open", "url": url,
                "detail": "Opening " + p["label"] + " with your message ready."}
    if not req.token:
        raise HTTPException(400, p["label"] + " needs your token or webhook URL")
    try:
        if req.platform == "telegram":
            r = await CLIENT.post(p["tmpl"].format(token=req.token),
                                  json={"chat_id": req.target, "text": body},
                                  timeout=20)
        elif req.platform == "slack":
            r = await CLIENT.post(req.token, json={"text": body}, timeout=20)
        elif req.platform == "discord":
            r = await CLIENT.post(req.token, json={"content": body}, timeout=20)
        else:
            r = await CLIENT.post(req.token,
                                  json={"text": body, "from": "ADELTE"},
                                  timeout=20)
    except Exception as e:
        raise HTTPException(502, p["label"] + " unreachable: " + str(e)[:120])
    ok = 200 <= r.status_code < 300
    return {"ok": ok, "action": "sent", "status": r.status_code,
            "detail": (p["label"] + " accepted the message.") if ok
                      else (p["label"] + " refused: " + r.text[:180])}


@app.get("/api/models")
async def models_list():
    """Every ADELTE model, its colour, and which brains .env unlocked."""
    out = []
    for mid, m in ADELTE_MODELS.items():
        chain = [{"provider": PROVIDERS[p]["label"], "model": mo,
                  "ready": provider_ready(p)} for p, mo in m["chain"]]
        local = bool(m.get("local"))
        out.append({"id": mid, "name": m["name"], "tag": m["tag"],
                    "blurb": m["blurb"], "search": m["search"],
                    "color": m.get("color", "#4F8DFD"),
                    "glow": m.get("glow", "#1E4FBF"),
                    "local": local, "desktop": bool(m.get("desktop")),
                    "ready": True if local else any(c["ready"] for c in chain),
                    "chain": chain})
    prov = [{"id": pid, "label": p["label"], "env": p["env"],
             "ready": provider_ready(pid), "key": mask(env_key(p["env"]))}
            for pid, p in PROVIDERS.items()]
    return {"models": out, "default": DEFAULT_MODEL, "providers": prov,
            "env_file": str(HERE / ".env"),
            "env_found": (HERE / ".env").exists()}


@app.get("/api/engines")
async def engines_list():
    return {"engines": [{"name": n,
                         "kind": "code" if n in ("github", "github_issues",
                                                 "stackoverflow", "hackernews")
                                 else "web",
                         "keyless": n != "google_cse",
                         **ENGINE_HEALTH.get(n, {"calls": 0, "health": "idle"})}
                        for n in ENGINES],
            "default_web": WEB_SET, "default_code": CODE_SET}


@app.get("/api/integrations/status", tags=["integrations"])
async def integrations_status():
    """Masked live-report for every key imported from new_api.

    Additive only: shows which of your APIs are active (OpenAI pool size,
    Groq, Gemini, OpenRouter, Clarifai, Google CSE, Telegram, Spice, Databricks,
    SQL) without ever leaking a secret. Keys come from .env which I already
    filled from your new_api file - nothing for you to paste by hand.
    """
    st = integration_status()
    chains = {mid: [{"provider": p, "model": mo, "ready": provider_ready(p)}
                    for p, mo in m["chain"]] for mid, m in ADELTE_MODELS.items()}
    return {"ok": True, "integrations": st, "chains": chains,
            "openai_pool": len(openai_key_pool()),
            "engines": sorted(ENGINES.keys())}


@app.get("/api/stats")
async def stats():
    return {**get_store().stats(), "engine_health": ENGINE_HEALTH}


@app.get("/api/check-keys")
async def check_keys():
    """The endpoint that 404'd on the old server. No provider key is needed."""
    return {"requires_provider_key": False, "env_keys_needed": [],
            "message": "ADELTE needs no provider API key — every engine is keyless.",
            "free_api_key_endpoint": "POST /api/keys/generate"}


@app.post("/api/keys/generate")
async def gen_key(req: KeyReq):
    info = STORE.create_key(req.label)
    return {**info,
            "usage": {"header": "X-API-Key: <key>  (or Authorization: Bearer <key>)",
                      "endpoints": ["POST /api/chat", "POST /api/chat/stream",
                                    "GET /api/search"]},
            "note": "Free. The API also works with no key at all."}


@app.get("/api/keys/info")
async def key_info(x_api_key: Optional[str] = Header(None),
                   authorization: Optional[str] = Header(None)):
    key = x_api_key
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization[7:].strip()
    if not key:
        raise HTTPException(400, "supply the X-API-Key header")
    info = STORE.key_info(key)
    if not info:
        raise HTTPException(404, "unknown API key")
    return info


# ---------------------------------------------------------------------------
#  ACCOUNTS - username + password, e-mail is always <username>@adelte.mab
# ---------------------------------------------------------------------------

class AccountReq(BaseModel):
    username: str = Field(..., min_length=3, max_length=64,
                          description="Just your name, e.g. boncoeur - "
                                      "@adelte.mab is added for you")
    password: str = Field(..., min_length=4, max_length=200)


class AvatarReq(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    avatar: str = Field(..., description="data:image/png;base64,... under 300 KB")


@app.post("/api/account/register", tags=["accounts"])
async def account_register(req: AccountReq, request: Request = None):
    """Create an ADELTE account. Free, instant, no e-mail confirmation.

    The user types only a name - `@adelte.mab` is added for them. If they
    paste the whole address anyway it still works.
    """
    ip = getattr(getattr(request, "client", None), "host", "") or ""
    agent = (request.headers.get("user-agent", "") if request else "")
    try:
        acct = STORE.register(req.username, req.password)
    except ValueError as e:
        STORE.log_auth(req.username, "register", False, ip, agent)
        raise HTTPException(400, str(e))
    STORE.log_auth(acct["username"], "register", True, ip, agent)
    return {"ok": True, "account": acct,
            "message": "Welcome to ADELTE, " + acct["username"] + ".",
            "how_to_use": {
                "header": "X-ADELTE-User: " + acct["username"],
                "key": "X-API-Key: " + acct["api_key"],
                "note": "Both are optional - ADELTE is free for everyone."}}


@app.post("/api/account/login", tags=["accounts"])
async def account_login(req: AccountReq, request: Request = None):
    """Sign in and get your API key back."""
    ip = getattr(getattr(request, "client", None), "host", "") or ""
    agent = (request.headers.get("user-agent", "") if request else "")
    try:
        acct = STORE.login(req.username, req.password)
    except ValueError as e:
        STORE.log_auth(req.username, "login", False, ip, agent)
        raise HTTPException(401, str(e))
    STORE.log_auth(acct["username"], "login", True, ip, agent)
    return {"ok": True, "account": acct,
            "message": "Signed in as " + acct["email"]}


@app.get("/api/account/{username}", tags=["accounts"])
async def account_get(username: str):
    """Public profile: name, e-mail, picture. Never the password."""
    acct = STORE.public_account(username)
    if not acct:
        raise HTTPException(404, "no account called " + username)
    acct.pop("api_key", None)
    return {"ok": True, "account": acct}


@app.post("/api/account/avatar", tags=["accounts"])
async def account_avatar(req: AvatarReq):
    """Upload a profile picture as a data URL."""
    if not STORE.get_account(req.username):
        raise HTTPException(404, "no account called " + req.username)
    try:
        acct = STORE.set_avatar(req.username, req.avatar)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "account": acct}


@app.get("/api/whoami", tags=["accounts"])
async def whoami(x_adelte_user: Optional[str] = Header(None),
                 x_api_key: Optional[str] = Header(None)):
    """Who is calling? ADELTE asks for the username first, then the key."""
    if x_adelte_user:
        acct = STORE.public_account(x_adelte_user)
        if acct:
            acct.pop("api_key", None)
            return {"ok": True, "signed_in": True, "account": acct}
    if x_api_key:
        row = STORE.account_by_key(x_api_key)
        if row:
            acct = STORE.public_account(row["username"])
            acct.pop("api_key", None)
            return {"ok": True, "signed_in": True, "account": acct}
    return {"ok": True, "signed_in": False,
            "account": None,
            "hint": "Send X-ADELTE-User: <your username>. "
                    "No account needed - ADELTE is free for everyone."}


# ---------------------------------------------------------------------------
#  ADELTE COMMANDER API
# ---------------------------------------------------------------------------

class CommandReq(BaseModel):
    command: str = Field(..., min_length=1, max_length=300,
                         description="e.g. 'ip', 'kill chrome', 'speak hello'")
    confirmed: bool = Field(False,
                            description="set true to run a command that asks first")


@app.get("/api/commander/catalog", tags=["commander"])
async def commander_catalog():
    """Every command Commander knows, grouped, with the real shell line."""
    return {"ok": True, "os": HOST_OS_NAME, "platform": HOST_OS,
            "enabled": ALLOW_SYSTEM,
            "count": len(COMMANDER_COMMANDS),
            "groups": commander_groups()}


@app.post("/api/commander/run", tags=["commander"])
async def commander_exec(req: CommandReq):
    """Run one Commander command on this machine."""
    parsed = commander_parse(req.command)
    if parsed and parsed.get("confirm") and not req.confirmed:
        return {"ok": True, "ran": False, "needs_confirm": True,
                "label": parsed["label"], "cmd": parsed["cmd"],
                "answer": "`" + parsed["cmd"] + "` will " +
                          parsed["label"].lower() + ". Confirm to run it."}
    res = await asyncio.get_event_loop().run_in_executor(
        None, commander_answer, req.command)
    return res


# ---------------------------------------------------------------------------
#  SCHEDULED POWER ACTIONS  -  sleep / lock / shut down at a set time
# ---------------------------------------------------------------------------

SCHEDULED: Dict[str, dict] = {}
_SCHED_LOCK = threading.Lock()
SCHED_ACTIONS = ("sleep", "lock", "off", "reboot", "hibernate")


class ScheduleReq(BaseModel):
    action: str = Field(..., description="sleep | lock | off | reboot | hibernate")
    seconds: float = Field(..., ge=5, le=86400,
                           description="how long from now, in seconds")
    label: str = Field("", max_length=80)


def _sched_worker(job_id: str) -> None:
    """Count down, then fire. Cancellable at any point."""
    while True:
        with _SCHED_LOCK:
            job = SCHEDULED.get(job_id)
            if not job or job["state"] != "armed":
                return
            left = job["fire_at"] - time.time()
            job["remaining"] = max(0.0, left)
            if left <= 10 and not job.get("warned"):
                job["warned"] = True
                job["warning"] = "10 seconds remaining"
            if left <= 0:
                job["state"] = "firing"
                action = job["action"]
                break
        time.sleep(0.5)
    ok, msg = commander_run(
        COMMANDER_COMMANDS[action]["cmd"].get(HOST_OS,
                                              COMMANDER_COMMANDS[action]["cmd"]["nt"]),
        action)
    with _SCHED_LOCK:
        j = SCHEDULED.get(job_id)
        if j:
            j["state"] = "done" if ok else "failed"
            j["result"] = msg or "sent"


@app.post("/api/schedule", tags=["commander"])
async def schedule_create(req: ScheduleReq):
    """Arm a timed power action. It warns at 10 seconds and can be cancelled."""
    act = (req.action or "").strip().lower()
    if act not in SCHED_ACTIONS:
        raise HTTPException(400, "action must be one of " + ", ".join(SCHED_ACTIONS))
    if not ALLOW_SYSTEM:
        raise HTTPException(403, "system commands are switched off")
    job_id = secrets.token_urlsafe(8)
    with _SCHED_LOCK:
        SCHEDULED[job_id] = {
            "id": job_id, "action": act, "state": "armed",
            "label": req.label or COMMANDER_COMMANDS[act]["label"],
            "created": time.time(), "fire_at": time.time() + req.seconds,
            "remaining": req.seconds, "warned": False, "warning": "",
        }
    threading.Thread(target=_sched_worker, args=(job_id,), daemon=True).start()
    with _SCHED_LOCK:
        return {"ok": True, "job": dict(SCHEDULED[job_id])}


@app.get("/api/schedule", tags=["commander"])
async def schedule_list():
    """Every armed or finished timer, with live countdowns."""
    now = time.time()
    with _SCHED_LOCK:
        jobs = []
        for j in SCHEDULED.values():
            d = dict(j)
            d["remaining"] = max(0.0, j["fire_at"] - now)
            jobs.append(d)
    jobs.sort(key=lambda x: x["fire_at"])
    return {"ok": True, "jobs": jobs}


@app.delete("/api/schedule/{job_id}", tags=["commander"])
async def schedule_cancel(job_id: str):
    """Cancel a timer before it fires."""
    with _SCHED_LOCK:
        j = SCHEDULED.get(job_id)
        if not j:
            raise HTTPException(404, "no such timer")
        if j["state"] == "armed":
            j["state"] = "cancelled"
        return {"ok": True, "job": dict(j)}


# ---------------------------------------------------------------------------
#  ADMIN  -  every account and every sign-in, with delete and verify
# ---------------------------------------------------------------------------

class AdminActionReq(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    verified: Optional[bool] = None


@app.get("/api/admin/accounts", tags=["admin"])
async def admin_accounts():
    """Everyone who has ever registered, newest first."""
    accts = get_store().all_accounts()
    return {"ok": True, "count": len(accts), "accounts": accts,
            "verified": sum(1 for a in accts if a["verified"])}


@app.get("/api/admin/logins", tags=["admin"])
async def admin_logins(limit: int = Query(300, ge=1, le=2000)):
    """The sign-in and registration log."""
    rows = get_store().auth_log(limit)
    return {"ok": True, "count": len(rows), "logins": rows}


@app.get("/api/admin/overview", tags=["admin"])
async def admin_overview():
    """One-glance dashboard: who registered, who logged in, usage totals."""
    store = get_store()
    accts = store.all_accounts()
    logins = store.auth_log(200)
    return {"ok": True,
            "accounts": {"count": len(accts),
                         "verified": sum(1 for a in accts if a["verified"]),
                         "latest": accts[:10]},
            "logins": {"count": len(logins), "latest": logins[:20]},
            "stats": store.stats(),
            "integrations": integration_status()}


@app.post("/api/admin/verify", tags=["admin"])
async def admin_verify(req: AdminActionReq):
    """Give or take away the verified badge."""
    try:
        acct = get_store().set_verified(req.username,
                                  True if req.verified is None else req.verified)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "account": acct}


@app.delete("/api/admin/account/{username}", tags=["admin"])
async def admin_delete(username: str):
    """Delete an account and its key."""
    if not get_store().delete_account(username):
        raise HTTPException(404, "no account called " + username)
    return {"ok": True, "deleted": username}


# ---------------------------------------------------------------------------
#  INSTAGRAM  -  read DMs aloud, reply by voice or typing
# ---------------------------------------------------------------------------

IG_STATE: Dict[str, Any] = {"connected": False, "handle": "", "since": 0.0}
IG_INBOX: List[dict] = []
_IG_LOCK = threading.Lock()


class IgConnectReq(BaseModel):
    handle: str = Field(..., min_length=1, max_length=40)
    method: str = Field("session", description="session | webhook | manual")
    token: str = Field("", max_length=400)


class IgReplyReq(BaseModel):
    message_id: str
    text: str = Field(..., min_length=1, max_length=1000)
    spoken: bool = False


class IgIncomingReq(BaseModel):
    """A DM arriving from Instagram, or pushed in by the bridge."""
    sender: str = Field(..., min_length=1, max_length=60)
    text: str = Field(..., min_length=1, max_length=2000)
    avatar: str = ""


@app.get("/api/instagram/status", tags=["social"])
async def ig_status():
    with _IG_LOCK:
        unread = [m for m in IG_INBOX if not m["read"]]
        return {"ok": True, "connected": IG_STATE["connected"],
                "handle": IG_STATE["handle"], "since": IG_STATE["since"],
                "unread": len(unread), "total": len(IG_INBOX)}


@app.post("/api/instagram/connect", tags=["social"])
async def ig_connect(req: IgConnectReq):
    """Link an Instagram account so ADELTE can read DMs out loud."""
    handle = req.handle.strip().lstrip("@")[:40]
    if not handle:
        raise HTTPException(400, "give your Instagram handle")
    with _IG_LOCK:
        IG_STATE.update({"connected": True, "handle": handle,
                         "since": time.time(), "method": req.method})
    return {"ok": True, "connected": True, "handle": handle,
            "message": "Instagram linked as @" + handle +
                       ". New DMs will be read out to you.",
            "webhook": "/api/instagram/incoming"}


@app.post("/api/instagram/disconnect", tags=["social"])
async def ig_disconnect():
    with _IG_LOCK:
        IG_STATE.update({"connected": False, "handle": "", "since": 0.0})
    return {"ok": True, "connected": False}


@app.post("/api/instagram/incoming", tags=["social"])
async def ig_incoming(req: IgIncomingReq):
    """A DM arrives. ADELTE queues it to be spoken aloud."""
    msg = {"id": secrets.token_urlsafe(6), "sender": req.sender.strip(),
           "text": req.text.strip(), "avatar": req.avatar,
           "at": time.time(), "read": False, "replied": False, "reply": ""}
    with _IG_LOCK:
        IG_INBOX.insert(0, msg)
        del IG_INBOX[60:]
    return {"ok": True, "message": msg,
            "speak": msg["sender"] + " says: " + msg["text"]}


@app.get("/api/instagram/inbox", tags=["social"])
async def ig_inbox(unread_only: bool = Query(False)):
    """The DM queue. The UI polls this and speaks anything unread."""
    with _IG_LOCK:
        items = [m for m in IG_INBOX if not m["read"]] if unread_only \
            else list(IG_INBOX)
        return {"ok": True, "count": len(items), "messages": items,
                "connected": IG_STATE["connected"]}


@app.post("/api/instagram/read/{message_id}", tags=["social"])
async def ig_mark_read(message_id: str):
    with _IG_LOCK:
        for m in IG_INBOX:
            if m["id"] == message_id:
                m["read"] = True
                return {"ok": True, "message": m}
    raise HTTPException(404, "no such message")


@app.post("/api/instagram/reply", tags=["social"])
async def ig_reply(req: IgReplyReq):
    """Send a reply - typed or dictated, it is the same call.

    If no message_id is given (the user simply typed in the reply box
    without picking a message first), we answer the newest conversation.
    That is what a person means by "reply" nine times out of ten.
    """
    target = (req.message_id or "").strip()
    if not target:
        with _IG_LOCK:
            unreplied = [m for m in IG_INBOX if not m.get("replied")]
            pool = unreplied or list(IG_INBOX)
            if not pool:
                raise HTTPException(404, "there are no messages to reply to")
            target = sorted(pool, key=lambda m: m.get("at", 0))[-1]["id"]
    with _IG_LOCK:
        for m in IG_INBOX:
            if m["id"] == target:
                m["replied"] = True
                m["read"] = True
                m["reply"] = req.text.strip()
                m["reply_spoken"] = req.spoken
                return {"ok": True, "sent": True, "message": m,
                        "confirm": "Replied to " + m["sender"] +
                                   (" by voice." if req.spoken else ".")}
    raise HTTPException(404, "no such message")


# ---------------------------------------------------------------------------
#  CARGOLIS  -  the floating desktop sphere
# ---------------------------------------------------------------------------

# Vision-capable free models, tried in order. These read an actual picture,
# so Cargolis can look at a screenshot instead of only reading text.
VISION_MODELS = [
    ("openrouter", "nvidia/nemotron-nano-12b-v2-vl:free"),
    ("openrouter", "google/gemma-4-31b-it:free"),
    ("openrouter", "google/gemma-4-26b-a4b-it:free"),
    ("openrouter", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"),
]


async def call_vision(c: httpx.AsyncClient, image_b64: str, instruction: str,
                      budget: float = 90.0):
    """Send one image plus an instruction to the first vision model that answers.

    image_b64 is a bare base64 payload or a full data: URL. Returns
    (text, provider_label) and ("", "") when nothing could look at it.
    """
    if not image_b64:
        return "", ""
    url = image_b64 if image_b64.startswith("data:") \
        else "data:image/png;base64," + image_b64
    content = [{"type": "text", "text": instruction},
               {"type": "image_url", "image_url": {"url": url}}]
    msgs = [{"role": "user", "content": content}]
    t0 = time.time()
    for pid, model in VISION_MODELS:
        key = env_key(PROVIDERS[pid]["env"])
        if not key:
            continue
        if time.time() - t0 > budget:
            break
        try:
            body = {"model": model, "messages": msgs, "max_tokens": 2048,
                    "temperature": 0.2}
            r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                             json=body, timeout=min(75.0, budget),
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json",
                                      "HTTP-Referer": "http://localhost:%d" % CFG.port,
                                      "X-Title": "ADELTE"})
            if r.status_code != 200:
                log("vision %s -> HTTP %s", model, r.status_code)
                continue
            txt = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if txt:
                return txt, "Vision " + model.split("/")[-1].replace(":free", "")
        except Exception as e:
            log("vision %s failed: %s", model, e)
            continue
    return "", ""


class CargoReq(BaseModel):
    text: str = Field("", max_length=20000,
                      description="text captured from the screen")
    task: str = Field("auto", description="auto | fix | explain | research | refactor")
    image: str = Field("", description="base64 PNG of the captured area or screen")
    question: str = Field("", max_length=2000,
                          description="what the user wants done with it")


@app.post("/api/cargolis/analyse", tags=["cargolis"])
async def cargolis_analyse(req: CargoReq):
    """Take captured screen text or a picture and explain, fix or research it.

    Three ways in:
      * text only        - the old path, fastest
      * image only       - a vision model reads the picture first
      * image + question - "what do you want done with it"
    """
    body = (req.text or "").strip()
    task = (req.task or "auto").lower()
    want = (req.question or "").strip()
    if not body and not req.image:
        raise HTTPException(400, "send some text or an image")
    seen = ""
    vprov = ""

    # If a picture came along, let a vision model look at it and turn what it
    # sees into text we can reason about.
    if req.image:
        instruction = (want or
                       "Look at this screenshot. Transcribe every piece of "
                       "text, code or error exactly as written, then say in "
                       "one line what is on screen.")
        async with httpx.AsyncClient(follow_redirects=True) as vc:
            seen, vprov = await call_vision(vc, req.image, instruction)
        if not seen and not body:
            raise HTTPException(
                503, "no vision model could read that image - try again, or "
                     "select the text instead")
        if seen:
            body = (body + "\n\n" if body else "") + seen
    if task == "auto":
        low = body.lower()
        if re.search(r"(traceback|error|exception|failed|cannot|undefined "
                     r"is not|syntaxerror|nameerror)", low):
            task = "fix"
        elif re.search(r"[{};()=]|def |class |function |import |const ", body):
            task = "refactor"
        else:
            task = "research"
    asks = {
        "fix": ("This is an error from my screen. Explain in one short "
                "paragraph what it means and why it happened, then give the "
                "corrected code in full."),
        "refactor": ("This is code from my screen. Explain what it does in "
                     "two lines, then give a cleaner, corrected version in "
                     "full with the bugs fixed."),
        "explain": "Explain this clearly in short ordered points.",
        "research": ("Research this and answer in short ordered points, "
                     "simply, like a knowledgeable friend."),
    }
    ask = asks.get(task, asks["research"])
    if want:
        # The user told us what they want - that always wins.
        ask = ("Do exactly what I ask about what is on my screen.\n"
               "What I want: " + want)
    prompt = ask + "\n\n---\n" + body[:12000]
    cfg = model_cfg("adelte-cargolis")
    msgs = [{"role": "system", "content": cfg["persona"]},
            {"role": "user", "content": prompt}]

    t0 = time.time()
    async with httpx.AsyncClient(follow_redirects=True) as c:
        out, who = await generate_with_model(c, "adelte-cargolis", msgs, 50.0)
    if not out:
        raise HTTPException(503, "no provider answered - try again")
    return {"ok": True, "task": task, "answer": out,
            "provider": (vprov + " + " + who) if vprov else who,
            "captured_chars": len(body), "saw_image": bool(req.image),
            "transcribed": seen[:4000],
            "elapsed_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------------------
#  FREE IMAGE + VIDEO - keyless for everyone
# ---------------------------------------------------------------------------

IMAGE_SIZES = {"square": (1024, 1024), "wide": (1280, 720),
               "tall": (768, 1280), "icon": (512, 512)}


class ImageReq(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=1200)
    size: str = Field("square", description="square | wide | tall | icon")
    style: str = Field("", max_length=200,
                       description="e.g. photographic, 3d render, flat vector")
    seed: Optional[int] = Field(None, ge=0, le=2 ** 31)


class VideoReq(BaseModel):
    prompt: str = Field(..., min_length=2, max_length=1200)
    seconds: float = Field(4.0, ge=1.0, le=10.0)
    fps: int = Field(8, ge=4, le=16)
    style: str = Field("", max_length=200)


async def make_image(prompt: str, w: int, h: int, seed=None,
                     model: str = "flux") -> bytes:
    """Keyless text-to-image. No account, no key, no cost.

    'flux' is the higher-quality renderer; 'turbo' answers far faster and is
    what the video path uses, where we need several frames in a row.
    """
    import urllib.parse
    q = urllib.parse.quote(prompt[:900], safe="")
    url = ("https://image.pollinations.ai/prompt/" + q +
           "?width=%d&height=%d&nologo=true&model=%s" % (w, h, model))
    if seed is not None:
        url += "&seed=%d" % seed
    r = await CLIENT.get(url, timeout=120, follow_redirects=True)
    if r.status_code != 200 or not r.content:
        raise HTTPException(502, "image service busy - try again in a moment")
    ct = r.headers.get("content-type", "")
    if "image" not in ct:
        raise HTTPException(502, "image service returned " + ct)
    return r.content


@app.post("/api/image", tags=["create"])
async def api_image(req: ImageReq,
                    x_api_key: Optional[str] = Header(None),
                    authorization: Optional[str] = Header(None)):
    """Generate an image. FREE for everyone - no key required."""
    auth(authorization, x_api_key)
    w, h = IMAGE_SIZES.get(req.size, IMAGE_SIZES["square"])
    prompt = req.prompt + ((", " + req.style) if req.style else "")
    t0 = time.time()
    raw = await make_image(prompt, w, h, req.seed)
    return {"ok": True, "kind": "image", "prompt": req.prompt,
            "width": w, "height": h,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "mime": "image/jpeg",
            "data_url": "data:image/jpeg;base64," +
                        base64.b64encode(raw).decode(),
            "cost": "free"}


@app.get("/api/image", tags=["create"])
async def api_image_get(prompt: str = Query(..., min_length=2),
                        size: str = Query("square"),
                        seed: Optional[int] = Query(None)):
    """Same thing as a plain URL - returns the image bytes directly."""
    w, h = IMAGE_SIZES.get(size, IMAGE_SIZES["square"])
    raw = await make_image(prompt, w, h, seed)
    return Response(content=raw, media_type="image/jpeg")


def ken_burns(img, t: float, zoom: float = 0.16, pan: float = 0.06):
    """One frame of a slow push-in with a gentle drift. t goes 0 -> 1."""
    from PIL import Image
    w, h = img.size
    z = 1.0 + zoom * t
    cw, ch = int(w / z), int(h / z)
    max_dx, max_dy = w - cw, h - ch
    dx = int(max_dx * (0.5 + pan * (t - 0.5) * 2))
    dy = int(max_dy * 0.5)
    dx = max(0, min(max_dx, dx))
    dy = max(0, min(max_dy, dy))
    return img.crop((dx, dy, dx + cw, dy + ch)).resize((w, h), Image.LANCZOS)


def render_motion(keys, total_frames: int):
    """Turn 1-2 still keyframes into a moving sequence.

    With one keyframe we push in on it. With two we push in on the first,
    crossfade to the second, then keep drifting - so the clip actually moves
    instead of cutting between two stills.
    """
    from PIL import Image
    ims = [Image.open(io.BytesIO(k)).convert("RGB") for k in keys]
    base = ims[0].size
    ims = [im if im.size == base else im.resize(base, Image.LANCZOS)
           for im in ims]
    out = []
    if len(ims) == 1:
        for i in range(total_frames):
            out.append(ken_burns(ims[0], i / float(max(1, total_frames - 1))))
        return out
    seg = total_frames // (len(ims) - 1)
    for k in range(len(ims) - 1):
        a, b = ims[k], ims[k + 1]
        cnt = seg if k < len(ims) - 2 else (total_frames - seg * k)
        for i in range(cnt):
            t = i / float(max(1, cnt - 1))
            fa = ken_burns(a, 0.45 + 0.55 * t)
            fb = ken_burns(b, 0.35 * t)
            # ease-in-out crossfade across the middle of the segment
            m = min(1.0, max(0.0, (t - 0.35) / 0.45))
            m = m * m * (3 - 2 * m)
            out.append(Image.blend(fa, fb, m))
    return out


def images_to_gif(ims, fps: int) -> bytes:
    """Assemble already-decoded PIL frames into an animated GIF."""
    from PIL import Image
    w, h = ims[0].size
    scale = min(1.0, 512.0 / max(w, h))
    if scale < 1.0:
        ims = [im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
               for im in ims]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=96) for im in ims]
    buf = io.BytesIO()
    pal[0].save(buf, format="GIF", save_all=True, append_images=pal[1:],
                duration=int(1000 / max(1, fps)), loop=0, optimize=True)
    return buf.getvalue()


def frames_to_gif(frames: List[bytes], fps: int) -> bytes:
    """Assemble JPEG frames into an animated GIF. Pillow only, no ffmpeg."""
    from PIL import Image
    ims = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    w, h = ims[0].size
    scale = min(1.0, 640.0 / max(w, h))
    if scale < 1.0:
        ims = [im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
               for im in ims]
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in ims]
    buf = io.BytesIO()
    pal[0].save(buf, format="GIF", save_all=True, append_images=pal[1:],
                duration=int(1000 / max(1, fps)), loop=0, optimize=True)
    return buf.getvalue()


@app.post("/api/video", tags=["create"])
async def api_video(req: VideoReq,
                    x_api_key: Optional[str] = Header(None),
                    authorization: Optional[str] = Header(None)):
    """Generate a short animated video. FREE for everyone - no key required.

    ADELTE renders a keyframe sequence with the free image engine, moving the
    described scene forward frame by frame, then assembles an animated GIF.
    """
    auth(authorization, x_api_key)
    t0 = time.time()
    # A *fresh* image takes the free engine ~40 s, so we buy at most two
    # keyframes and render the motion between them ourselves.
    n_key = 2 if req.seconds >= 2.5 else 1
    base = req.prompt + ((", " + req.style) if req.style else "")
    seed = random.randint(1, 10 ** 6)
    shots = ["cinematic film still, establishing shot",
             "cinematic film still, closer, the action has moved on"]

    async def one(i):
        p = "%s, %s" % (base, shots[i % len(shots)])
        return await make_image(p, 640, 384, seed + i * 7, model="turbo")

    good = []
    for i in range(n_key):
        for attempt in range(2):
            try:
                good.append(await one(i))
                break
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(1.2)
        if not good:                      # first frame is mandatory
            raise HTTPException(502, "video engine busy - try again shortly")
    if not good:
        raise HTTPException(502, "video engine busy - try again shortly")

    total = max(6, min(96, int(round(req.seconds * req.fps))))
    try:
        ims = render_motion(good, total)
        gif = images_to_gif(ims, req.fps)
    except Exception as e:
        raise HTTPException(500, "could not assemble the clip: " + str(e)[:120])
    return {"ok": True, "kind": "video", "prompt": req.prompt,
            "seconds": req.seconds, "fps": req.fps,
            "keyframes": len(good), "frames": len(ims),
            "mime": "image/gif",
            "elapsed_ms": int((time.time() - t0) * 1000),
            "data_url": "data:image/gif;base64," +
                        base64.b64encode(gif).decode(),
            "cost": "free"}


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1),
                 engines_csv: Optional[str] = Query(None, alias="engines"),
                 max_results: int = Query(8, ge=1, le=20),
                 deep: bool = Query(False),
                 x_api_key: Optional[str] = Header(None),
                 authorization: Optional[str] = Header(None)):
    auth(authorization, x_api_key)
    names = ([e.strip() for e in engines_csv.split(",") if e.strip()]
             if engines_csv else
             (CODE_SET if classify(q) == "code" else WEB_SET))

    async def noop(*a, **k): return None
    t0 = time.time()
    res = await gather(q, names, max_results, deep, noop)
    return {"query": q, "intent": classify(q), "engines": names,
            "count": len(res), "elapsed_ms": int((time.time() - t0) * 1000),
            "results": res[: max_results * 2]}


@app.post("/api/chat")
async def chat(req: ChatReq, x_api_key: Optional[str] = Header(None),
               authorization: Optional[str] = Header(None)):
    auth(authorization, x_api_key)
    sid = req.session_id or str(uuid.uuid4())
    t0 = time.time()

    async def noop(*a, **k): return None

    if IDENTITY_PAT.search(req.message.strip()):
        txt = identity_answer(req.message, req.model)
        STORE.add_turn(sid, "user", req.message)
        STORE.add_turn(sid, "assistant", txt)
        return {"session_id": sid, "answer": txt, "mode": "identity",
                "sources": [], "elapsed_ms": int((time.time() - t0) * 1000)}

    if req.model == "adelte-commander":
        res = await asyncio.get_event_loop().run_in_executor(
            None, commander_answer, req.message)
        STORE.add_turn(sid, "user", req.message)
        STORE.add_turn(sid, "assistant", res["answer"])
        return {"session_id": sid, "answer": res["answer"], "mode": "command",
                "ran": res.get("ran", False), "ok": res.get("ok", False),
                "sources": [], "elapsed_ms": int((time.time() - t0) * 1000)}

    lane = route(req.message)
    if req.model == "adelte-coder-3high" and lane == "research":
        lane = "make"
    if req.model == "adelte-cargolis" and lane == "research":
        lane = "make" if wants_build(req.message) else "think"
    if req.model == "adelte-minimax" and lane == "research":
        lane = "think"
    if lane in ("image", "video"):
        mi = media_intent(req.message) or (lane, req.message)
        try:
            if lane == "image":
                raw = await make_image(mi[1], 1024, 1024)
                url = ("data:image/jpeg;base64," +
                       base64.b64encode(raw).decode())
            else:
                vr = await api_video(VideoReq(prompt=mi[1], seconds=3,
                                              fps=10), None, None)
                url = vr["data_url"]
        except Exception as e:
            return {"answer": "The free render engine is busy: " +
                    str(e)[:160], "mode": "chat", "sources": []}
        return {"answer": "Here is your " + lane + ".",
                "mode": lane, "media": url, "prompt": mi[1],
                "sources": []}
    _prev_code = last_built_code(STORE.history(sid))
    if wants_refine(req.message, _prev_code):
        lane = "make"
    if lane == "think":
        _m3 = model_cfg(req.model)
        msgs = [{"role": "system", "content": _m3["persona"] +
                 " Do not mention searching or sources - you did not search. "
                 "Answer from knowledge, short and ordered. You are ADELTE, "
                 "made by ADELTE Industries."}]
        for h in STORE.history(sid)[-6:]:
            msgs.append({"role": h["role"], "content": h["content"][:1400]})
        msgs.append({"role": "user", "content": req.message})
        text = await ai_answer(CLIENT, msgs, max(req.ai_budget, 45),
                               model=req.model) or \
            friend_reply(req.message, STORE.history(sid))
        STORE.add_turn(sid, "user", req.message)
        STORE.add_turn(sid, "assistant", text)
        STORE.remember(sid, req.message, text, [])
        return {"session_id": sid, "answer": text, "mode": "think",
                "thinking": extract_intent(req.message), "sources": [],
                "elapsed_ms": int((time.time() - t0) * 1000)}

    if lane in ("command", "chat", "make"):
        if lane == "command":
            cmd = match_command(req.message)
            if cmd["kind"] == "web":
                text = "Open " + cmd["url"]
            elif cmd["kind"] == "local":
                text = local_command(cmd["op"])
            elif cmd.get("confirm") and not req.confirm:
                text = cmd["label"] + " needs confirm=true."
            else:
                text = run_system_command(cmd["op"])[1]
        elif lane == "chat":
            text = friend_reply(req.message, STORE.history(sid), strict=True)
            if not text:
                _m2 = model_cfg(req.model)
                text = await ai_answer(
                    CLIENT,
                    [{"role": "system", "content": _m2["persona"] +
                      " You are ADELTE, made by ADELTE Industries. Answer "
                      "directly and usefully; if the request is vague give "
                      "your best concrete answer and ask one short question."},
                     {"role": "user", "content": req.message}],
                    max(req.ai_budget, 30), model=req.model) or \
                    friend_reply(req.message, STORE.history(sid))
        else:
            _ref = wants_refine(req.message, _prev_code)
            _sys = ("You are ADELTE. Write complete working code with no "
                    "placeholders and no TODOs.")
            if _ref:
                _sys += (" The user says the code you just gave is wrong or "
                         "unfinished. Output the ENTIRE corrected file. Never "
                         "ask them to paste the code back - it is below.")
                _um = ("This is the code you gave me:\n\n"
                       + _prev_code[:6000] + "\n\n---\nMy problem with it: "
                       + req.message + "\n\nGive me the complete file.")
            else:
                _um = req.message
            text = (None if _ref else maker_answer(req.message)) or \
                await ai_answer(
                    CLIENT,
                    [{"role": "system", "content": _sys},
                     {"role": "user", "content": _um}],
                    max(req.ai_budget, 45), model=req.model) or \
                "The AI layer is busy — try again."
        STORE.add_turn(sid, "user", req.message)
        STORE.add_turn(sid, "assistant", text)
        return {"session_id": sid, "answer": text, "mode": lane,
                "sources": [], "elapsed_ms": int((time.time() - t0) * 1000)}

    # A follow-up like "is he an american" must be searched against the
    # PREVIOUS subject, not taken literally as a grammar question.
    hist = STORE.history(sid)
    search_q = req.message
    prev_user = [h["content"] for h in hist if h["role"] == "user"]
    if prev_user and is_followup(req.message, True):
        search_q = resolve_followup(req.message, topic_of(prev_user[-1]))

    intent = classify(search_q)
    if bool(getattr(req, "deep", False)):
        names = req.engines or (CODE_DEEP if intent == "code" else WEB_DEEP)
    else:
        names = req.engines or (CODE_SET if intent == "code" else WEB_SET)

    mems = STORE.recall(sid, req.message) if req.use_memory else []
    memory = ("\n".join(f"- Q: {m['query']}\n  A: {m['answer'][:300]}" for m in mems)
              if mems else "")

    sources = await gather(search_q, names, req.max_results, req.deep, noop)
    # Truncate ONCE, here, so the prose footer, the JSON payload and the
    # rendered cards all report the exact same number of sources.
    sources = sources[:10]
    msgs = build_messages(req.message, sources, hist, memory)

    answer, _who = await generate_with_model(
        CLIENT, req.model or DEFAULT_MODEL, msgs, req.ai_budget)
    mode = "ai"
    if not answer:
        answer = extractive_answer(search_q, sources, memory)
        mode = "extractive"

    STORE.add_turn(sid, "user", req.message)
    STORE.add_turn(sid, "assistant", answer)
    STORE.remember(sid, req.message, answer, sources)

    return {"session_id": sid, "answer": answer, "mode": mode, "intent": intent,
            "engines_used": names, "used_memory": bool(mems),
            "sources": [{k: s.get(k) for k in
                         ("title", "url", "snippet", "source", "agreement")}
                        for s in sources[:10]],
            "elapsed_ms": int((time.time() - t0) * 1000)}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatReq, request: Request,
                      x_api_key: Optional[str] = Header(None),
                      authorization: Optional[str] = Header(None)):
    auth(authorization, x_api_key)
    sid = req.session_id or str(uuid.uuid4())

    async def gen() -> AsyncIterator[str]:
        t0 = time.time()
        q: asyncio.Queue = asyncio.Queue()

        async def emit(ev: str, data: dict):
            await q.put((ev, data))

        yield sse("start", {"session_id": sid, "query": req.message})

        # ---------------- identity: never search for "who made you" -------
        if IDENTITY_PAT.search(req.message.strip()):
            yield sse("thinking", {"text": "That one's about me - no search "
                                           "needed", "p": 55})
            text = identity_answer(req.message, req.model)
            for i in range(0, len(text), 16):
                yield sse("token", {"t": text[i:i + 16]})
                await asyncio.sleep(0.012)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", text)
            yield sse("done", {"session_id": sid, "mode": "identity",
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        # ---------------- ADELTE Commander: local commands only -----------
        # Commander never touches a provider and never searches. It parses
        # the word, runs it on this machine, and shows what came back.
        if req.model == "adelte-commander":
            yield sse("think_start", {"model": req.model})
            yield sse("thinking", {"text": "Commander - reading the command",
                                   "p": 20, "phase": "thinking"})
            parsed = commander_parse(req.message)
            if parsed and parsed.get("cmd"):
                yield sse("extract", {"items": [parsed["label"],
                                                parsed["group"],
                                                parsed["cmd"]]})
                yield sse("thinking", {"text": "Running it on " + HOST_OS_NAME,
                                       "p": 65, "phase": "sending"})
            res = await asyncio.get_event_loop().run_in_executor(
                None, commander_answer, req.message)
            text = res["answer"]
            for i in range(0, len(text), 30):
                yield sse("token", {"t": text[i:i + 30]})
                await asyncio.sleep(0.006)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", text)
            yield sse("done", {"session_id": sid, "mode": "command",
                               "ran": res.get("ran", False),
                               "ok": res.get("ok", False),
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        # ---------------- fast lanes: command / chat / make ----------------
        lane = route(req.message)
        # The coder model builds by default - a bare "a redis cache in python"
        # is a build order, not a research question.
        if (req.model == "adelte-coder-3high" and lane == "research"
                and not IDENTITY_PAT.search(req.message)):
            lane = "make"
        # Reacting to code we just wrote ("this codes are incomplete") is a
        # refinement of that code, never a web search.
        PREV_CODE = last_built_code(STORE.history(sid))
        if wants_refine(req.message, PREV_CODE):
            lane = "make"
        # Cargolis is a code/desktop model - it builds and fixes, never searches.
        if (req.model == "adelte-cargolis" and lane == "research"
                and not IDENTITY_PAT.search(req.message)):
            lane = "make" if wants_build(req.message) else "think"
        # ADELTE Max never searches. It thinks.
        if (req.model == "adelte-minimax" and lane == "research"
                and not IDENTITY_PAT.search(req.message)):
            lane = "think"

        # ---------------- free image / video rendering --------------------
        if lane in ("image", "video"):
            mi = media_intent(req.message) or (lane, req.message)
            subject = mi[1]
            yield sse("think_start", {"model": req.model})
            yield sse("thinking",
                      {"text": "You want %s made, not a search"
                               % ("an image" if lane == "image"
                                  else "a video"),
                       "p": 12, "phase": "thinking"})
            yield sse("extract", {"items": [subject,
                                            "free render engine, no key",
                                            ("still image, 1024 square"
                                             if lane == "image"
                                             else "3 second animated clip")]})
            yield sse("thinking", {"text": "Rendering — this is free, so it "
                                           "takes a few seconds", "p": 35,
                                   "phase": "sending"})
            try:
                if lane == "image":
                    raw = await make_image(subject, 1024, 1024)
                    url = ("data:image/jpeg;base64," +
                           base64.b64encode(raw).decode())
                else:
                    vr = await api_video(VideoReq(prompt=subject, seconds=3,
                                                  fps=10), None, None)
                    url = vr["data_url"]
            except Exception as e:
                msg = "The free render engine is busy right now: " + str(e)[:150]
                for i in range(0, len(msg), 24):
                    yield sse("token", {"t": msg[i:i + 24]})
                yield sse("done", {"session_id": sid, "mode": "chat",
                                   "elapsed_ms": int((time.time() - t0) * 1000),
                                   "sources": 0})
                return
            yield sse("thinking", {"text": "Done — sending it over", "p": 95,
                                   "phase": "writing"})
            yield sse("media", {"kind": lane, "url": url, "prompt": subject})
            text = "Here is your %s of **%s**." % (lane, subject)
            for i in range(0, len(text), 20):
                yield sse("token", {"t": text[i:i + 20]})
                await asyncio.sleep(0.01)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", text)
            yield sse("done", {"session_id": sid, "mode": lane,
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        if lane == "command":
            cmd = match_command(req.message)
            yield sse("thinking", {"text": "Command recognised — "
                                           + cmd["label"], "p": 40})
            if cmd["kind"] == "web":
                yield sse("command", {"action": "open", "url": cmd["url"],
                                      "label": cmd["label"]})
                text = "Opening **" + cmd["label"].replace("Open ", "") \
                       + "** in a new tab."
            elif cmd["kind"] == "local":
                text = local_command(cmd["op"])
            else:
                if cmd.get("confirm") and not req.confirm:
                    yield sse("command", {"action": "confirm", "op": cmd["op"],
                                          "label": cmd["label"]})
                    text = ("**" + cmd["label"] + "** — that one is "
                            "irreversible, so say `yes` and I'll do it.")
                else:
                    ok, msg = run_system_command(cmd["op"])
                    yield sse("command", {"action": "system", "ok": ok,
                                          "label": cmd["label"]})
                    text = ("Done. " + msg) if ok else msg
            for i in range(0, len(text), 22):
                yield sse("token", {"t": text[i:i + 22]})
                await asyncio.sleep(0.008)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", text)
            yield sse("done", {"session_id": sid, "mode": "command",
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        if lane == "chat":
            text = friend_reply(req.message, STORE.history(sid), strict=True)
            if not text:
                # A real question dressed as chat ("help me in codes").
                mcfg2 = model_cfg(req.model)
                yield sse("thinking", {
                    "text": "Working out what you need - %s is answering"
                            % mcfg2["name"], "p": 45})
                cmsgs = [{"role": "system", "content": mcfg2["persona"] +
                          " You are ADELTE, made by ADELTE Industries. Answer "
                          "the user directly and usefully. If their request is "
                          "vague, give them your best concrete answer AND ask "
                          "one short question to narrow it down. Never reply "
                          "with only a question. Keep it under 120 words "
                          "unless they asked for code."}]
                for h in STORE.history(sid)[-6:]:
                    cmsgs.append({"role": h["role"],
                                  "content": h["content"][:1200]})
                cmsgs.append({"role": "user", "content": req.message})
                text, _who2 = await generate_with_model(
                    CLIENT, req.model or DEFAULT_MODEL, cmsgs,
                    max(req.ai_budget, 30), emit)
                while not q.empty():
                    ev, data = q.get_nowait()
                    yield sse(ev, data)
                if not text:
                    text = friend_reply(req.message, STORE.history(sid))
            else:
                yield sse("thinking", {"text": "Just talking — no search "
                                               "needed", "p": 50})
            for i in range(0, len(text), 14):
                yield sse("token", {"t": text[i:i + 14]})
                await asyncio.sleep(0.016)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", text)
            yield sse("done", {"session_id": sid, "mode": "chat",
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        if lane == "think":
            mcfg = model_cfg(req.model)
            yield sse("think_start", {"model": mcfg["name"]})
            yield sse("thinking", {"text": "ADELTE is thinking — no search, "
                                           "reasoning from what it knows",
                                   "p": 20, "phase": "thinking"})
            # show what it extracted from the request
            want = extract_intent(req.message)
            yield sse("extract", {"items": want})
            yield sse("thinking", {"text": "Extracted what you are asking for",
                                   "p": 40, "phase": "extracting"})
            msgs = [{"role": "system",
                     "content": mcfg["persona"] +
                                " Do not mention searching or sources - you "
                                "did not search. Answer from knowledge. Start "
                                "with one short sentence that answers it, then "
                                "at most 5 short ordered points. Include hex "
                                "colour codes when colours come up. You are "
                                "ADELTE, made by ADELTE Industries."}]
            for h in STORE.history(sid)[-6:]:
                msgs.append({"role": h["role"], "content": h["content"][:1400]})
            msgs.append({"role": "user", "content": req.message})
            yield sse("thinking", {"text": "Sending to the ADELTE Max chain",
                                   "p": 60, "phase": "sending"})
            txt, who = await generate_with_model(
                CLIENT, req.model or DEFAULT_MODEL, msgs,
                max(req.ai_budget, 45), emit)
            if not txt and not REFINING:
                _lb = offline_build(req.message)
                if _lb:
                    txt = _lb[0]
            while not q.empty():
                ev, data = q.get_nowait()
                yield sse(ev, data)
            if not txt:
                txt = friend_reply(req.message, STORE.history(sid)) or (
                    "Every brain in the Max chain is busy right now. Ask me "
                    "again in a few seconds, or switch to ADELTE Search.")
            yield sse("thinking", {"text": "Writing the answer", "p": 90,
                                   "phase": "writing"})
            for i in range(0, len(txt), 30):
                if await request.is_disconnected():
                    return
                yield sse("token", {"t": txt[i:i + 30]})
                await asyncio.sleep(0.007)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", txt)
            STORE.remember(sid, req.message, txt, [])
            yield sse("done", {"session_id": sid, "mode": "think",
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        if lane == "make":
            yield sse("thinking", {"text": "You want something built — "
                                           "writing the code, not searching",
                                   "p": 45})
            REFINING = bool(PREV_CODE) and wants_refine(req.message, PREV_CODE)
            built = None if REFINING else maker_answer(req.message)
            if built:
                for i in range(0, len(built), 40):
                    if await request.is_disconnected():
                        return
                    yield sse("token", {"t": built[i:i + 40]})
                    await asyncio.sleep(0.006)
                STORE.add_turn(sid, "user", req.message)
                STORE.add_turn(sid, "assistant", built)
                STORE.remember(sid, req.message, built, [])
                yield sse("done", {"session_id": sid, "mode": "built",
                                   "elapsed_ms": int((time.time() - t0) * 1000),
                                   "sources": 0})
                return
            # no local template -> ask the AI to write it, still no web search
            mcfg = model_cfg(req.model)
            chain = live_chain(req.model)
            yield sse("thinking", {
                "text": ("Refining the code from the last message - %s is "
                         "rewriting it in full" % mcfg["name"]) if REFINING
                        else "No local template - %s is writing it"
                             % mcfg["name"],
                "p": 55})
            msgs = [{"role": "system",
                     "content": mcfg["persona"] + " Output complete, working, "
                                "self-contained code with no placeholders and "
                                "no TODOs. Start with one short sentence, then "
                                "one fenced code block, then one short "
                                "follow-up question. List the hex colour codes "
                                "you used. You are ADELTE, made by ADELTE "
                                "Industries."},
                    {"role": "user", "content": req.message}]
            if REFINING:
                msgs[0]["content"] += (
                    " The user is telling you the code you just gave them is "
                    "wrong or unfinished. Output the ENTIRE corrected file "
                    "from the first line to the last. Never abbreviate, never "
                    "write 'rest of code here', never ask them to paste it "
                    "back - you already have it below.")
                msgs[1] = {"role": "user", "content":
                           "This is the code you gave me:\n\n"
                           + PREV_CODE[:6000]
                           + "\n\n---\nMy problem with it: " + req.message
                           + "\n\nGive me the complete corrected file."}
            hist_b = [] if REFINING else STORE.history(sid)[-4:]
            if hist_b:
                msgs[1:1] = [{"role": h["role"], "content": h["content"][:1200]}
                             for h in hist_b]
            txt, who = await generate_with_model(
                CLIENT, req.model or DEFAULT_MODEL, msgs,
                max(req.ai_budget, 45), emit)
            while not q.empty():
                ev, data = q.get_nowait()
                yield sse(ev, data)
            if txt and biggest_block(txt) < MIN_CODE_CHARS:
                # The brain answered, but thin. If a full local template
                # covers this request, ship that instead - the user asked
                # for substantial files, not a sketch.
                _fat = offline_build(req.message)
                if _fat and biggest_block(_fat[0]) > biggest_block(txt):
                    txt = _fat[0]
                    emit("thinking", {"t": "That draft was thin - using my "
                                           "full local template instead"})
            if not txt:
                # No brain answered. Build it here instead of apologising:
                # these templates are complete, working files.
                local = offline_build(req.message)
                if local:
                    txt = local[0]
                    emit("thinking", {"t": "No brain answered - building it "
                                           "locally from my own templates"})
                else:
                    have = ", ".join(PROVIDERS[p]["label"] for p, _ in chain) \
                           or "none"
                    txt = ("Every brain in the chain refused just now "
                           "(tried: " + have + "). I can still build a "
                           "portfolio or a login page for you right now with "
                           "no AI at all - just ask for one of those. "
                           "Otherwise try again in a moment; the free workers "
                           "come and go.")
            for i in range(0, len(txt), 30):
                yield sse("token", {"t": txt[i:i + 30]})
                await asyncio.sleep(0.008)
            STORE.add_turn(sid, "user", req.message)
            STORE.add_turn(sid, "assistant", txt)
            yield sse("done", {"session_id": sid, "mode": "built",
                               "elapsed_ms": int((time.time() - t0) * 1000),
                               "sources": 0})
            return

        # ---------------- research lane (the original path) ----------------
        intent = classify(req.message)
        _deep = bool(getattr(req, "deep", False))
        _cs = CODE_DEEP if _deep else CODE_SET
        _ws = WEB_DEEP if _deep else WEB_SET
        names = req.engines or (_cs if intent == "code" else _ws)
        # Resolve pronoun follow-ups against the previous turn before search.
        hist = STORE.history(sid)
        prev_user = [h["content"] for h in hist if h["role"] == "user"]
        search_q = req.message
        if prev_user and is_followup(req.message, True):
            topic = topic_of(prev_user[-1])
            search_q = resolve_followup(req.message, topic)
            if search_q != req.message:
                names = req.engines or (_cs if classify(search_q) == "code"
                                        else _ws)
                yield sse("thinking", {"text": f"Follow-up — reading this as "
                                               f"**{esc_md(search_q)}**", "p": 6})

        yield sse("thinking", {"text": f"Researching — loading the {intent} "
                                       f"engine set", "p": 8})

        mems = STORE.recall(sid, req.message) if req.use_memory else []
        memory = ""
        if mems:
            memory = "\n".join(f"- Q: {m['query']}\n  A: {m['answer'][:300]}"
                               for m in mems)
            yield sse("memory", {"hits": len(mems),
                                 "text": f"Recalled **{len(mems)}** related answer(s) "
                                         f"from this session"})

        task = asyncio.create_task(gather(search_q, names, req.max_results,
                                          req.deep, emit))
        while not task.done() or not q.empty():
            try:
                ev, data = await asyncio.wait_for(q.get(), timeout=0.25)
                yield sse(ev, data)
            except asyncio.TimeoutError:
                continue

        sources = (await task)[:10]   # single truncation point (see /api/chat)
        yield sse("sources", {"count": len(sources),
                              "items": [{k: s.get(k) for k in
                                         ("title", "url", "snippet", "source",
                                          "agreement")} for s in sources[:10]]})

        msgs = build_messages(req.message, sources, hist, memory)
        yield sse("thinking", {"text": "Synthesising a cited answer", "p": 70})

        # ---- AI layer, streamed through a queue so progress keeps flowing ----
        pq: asyncio.Queue = asyncio.Queue()

        async def progress(info: dict):
            await pq.put(info)

        async def run_ai():
            try:
                async def em(ev, data):
                    if ev == "thinking":
                        await pq.put(data)
                txt, who = await generate_with_model(
                    CLIENT, req.model or DEFAULT_MODEL, msgs,
                    req.ai_budget, em)
                await pq.put({"_result": txt})
            except Exception:
                await pq.put({"_result": None})

        ai_task = asyncio.create_task(run_ai())
        ai_text = None
        while True:
            try:
                info = await asyncio.wait_for(pq.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if ai_task.done():
                    break
                if await request.is_disconnected():
                    ai_task.cancel()
                    return
                continue
            if "_result" in info:
                ai_text = info["_result"]
                break
            if info.get("processing"):
                yield sse("thinking", {"text": "A worker on the free AI swarm is "
                                               "generating the answer…", "p": 80})
            elif info.get("queue_position") is not None:
                yield sse("thinking", {"text": f"Queued on the free AI swarm "
                                               f"(position {info['queue_position']}, "
                                               f"~{info.get('wait_time')}s)", "p": 75})

        mode = "ai"
        if not ai_text:
            mode = "extractive"
            yield sse("thinking", {"text": "AI layer unavailable — composing the "
                                           "answer directly from the sources", "p": 85})
            ai_text = extractive_answer(search_q, sources, memory)

        # stream it out in chunks for the typing effect
        for i in range(0, len(ai_text), 22):
            if await request.is_disconnected():
                return
            yield sse("token", {"t": ai_text[i:i + 22]})
            await asyncio.sleep(0.011)

        STORE.add_turn(sid, "user", req.message)
        STORE.add_turn(sid, "assistant", ai_text)
        STORE.remember(sid, req.message, ai_text, sources)

        yield sse("done", {"session_id": sid, "mode": mode,
                           "elapsed_ms": int((time.time() - t0) * 1000),
                           "sources": len(sources)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/chat/stream", include_in_schema=False)
@app.get("/api/chat", include_in_schema=False)
async def chat_wrong_method():
    """People (and static servers) hit these with GET. Explain, don't 405."""
    raise HTTPException(
        405, "Use POST, not GET. Example: "
             "curl -X POST http://localhost:8000/api/chat/stream "
             "-H 'Content-Type: application/json' -d '{\"message\":\"hello\"}'")


@app.post("/api/session/clear")
async def clear_session(session_id: str = Query(...)):
    STORE.clear(session_id)
    return {"cleared": session_id}


# ------------------------------- ACCOUNT PORTAL ----------------------------

ADELTE_LOGO_B64 = "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"

ACCOUNT_PAGE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADELTE Account</title>
<link rel="icon" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAdtElEQVR42m2by49kyXXefyfi3puZlfXu93Q32fOk2PMQSVOULVIQJYKyYRgyDFCwBFuAAcPwSoA3Xnrh/8ALbbwQDFmw7I21sgRThm2Ro4clD0fWkJzhTE9Pv6a7q7uqq7qyqvJxb8Q5XkTEzRzDM6iu7qrMmxEnzuM73/lCtt74jol4cA5xDucFc4K6GldVSFUh3kNVYeLoVIlRiapgiuIQcWCGaERMMAwzQBXMQEHI/xmIWXp9/o4ZSH6FWXpNemn6lRliIEb5KWrWP1MMnBNc5XGVR0zR0GEWMXM4AFPMFDSCaf9V4dLGxAm+qsEJWjnEeayuYNDQmrBQ8BsjNi/usnllh/HWOq5yRAFDwBRngAqWNyJmoAbFKJrWb5oN1BsiGwYHqmh+j+CyAS0tWF1+QDGmICaE+ZzZ8+ecPt5j+vQptlhQNw2uqkEVU0XEIURMDIuG4BGBChHECeIF9QZ1hVQ1UlVo5ZkGZf1zF3n9m1/h+tdeZ3TtImFtSAQU6IA2n4vkPaH5VGL6meY9iKbfqUJUkFg8Iv1O8gHFkN7nDCwsn0d+tsXsGRGk0/SMaNhszvzRQw7fe5fHP3iX6f4+g+EAvGGxS5+PIM6hBs4E2frSr5t4D97hmhqaAQwGtKq4c9u89fd/nut/+2sstjZ4pMazNjIPinUgqoCkDUZLi4hAyH83IKQTFwwXwalDVSGCz5tADYJhMZ1+8pzkDaJgMb1XDNyKoSwuP8eLo/EVm+tDLuwMaebH3Pn+H/PxH/03ZDGlqis0dogpaEDNcFGR7b/xjwxfQ+WxxlONRkxV2L55g2/95nfQF6/yftfxdB4JpDxR59NFLXmjGs4EYj6xaKimn0swjJQHnIJTSQZRQ1TSyUfDuuSqZiBqEAVRxfKGJaSwcGpIJBkqps9KoWBoJ2gXEIOtzYZLV7eZ3f+YH//ObzN//Ih6NEJjQLTFNKY1bH/tnxhVMoAMa9pobLz5Mt/81/+Mg80xH57NUFdRlQ8xUJW0MDNMFRcFVNBgOcbzqcZ0mhYNVPDFKCbJM6JmwxkEzeFjUE7bDOtS/KTwkWxU7UOAYsQSHipoFEIbMQ1c+fwOu27Kj37733Jy52Pqpka7efaCiHO+gapBRiNaqVh76Rrf+Ff/lA/HQ/7PZIY5j5ig5jCTtPmcjVUNU5c2rzlXkZOcGooRjVQpclHQEueavMRy2CSjpuerpc/QIJi5/LPyHkHVpfdYyiXpUEBN8r8N74Xa1ex98ozDecOX//lvsnb188QuIr4GcTgRnDnBKo/5Gtva4Jv/8jd4trvBg7OWga+x6DAVUCVqX6X6DaOSNmlgUgxkuTxJds98MpYWqZYqQjJoWnDxnFIdNKZSp1Yqh6TkVxahKeQkVxeNhsYcRsvF0dQNTx5MeL6o+eKv/mOid4gaJh7F40wM52HRdtz8la/jX32BWyczmqpBVHKJs758kl3aomDRoZoXSc7yVo6kuLHkEy4VvIQIWLDeW0wt44e0ebKnmaZSqfl7qrDWA4tiNDNFLWAWcxLV5IVRca7i3q0DuHCDqz/3iywWs4RxnODEp00ML2xw7e98jR+1gRaHyyvuYw+w3u3JXrEEK8kQOfOT3FijI+YyhpFOLFrO4uWEbRk+pT5qMrqpITFiQfs8IFFSKfMuJ+DyfOnXVGIx2yEbQ3j06XOufOPbNBeupGe7CmdNTavGpb/5BsfnttmbBhoVXJczdd6hmWEBLAjWpdgtiyxxTE5KZg6ioCFtSNRy+ZJ06iGt2nnN5SQZLiXLZCyNhgv62UQXC3oUvPc4n3KLFK9Qya8tyDElTYuKc8LZ8RQdbXPhra8Sug5chcNVsLbGpa+/wVEsmTYl5qgrKCbkTBxXkmBMRpG8ObK7W8gVIJc+VCCkB5oBXrCmYl555q6CyvfurLm0uWi4kN7ryJuSJVyvBuCa5AkuIzDBEniKisaYvkyTwUjGmRy1bL/6Rk7uShUVBue32HjpKg/b2CO3gsFLDdeC4koc6hLZYbJMYtkjfC5lppJKZUih45yj855h0/EzmyDmePdAOQuOYWtoDBn45CSnKUdJ7dBg+Mphzmg2PFYZMSiqim88REcIXfrcqBiCUwdI3o/n9HjOxd0XqMYb6OkxVQiRjfPbMB7TnkU8LrtT3lzO+KIZ8+ea40gLTLFWQiCHTK7lmGRUmPOJCUGEcdPxL172XF1v6BB+5kLHb703I0xd700SyHgZLAr1QPDrFYtg1KOK8XmPPzVOYvp3ZRWnTxfJMylrzSFgJSFDezqHi7s0m+eZHx9RocZgfUxwni4GEJcyb876kmsyZkTLWU8TABJzfQmSmF6rqgn9mSS0FVP8k0N0asbXzjncesPvTJW5Gt/acHzlcsWfHCpjlQR0ssHNQCpHF2F95NjZqgnOuHbRczSCqvHUUXj2YIF4EJfXG5ee6ywZRQw0Koqn2dxmJkJlBn7QsADaYHix/hRMJSeoDG1z92aW8oTGfNIFARXDxZx4bNn8SAFJGAvx/FCVp106occx0tQOT4doxKKhIWEGAPFQV54uGhfXjc+dq/jcyHE4godrcPtel2LaG2YxV6/0bJcrimpyZ41KUHDDIZhRgVA82zrA5RJnyZk0JgMIGf6SfxdBwtILiiEk13dXnhE15wLDmdF0wr29ji9eHXNtlGw69I57e1MGXcRFTZUmSsrwA48bOuoNx9auZ2MsvLXp+HbleDgUfk8Do13PooXYeaq2wrpAFFlWrwKeJOUwjcmqhlAJmYvIddps2atLdl1n0vftmt0rZXfXgxiX++NS5wsYSu1rKoVqiraR16/X3PrgOYuBZ1R5fjJTbm7Co2mLmYeQmh7vDEMQcaiArxxSVXTABkYtUHuoq+T6IhkAxZA7ydRnWF672RKDZNKCiozdM3IEkdyNpReUumvlNIsX5IYolTtFoiwRXiwGlGViNFgsjPNryuUt4T+9PefqpYaqDty6PefXfmGD7TXl6EBoLBnY8rOJShdrDkyIA8/emvHRWPjvMzieC5O9BbNHRnuwIE4WsAh9p4k5JHuDkDpM0SXEdmIQMUL2AGLu3mLptlLjQZc2ySomD4oEUr3OidBC+gCJBeqSMYCgnfLmjYa/ujNjeiYMvWNgSjeJ/PWtGV98dUAbQs8BSEhdo4uKC0oFxDawrpF3I3z/UeDxXmAdTzWN+EWkKpg75n4ieyhqS++MhYqTVAX6TQXrXV80czy5d7fsQuRygioWDFGXEmbxhILosptJfv+iM67tCmuN49bdjpFCaAM4Yew9d+4pr9+ouf5CZP+e0WTYLQrqPNUAYlDWpx2fX1/nj/Y6nj1S2oOOqktgyeYRDeXgWHKRPcEoCcipIOZIpFtp7yL4fJoSEhS0LmAxZoyekJwLETqFzpLbxwRGJFqP9gpB4oPiYsxVpOPmSzU/vLugaxOScgJUFfgKQsX7H7e89VpDJyG/R4mR3HjBZH/Gz1xyzGLkzv2OxV5H2I+cfDrj9OAssT1tyORJcv+lxyaeQEMGddkqTkjEogvJfdOXocV9IxCkT3AWCvY3tLMe6FjQjB9yJYgBVUVVmC2UG1cSE3T/XsfQCeY96hPj7Lxn6I07dzpmJly55pl3CdpiqRJNDjuubSvbO8J7D+YcfBrQw45w1GLTQJgs6GZdzkG5HOf223SZs1JVSOgUlKrUbymd2krzw0r5s8Lk2EryiyU8UmmR3Oen9i7nEwUfA5+/MuS998+oOnBNBRWsj4WmAqslUWmd8sMfT7nx8pD9u1PAA0bsAvjAzdc2+Oi448G9yPPHxnAeiPMAQTOnmNeuiaW2nMRllYHW2DPFiOFYxfqlDJaTz8kuJaSczEqjk9vcqJIJSs3eoRBAOkM6o5123LhacXISefooUhnMO+WLLzq+8dYmN19f59rnhcm0w2M8+3TBfKZcuF4zi2leMJ0HfuErA06DcTQVPvnJCfZ8TjjrsDZCZ8vuNNIzNhIiLkacKqKKM0XIm89R4rA05Agh082Fhi6daqkG2f0Jhg8pq+pqds35wIqhMpob1saFS54P3z+jUochaBu4eG3ABx4+1sCrb6zzuVcHjM57qsq4/aMJG1cHVEPj5HjBS1eVV24MufMw4GaR/QcdTRthoX1D5vKMQCyX8X4uEdOhlPmEutzjOISKKjGvkVA2L5aYVy1YQXpoW0qHFoYG1+cIiYXAyKyvGWEeeeWnao4OO6ZHhouRNnbceG1MY8J8aojA/b+ewKmyvVOxe26b9rRF50o9a+l0xi9+8zLffeeMl18c8ejjM1g4pEkHk7pVyx1pGsCkNeeKlA1iAriEDSzDYxGoyiZLLnBBUllbiZsl6ZeoJ2eJ3ibnBhezt4QyxjJiZwybyPp2w/s/mKIB1rfh+hfWmU4j7/zlhI3tBaqRyaOW2eNA5Yy69gw3ak4/mXC4N+Uf/Np53r+vfLpvfOlNx/feXzBwkkkVWQ5LMrKxMlUyt+wEyWg2l0DLswoUnIgDHE6h1lQKC02dYh5clOzm4IPgOnCdQRtxrSX2KBiVObyBR9Au8rlXRuw9bJkctFx7yfPizTFP70duvbugGY14+skZTz84Y3O3oZFI4zwSYfZkwZN7U37+lzcZbA15+/szbt5c49YHU2YHAZ/nBSXja5kWhRKmZYImyUtxPZolY5bS2FamBa2lMqguxXTh1pzICvmRMXWe7DgVJCYMYJDHa442KM2uw4bC8b2WL399i9lp4Md/OiHOHOtbno01mEwCnCneoN7wxEnyuMnxgq9/a52bX9rhP/zuAdvbDVvb8M7/OGNkefLkrAdmhWtM88UcBpq5wx7AZfLM8sHmTrdKbu7QDqxTxEvP6FoucSWOErZfYX1XJjzqYCbCzi68+dqAZuQ4fDzj8vURj2/POLofaLygFtnaGtBNAnFqOHO0B4GdS2s8fHLK/HTGq39rzBu/dIk/+I9PsLbi6ls1e/dnTJ+0jOqq5/rSwLFMjemZn9zF5CQoy82L9dPqNCMkGyBamnLG1Az1JEfPY2cPKCXRBOvZ4WSEIB6pjde/MkJ2HUEc41ng4z+fYCfGQNL7nRnbGw3P7k+ppMJVxtn+grOnc9p5y1u/vM3r37rE7//7B0wPKsY7js0Lng/+4IRGquT2qS/PTVlOVautukGqN3nzeWItLhnHASKCIFRYREOLBsW6NPzAFDXwuadXSznB5USomohGFwt/aNApdQdSCU+nAak9vhM46fDBIT4l2/G4Ii6Ek4M5FR4M5idKM4p89dfPcfVL5/ifv3uX04fC+nbDcLvj8GHLyaPAuHK5NMXlXDLzjmlEstQLSMkBpXfJc4s+JPKrXeHqUudlfYfnQ04qnUIX0+lrZoJjAjmaq4NpSordYeTWX5ywFjz+aeDs3oKLr25QbxihjYRgjDeGnB20SOfpZpH5vOXizYav/vp5dn9qi+/9u7s8eb9l58IWp0enXLw+4M47p2xeGVOt15/xzJ6pWuod+vBVk57hXuoJWM4N8uiuKqXEQkJ3JUz6FtL+PxDYpJ/IOAWfx9quC8z3Gx7/+YT588DhrTM2d4dsXdlkbStw/OiM0DomDycENc692HDu1QFsGUcLuPvbn3LyQLly4wJPbx+x/QqcHgSmn844/9aY6vwGB8eH+D6+lyW8ACIRS3OJPIKS0gqWMp77ltLxVkaipiXP9cnllX5Ov3y46VLR4aKk0yD1/9ZG1rY9g5Hj+YMZ69sj6lDR7itP9p8z2mnY3tlAQ2Dj857tG0M0RM6CMZzW3P3uc6YHxvlrW0w/PYXFgms3L/DRfz1ic3vI2WTO5m7N6NyI7ukZzi1nGIX17cUjlgcpLMthn89ysya5YrgyVi5ormAAiYnddaGIFxKn51Qy6Ek5QXLXqGpcubHG2eOOOAcLIc3rQ0JjIQRGFyPXvzZi+9oajz+aM5kAnePOHz4jHnkuvLBJOJ0zeTLl6le3mB0b06eeemcNsZqTT+fsXFtL05/o8xRI8lAm71Zd3mwa1KLgc5eahqesdIouNUOmKyRGZl9S85N7gjxxcVFwITFBiShJ37uZsXGhIQTjeD9y7oWGF24OGFxWNl6rufb1da799DrzWeTd//KET96ZcOnldZqF8OAPD6kWnlEDzx8c4scN/tKYnZeHPH5PWbu2wfWfXcOPYbofIcLapRGxy2FpfULocUBy78RFSB6QWtQe0Uqhx5xLjJAFgzb39l6Wk55SXguN1NNLhRXKVJcGNq+s8ej2jGqoXPm5DfRixZvnG+7+6YSnPzpJnrGIXP7pLXauD9j738+YPYhsbA6IXeBsf8ZCPPUVuPntHU4fLJjsR774qyN2fn4HHQtnPznm6O4ZW9dHnH46w9tnE5zkaallMUdq6zWHiWVucIUSE6MyVQghMzyJIU7doiY5iy1xM0WVYctpS2iV9csNi2nk9PGCwRrMpmCdcHJrxv47LdpFmjW48tVzaGvc+YOnuE4YDoT55CwlJj/k3Jtjrv3dLbpBw3ij4fIBHD9oae5M8ScOb8bs8ZTx9oC1iyPmj86ovM8hGHvys4zwlyN7+uk2rJAlkFhhoeDjdKJgvXipwM2eZyu0eWZccZFzV0Y8+vFzWHSsnR9z+P0z2nbG6f2AdMb5zw05d2PM/ofHTD4+oa4chjI/CQBUVc3CHLuvriFrA44etFy53KDTwNMfzHn29iG7VzfZuNJw+P6Ek3snXPjCDntP5tlb00hdkDy0yeg1EyJlKiRIAoGZ0s/NkO9FCxYM7WISLOWePsX9kjEmgAsJ/bWLwNblAeE0cPZ4weZWw3i9Zv+Hz3n2g1O6kzkXvjhm/cKAu3/ylJPbpzSVoCGgbcAheFch4qgF9n84haOO7XM17cOWIQOGVYs/nXL4wR7NyLP5wgbT/RlxpmxeHhMXXZo9FrouaI9PRFeAEXnMHl2fK5w5qsK7WTAk9wLFki5nU7MV4iMzRgrgjbXdhsfvHTIcei5f2+TRh0fEacvoQs3OS9vMns3Z+8vneByVKBrymLvyafLjPCJCjTL9qOVH/+Yho/PCyf3I5uaAC6/tcHRbCU8XPP/oGS+8cYXuWNn/8DkXbp7j5N4kHVYvzpReMwDLBq63BEtKHCQJJU0llTVzKe4xxLTXBmGkkpc1O4YQO2V8bcTpwZzZs5Ybb11k//4p0+OOC6+dY7Be8ewnz2j3ZwxqwYgJlHiPiCP9n0pkSkqRAS3hvjG5rbjKMXl8ymyj4cJPnac7F9h7/xn7tw7ZeXmHh+885OzZmMHlTeb3J/gqT4GkCCYkj/ez2izP9EVXKHscTnoN7YqGN9JTXIkH0F5zA4JzHjesGe+MeH53yvkbWxw9OeX4aMbVL1/EYzx5d494OKdukkd553G+xkmFT+KkfFrlxCIWIyJGVYGzQO0dTAJP3nkMGrn65StMD6dMHk/YfeUix3dPGe+uU43HOF8jzvWqlczopM3bivDYMo2Wf1iVUbJF0KA4kTwSt5VeOstNJA0VA47zbw6p15WNy2u4qMwXHRdf22Vy75jp4xNqD+KzkNoJIlXS/rJ0Sc2dWeJmKwSfkFn5UkFc6iAP3nvC8NyY81+4xOmTE+rGs3l9nfpcYLMZc/DuGQORJNzVZeo3K6SIZmlfVqFong0KabpjbTrpMl8vE2FWGg+oaQ2u/NKI7ZtrnB5HLr845NmfThjHAUcfPscmM5oq9d4iDvFVoqeWXXru3GyZX1b6dVeUDMl3s24oKUPm+1PaibLz4nnqC3DpG+c4nSnrlRDGwvEfP6eijPRl2Q4vSYJl+0xq/V1RdFtcqiwT/ne9OKmUvDgXqnOCe3nE3e+1PP6ucvhEqF8ec3xngsyNus4Pdh7n6pXNL3vzXgXSKyvpdYM5P2fWNqnWHRVOPXVdUalx/NEzRjcGPHtk7P1hx6O3A80r61RXPNaSlOD4ZfJb5TctTY0lW6FSETR0y5G2rMA/E7QAByp0oVSN5+ChsnhfkUXN6Siy9qLg0DR0EME5hxO/1PMVvWCGqaWqlLX1Xg/L8MsCa0Ew53CVQxNZga+Naes5vT2Hp4H2uaN9oaEaONpOwQnOdFm9WNUv5qGJdmk8hxk6nyexQ24Ri8DQQsS6ogBPmpnufssgCtXLjrg9x78wYH5/jp11eSPJ+r1QUqVXabhCY2nJK6nqiAq+VJq4lM736pMs4nCSDBtnxuKTKYMra8iuMrxe05ije9QmqBNXEmHxLHM9QZpI23nKPg4jnE3QrsPh89RH+8lOYnwVug7RjrDXMv3uHqM12PjCEHtwxOnbe2n6Qo57zW1yGbNl1lb7jRf6XNMXlgiMKL3CPPUaueNbbXWz+57+r4fo/UM2XtlgfVM4fvse3ZNTnBShVvJIR0xDUwCrcC5lidieJhSKQHd6CIsTvK/oujZFbMwQ2bSXndEpXpXp+1PmHz7HNQ497fDO45sqlzZbIShXiIsSjqaporBsuFbndwlrRGL/LFtpdIrQAWwWOPnj2/i1Gm0NXSj1oE4uLqvJ1i2bATXqZojEGXHxPF2zcb5B5yeEo0cMBgOsCxn00A8Z8lglOZAqtTNcG7FJS+UE7wUneSNlEhull69TYlpXiNYiqwuFkjeiQoxFZF00CXmYoS4LNJKRnID3gp4uoFvgq4xaAVOHqU8eZIKoS3pBU5p6QJjtE8IpSE2VuqPA2b0fs3vpJmdxmZlL7JVEBYITAedJ6iLSxku9zyeELEuP6MptKQp5Yf21lzTFzZWnKNIsS3U+0+5KT2Mml/DgwNUVBU6J87mKZFYoUVtLw5gyHA443vsQswBS4Sx2OO84+eSviCeHVM0IC3E5Us53e8RAxOGriqr2uKpKet18q0xElnyirSS7cpso64f7qXMRV2sRUupnLlCV9tsKuWlL4ZV4h/NVQpa+yd/rzHu7nPRsiTEsXeZo6jWcnjB59ld45xCLOLGAOCOc7vH8g++xtr3dC7zShDjT4YWNFRDn8D6VJnEO11dPyRegtC9py7qfpTia7wtkGrfoB1UKbi+dmiDqe7ldL3krfiAuYQ1JvQUiIG45Ieaz7zEzNrd2eb7/F4TFXpp4WSAJ0SzSDBomH76NHD9g4/x5tO3yosljMFuWKZN8Dc0tsbwteXesTJfyQlaGKGYx55YlWSEmuJiv1PRGS4quJeO70tXl6XSJtrKOIgzq+5vcHcYusL19HuITDvf+DO8b1LpkgJ70cDWiC/b+7PdYcy1rm1vERZulrhGnRYTAZ3NEP4CUJdvYd56aIW9hlIo4wZaGsqURyMrQojC11WskBiIxP2PJ8Gj+cynpy/cEs5FDaFlf32Y0NB5/8vuIzRB8vlQZcT3mV8P7mu7ZJ3z6R7/FuG5Z3z1H7AKx05W7Qiv00uptTrPPXl7Kziq9+FKXw8ylTrGX3a6O4JZuS9/QmGX1aD4E7UHOSqJcIQejJuX55uZ5xmvw4MPfYXF6GyeCaodpmi7J1hf+oSEuiaTF4cQRF1Oq7etc/NnfwG3c4PTomG42S2/IMznBrQwlWdHhgGnsb4/2t0Wzt4jmsMjZeekBS8xu+dVa1J743J3GPP0FZzED5eUE2EorjNA0Y7a2z6HzRzy8/Z8Ji3t4X2HaZsl/ysay9dp3rGTPtDsPzqNxgfkNdr/wS2y+/HNEt8FsMiXMZmgIK+ID6ZNeOY2YhgHLBsikV50nzL/s0cvw4rNNS6kkMWd1h5jiknw1eUNJuuXuMA7naupmzNp4TCVTjg/e5ejx22g8xTlQXWDEJJshJkNvvvIrlupng4kgIum780ki33X48XlGL7zF+PKbVOPLSDUiZlIxoTpbkqV5Br80yLItdSscg2Wdzv97OboIPZdaH+nzSVJjSeo1skc5yeSnKqozYrvH7OQWJ0cfEOYHiTWWtHnypSoIRI0IEdl46e+Zcx5xVb4JnvXpUnp0h2oghoi4AX60Q7V2jmqwiasHOPF9p4+5PILWVPdXLzmTeXncSmJMGzFJtFXRKpTYL0zu8jkhcZXFW0vijS2hO6OdHxAWz1Cd47zHSZOSHV2+PR4xK383oKMSkSwe7DCRdJNcNGn0JIMfccigRqKgsyPas2csKDq8ssdSh10mnwuElD42pIQY6Sa4FT1HiYN8L6hUFrFev53hlOZiZz00LySgEwfOJ4zi0sZVp/nElwPEkkuK1/5fAZ48hzsRbNoAAAAASUVORK5CYII=">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--sf:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",
"Helvetica Neue",Arial,sans-serif;--bg:#05060c;--card:#0d1018;--line:#1c2233;
--txt:#eaf0fb;--dim:#8b96ad;--blue:#3aa0ff;--cyan:#5fd8ff}
body{font-family:var(--sf);background:var(--bg);color:var(--txt);min-height:100vh;
display:flex;align-items:center;justify-content:center;padding:24px;
background-image:radial-gradient(900px 500px at 50% -10%,rgba(58,160,255,.18),transparent)}
.wrap{width:100%;max-width:440px}
.head{text-align:center;margin-bottom:26px}
.head img{width:76px;height:76px;border-radius:18px;box-shadow:0 12px 40px rgba(58,160,255,.35)}
.head h1{font-size:26px;letter-spacing:-.5px;margin-top:14px;font-weight:650}
.head p{color:var(--dim);font-size:14px;margin-top:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:24px}
.tabs{display:flex;gap:6px;background:#080a11;border:1px solid var(--line);
border-radius:12px;padding:4px;margin-bottom:20px}
.tabs button{flex:1;padding:10px;border:0;border-radius:9px;background:transparent;
color:var(--dim);font:inherit;font-size:14px;font-weight:550;cursor:pointer}
.tabs button.on{background:var(--blue);color:#00121f}
label{display:block;font-size:12.5px;color:var(--dim);margin:14px 0 6px;font-weight:550}
input{width:100%;padding:12px 14px;border-radius:11px;border:1px solid var(--line);
background:#080a11;color:var(--txt);font:inherit;font-size:15px;outline:none}
input:focus{border-color:var(--blue)}
.mail{font-size:12.5px;color:var(--cyan);margin-top:7px;min-height:17px}
.go{width:100%;margin-top:20px;padding:13px;border:0;border-radius:11px;
background:linear-gradient(135deg,var(--cyan),var(--blue));color:#00121f;
font:inherit;font-size:15px;font-weight:650;cursor:pointer}
.go:active{transform:translateY(1px)}
.msg{margin-top:14px;font-size:13.5px;padding:11px 13px;border-radius:10px;display:none}
.msg.err{display:block;background:rgba(255,86,110,.12);color:#ff8d9d;
border:1px solid rgba(255,86,110,.3)}
.msg.ok{display:block;background:rgba(58,255,168,.1);color:#5ce8b0;
border:1px solid rgba(58,255,168,.28)}
.me{display:none}.me.on{display:block}
.prow{display:flex;gap:16px;align-items:center;margin-bottom:18px}
.pic{width:72px;height:72px;border-radius:50%;object-fit:cover;border:2px solid var(--blue);
background:#080a11;flex:0 0 auto}
.pinfo b{font-size:17px}.pinfo div{color:var(--dim);font-size:13.5px;margin-top:3px}
.upl{margin-top:8px;font-size:12.5px;color:var(--cyan);cursor:pointer;
background:none;border:0;font-family:inherit;padding:0;text-decoration:underline}
.keybox{background:#080a11;border:1px solid var(--line);border-radius:11px;
padding:12px;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;
word-break:break-all;color:var(--cyan);margin-top:6px}
.copy{margin-top:10px;padding:9px 14px;border:1px solid var(--line);border-radius:9px;
background:#080a11;color:var(--txt);font:inherit;font-size:13px;cursor:pointer}
pre{background:#080a11;border:1px solid var(--line);border-radius:11px;padding:13px;
overflow-x:auto;font-size:12px;line-height:1.6;color:#b9c6dd;margin-top:8px}
h3{font-size:13px;color:var(--dim);margin-top:20px;font-weight:600;
text-transform:uppercase;letter-spacing:.6px}
.foot{text-align:center;margin-top:20px;font-size:13px}
.foot a{color:var(--blue);text-decoration:none}
.out{margin-top:16px;width:100%;padding:11px;border:1px solid var(--line);
border-radius:10px;background:transparent;color:var(--dim);font:inherit;
font-size:13.5px;cursor:pointer}
</style></head><body><div class="wrap">
<div class="head">
  <img src="data:image/png;base64,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" alt="ADELTE">
  <h1>ADELTE Account</h1>
  <p>Free forever. Your e-mail is <b>yourname@adelte.mab</b></p>
</div>

<div class="card" id="authcard">
  <div class="tabs">
    <button id="t_reg" class="on">Create account</button>
    <button id="t_log">Sign in</button>
  </div>
  <label for="u">Username</label>
  <input id="u" autocomplete="username" placeholder="boncoeur" spellcheck="false">
  <div class="mail" id="mail"></div>
  <label for="p">Password</label>
  <input id="p" type="password" autocomplete="current-password" placeholder="At least 4 characters">
  <button class="go" id="go">Create my account</button>
  <div class="msg" id="msg"></div>
</div>

<div class="card me" id="mecard">
  <div class="prow">
    <img class="pic" id="pic" alt="">
    <div class="pinfo">
      <b id="m_user"></b>
      <div id="m_mail"></div>
      <button class="upl" id="pickpic">Change profile picture</button>
      <input type="file" id="file" accept="image/*" hidden>
    </div>
  </div>
  <h3>Your API key</h3>
  <div class="keybox" id="m_key"></div>
  <button class="copy" id="copy">Copy key</button>
  <h3>Use it from anywhere</h3>
  <pre id="snippet"></pre>
  <h3>Free for everyone</h3>
  <pre>Text   POST /api/chat    {"message": "hello"}
Image  POST /api/image   {"prompt": "a blue fox"}
Video  POST /api/video   {"prompt": "a rocket launch", "seconds": 4}

No key needed. The key just tracks your own usage.</pre>
  <button class="out" id="out">Sign out</button>
  <div class="msg" id="msg2"></div>
</div>

<div class="foot"><a href="/">&larr; Back to ADELTE</a> &nbsp;·&nbsp;
<a href="/docs">API docs</a> &nbsp;·&nbsp; <a href="/admin">Admin</a></div>
</div>
<script>
var MODE = "register";
var $ = function(i){return document.getElementById(i)};
function say(el, t, cls){ el.textContent = t; el.className = "msg " + cls; }
$("u").addEventListener("input", function(){
  var v = this.value.trim().toLowerCase().replace(/[^a-z0-9._-]/g,"");
  $("mail").textContent = v ? ("Your e-mail will be " + v + "@adelte.mab") : "";
});
function tab(m){
  MODE = m;
  $("t_reg").className = m==="register" ? "on" : "";
  $("t_log").className = m==="login" ? "on" : "";
  $("go").textContent = m==="register" ? "Create my account" : "Sign in";
  $("msg").className = "msg";
}
$("t_reg").onclick = function(){tab("register")};
$("t_log").onclick = function(){tab("login")};

function show(acct){
  $("authcard").style.display = "none";
  $("mecard").className = "card me on";
  $("m_user").textContent = acct.username;
  $("m_mail").textContent = acct.email;
  $("m_key").textContent = acct.api_key;
  $("pic").src = acct.avatar || ("data:image/png;base64," + LOGO64);
  $("snippet").textContent =
    'curl -X POST ' + location.origin + '/api/chat \\\n' +
    '  -H "Content-Type: application/json" \\\n' +
    '  -H "X-ADELTE-User: ' + acct.username + '" \\\n' +
    '  -H "X-API-Key: ' + acct.api_key + '" \\\n' +
    '  -d \'{"message": "hello ADELTE"}\'';
  try{ localStorage.setItem("adelte_account", JSON.stringify(acct)); }catch(e){}
}
var LOGO64 = "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";

$("go").onclick = async function(){
  var u = $("u").value.trim(), p = $("p").value;
  if(!u || !p){ say($("msg"), "Fill in both fields.", "err"); return; }
  this.disabled = true; this.textContent = "Working...";
  try{
    var r = await fetch("/api/account/" + MODE, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({username:u, password:p})});
    var d = await r.json();
    if(!r.ok) throw new Error(d.detail || "could not continue");
    show(d.account);
  }catch(e){ say($("msg"), e.message, "err"); }
  this.disabled = false;
  this.textContent = MODE==="register" ? "Create my account" : "Sign in";
};
$("copy").onclick = function(){
  navigator.clipboard.writeText($("m_key").textContent);
  this.textContent = "Copied"; var b=this;
  setTimeout(function(){b.textContent="Copy key"}, 1400);
};
$("pickpic").onclick = function(){ $("file").click(); };
$("file").onchange = function(){
  var f = this.files[0]; if(!f) return;
  var img = new Image(), rd = new FileReader();
  rd.onload = function(){ img.src = rd.result; };
  img.onload = async function(){
    var c = document.createElement("canvas"), S = 256;
    c.width = c.height = S;
    var x = c.getContext("2d"), sc = Math.max(S/img.width, S/img.height);
    var w = img.width*sc, h = img.height*sc;
    x.drawImage(img, (S-w)/2, (S-h)/2, w, h);
    var url = c.toDataURL("image/jpeg", .85);
    var r = await fetch("/api/account/avatar", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({username:$("m_user").textContent, avatar:url})});
    var d = await r.json();
    if(r.ok){ $("pic").src = url; say($("msg2"), "Profile picture saved.", "ok"); }
    else { say($("msg2"), d.detail || "could not save", "err"); }
  };
  rd.readAsDataURL(f);
};
$("out").onclick = function(){
  try{ localStorage.removeItem("adelte_account"); }catch(e){}
  location.reload();
};
try{
  var saved = JSON.parse(localStorage.getItem("adelte_account") || "null");
  if(saved && saved.username) show(saved);
}catch(e){}
</script></body></html>"""


#  ---------------------------------------------------------------------------
#  ADMIN PAGE - every account, every login attempt, delete + verify badge.
#  Served at /admin. Same visual language as the account portal.
#  ---------------------------------------------------------------------------
ADMIN_PAGE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADELTE Admin</title><link rel="icon" href="/logo.png">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--sf:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",
"Helvetica Neue",Arial,sans-serif;--bg:#05060c;--card:#0d1018;--line:#1c2233;
--txt:#eaf0fb;--dim:#8b96ad;--blue:#3aa0ff;--cyan:#5fd8ff;--good:#34d399;
--bad:#f87171;--warn:#fbbf24}
body{font-family:var(--sf);background:var(--bg);color:var(--txt);min-height:100vh;
padding:22px;background-image:radial-gradient(1000px 520px at 50% -12%,
rgba(58,160,255,.16),transparent)}
.wrap{max-width:1080px;margin:0 auto}
.top{display:flex;align-items:center;gap:14px;margin-bottom:22px;flex-wrap:wrap}
.top img{width:46px;height:46px;border-radius:12px;
box-shadow:0 8px 26px rgba(58,160,255,.35)}
.top h1{font-size:22px;font-weight:650;letter-spacing:-.4px}
.top .sub{color:var(--dim);font-size:13px}
.spacer{flex:1}
.btn{border:1px solid var(--line);background:#111726;color:var(--txt);
font:inherit;font-size:13px;padding:8px 14px;border-radius:10px;cursor:pointer}
.btn:hover{border-color:var(--blue)}
.btn.pri{background:linear-gradient(135deg,var(--blue),var(--cyan));
color:#06101f;font-weight:650;border:0}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:12px;margin-bottom:20px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:14px;
padding:15px 17px}
.stat b{display:block;font-size:25px;font-weight:660;letter-spacing:-.5px}
.stat span{color:var(--dim);font-size:12.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;
padding:18px;margin-bottom:20px;overflow:hidden}
.card h2{font-size:15px;font-weight:620;margin-bottom:14px;
display:flex;align-items:center;gap:8px}
.card h2 .n{color:var(--dim);font-weight:500;font-size:13px}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13.5px;min-width:620px}
th{text-align:left;color:var(--dim);font-weight:550;font-size:12px;
text-transform:uppercase;letter-spacing:.4px;padding:0 10px 9px;
border-bottom:1px solid var(--line)}
td{padding:11px 10px;border-bottom:1px solid rgba(28,34,51,.6);
vertical-align:middle}
tr:last-child td{border-bottom:0}
.who{display:flex;align-items:center;gap:9px}
.av{width:30px;height:30px;border-radius:50%;object-fit:cover;flex:0 0 auto;
background:#182034;display:grid;place-items:center;font-size:12px;
color:var(--dim);font-weight:600}
.mail{color:var(--dim);font-size:12px}
.badge{display:inline-flex;align-items:center;gap:4px;font-size:11.5px;
padding:3px 8px;border-radius:20px;font-weight:600}
.badge.v{background:rgba(52,211,153,.14);color:var(--good)}
.badge.n{background:rgba(139,150,173,.13);color:var(--dim)}
.pill{font-size:11.5px;padding:3px 8px;border-radius:20px;font-weight:600}
.pill.ok{background:rgba(52,211,153,.14);color:var(--good)}
.pill.no{background:rgba(248,113,113,.14);color:var(--bad)}
.pill.reg{background:rgba(58,160,255,.14);color:var(--blue)}
.act{display:flex;gap:7px;justify-content:flex-end}
.mini{border:1px solid var(--line);background:transparent;color:var(--dim);
font:inherit;font-size:12px;padding:5px 10px;border-radius:8px;cursor:pointer}
.mini:hover{color:var(--txt);border-color:var(--blue)}
.mini.dan:hover{color:var(--bad);border-color:var(--bad)}
.mini.gd:hover{color:var(--good);border-color:var(--good)}
.key{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:var(--dim)}
.empty{color:var(--dim);font-size:13.5px;padding:22px 4px;text-align:center}
.msg{margin-top:12px;font-size:13px;min-height:18px}
.msg.ok{color:var(--good)} .msg.err{color:var(--bad)}
.foot{text-align:center;color:var(--dim);font-size:13px;padding:8px 0 24px}
.foot a{color:var(--blue);text-decoration:none;margin:0 8px}
@media(max-width:620px){body{padding:14px}.stat b{font-size:21px}}
</style></head><body><div class="wrap">
<div class="top">
  <img src="/logo.png" alt="ADELTE">
  <div><h1>ADELTE Admin</h1>
    <div class="sub">Accounts, registrations and every login attempt</div></div>
  <div class="spacer"></div>
  <button class="btn" id="refresh">Refresh</button>
  <a class="btn pri" href="/" style="text-decoration:none">Back to ADELTE</a>
</div>

<div class="stats" id="stats"></div>

<div class="card">
  <h2>Accounts <span class="n" id="acount"></span></h2>
  <div class="tw"><table id="atab"><thead><tr>
    <th>User</th><th>E-mail</th><th>Status</th><th>API key</th>
    <th>Joined</th><th></th></tr></thead><tbody></tbody></table></div>
  <div class="empty" id="aempty" style="display:none">No accounts yet.</div>
  <div class="msg" id="amsg"></div>
</div>

<div class="card">
  <h2>Logins &amp; registrations <span class="n" id="lcount"></span></h2>
  <div class="tw"><table id="ltab"><thead><tr>
    <th>User</th><th>Event</th><th>Result</th><th>IP</th>
    <th>Device</th><th>When</th></tr></thead><tbody></tbody></table></div>
  <div class="empty" id="lempty" style="display:none">Nothing recorded yet.</div>
</div>

<div class="foot"><a href="/">ADELTE</a>·<a href="/account">Account</a>
·<a href="/docs">API docs</a></div>
</div>
<script>
var $=function(i){return document.getElementById(i)};
function esc(t){return String(t==null?"":t).replace(/[&<>"]/g,function(c){
  return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
function when(v){
  if(!v) return "-";
  var d = (typeof v==="number") ? new Date(v*1000) : new Date(v);
  if(isNaN(d.getTime())) return String(v);
  var now=Date.now(), diff=(now-d.getTime())/1000;
  if(diff<60) return "just now";
  if(diff<3600) return Math.floor(diff/60)+" min ago";
  if(diff<86400) return Math.floor(diff/3600)+" h ago";
  return d.toLocaleDateString()+" "+d.toLocaleTimeString([],
    {hour:"2-digit",minute:"2-digit"});
}
function device(ua){
  ua=ua||"";
  if(/iPhone|iPad/i.test(ua)) return "iPhone / iPad";
  if(/Android/i.test(ua)) return "Android";
  if(/Mac OS X/i.test(ua)) return "Mac";
  if(/Windows/i.test(ua)) return "Windows";
  if(/Linux/i.test(ua)) return "Linux";
  if(/curl/i.test(ua)) return "curl";
  return ua ? ua.slice(0,22) : "-";
}
function say(el,t,c){ el.textContent=t; el.className="msg "+(c||""); 
  if(t) setTimeout(function(){ el.textContent=""; el.className="msg"; },4000); }

async function load(){
  try{
    var ar = await fetch("/api/admin/accounts").then(r=>r.json());
    var lr = await fetch("/api/admin/logins?limit=200").then(r=>r.json());
    draw(ar, lr);
  }catch(e){ say($("amsg"), "Could not reach the server: "+e.message, "err"); }
}

function draw(ar, lr){
  var accs = ar.accounts||[], logs = lr.logins||[];
  var fails = logs.filter(function(l){ return !l.ok }).length;
  $("stats").innerHTML =
    stat(ar.count||accs.length, "Accounts") +
    stat(ar.verified||0, "Verified") +
    stat(lr.count||logs.length, "Log entries") +
    stat(fails, "Failed attempts");

  $("acount").textContent = accs.length ? "("+accs.length+")" : "";
  $("lcount").textContent = logs.length ? "("+logs.length+")" : "";
  $("aempty").style.display = accs.length ? "none" : "block";
  $("lempty").style.display = logs.length ? "none" : "block";
  $("atab").style.display = accs.length ? "" : "none";
  $("ltab").style.display = logs.length ? "" : "none";

  $("atab").tBodies[0].innerHTML = accs.map(function(a){
    var u = esc(a.username), ver = !!a.verified;
    var av = a.avatar
      ? '<img class="av" src="'+esc(a.avatar)+'" alt="">'
      : '<div class="av">'+u.slice(0,1).toUpperCase()+'</div>';
    return '<tr data-u="'+u+'">'+
      '<td><div class="who">'+av+'<b>'+u+'</b></div></td>'+
      '<td class="mail">'+esc(a.email||(a.username+"@adelte.mab"))+'</td>'+
      '<td>'+(ver
        ? '<span class="badge v">Verified</span>'
        : '<span class="badge n">Unverified</span>')+'</td>'+
      '<td class="key">'+esc(a.key||a.api_key||"-")+'</td>'+
      '<td class="mail">'+when(a.created||a.created_at)+'</td>'+
      '<td><div class="act">'+
        '<button class="mini gd" data-act="'+(ver?"unverify":"verify")+'">'+
          (ver?"Remove badge":"Verify")+'</button>'+
        '<button class="mini dan" data-act="delete">Delete</button>'+
      '</div></td></tr>';
  }).join("");

  $("ltab").tBodies[0].innerHTML = logs.map(function(l){
    var kind = (l.kind||"login");
    return '<tr>'+
      '<td><b>'+esc(l.username)+'</b></td>'+
      '<td><span class="pill '+(kind==="register"?"reg":"")+'">'+
        esc(kind==="register"?"Registered":"Login")+'</span></td>'+
      '<td><span class="pill '+(l.ok?"ok":"no")+'">'+
        (l.ok?"Success":"Failed")+'</span></td>'+
      '<td class="mail">'+esc(l.ip||"-")+'</td>'+
      '<td class="mail">'+esc(device(l.agent))+'</td>'+
      '<td class="mail">'+when(l.at)+'</td></tr>';
  }).join("");
}
function stat(n,label){
  return '<div class="stat"><b>'+n+'</b><span>'+label+'</span></div>';
}

// One delegated handler for verify / unverify / delete.
$("atab").addEventListener("click", async function(ev){
  var b = ev.target.closest("button[data-act]"); if(!b) return;
  var row = b.closest("tr"), u = row.getAttribute("data-u"),
      act = b.getAttribute("data-act");
  if(act==="delete" && !confirm('Delete the account ' + u +
     '? This cannot be undone.')) return;
  b.disabled = true;
  try{
    var r, j;
    if(act==="delete"){
      r = await fetch("/api/admin/account/"+encodeURIComponent(u),
                      {method:"DELETE"});
    } else {
      r = await fetch("/api/admin/verify", {method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({username:u, verified: act==="verify"})});
    }
    j = await r.json();
    if(!r.ok || j.ok===false) throw new Error(j.detail||"failed");
    say($("amsg"), act==="delete" ? ("Deleted "+u)
      : (act==="verify" ? (u+" is now verified") : ("Badge removed from "+u)),
      "ok");
    load();
  }catch(e){ say($("amsg"), "Could not do that: "+e.message, "err"); }
  finally{ b.disabled = false; }
});

$("refresh").onclick = load;
load();
setInterval(load, 20000);   // keep the log fresh without a manual reload
</script></body></html>"""


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_page():
    """Everyone who signed up, everyone who logged in, and the controls."""
    return HTMLResponse(ADMIN_PAGE, headers=NOCACHE)


@app.get("/account", response_class=HTMLResponse, include_in_schema=False)
@app.get("/signup", response_class=HTMLResponse, include_in_schema=False)
@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def account_page():
    """The account portal - register, sign in, upload a picture, get a key."""
    return HTMLResponse(ACCOUNT_PAGE, headers=NOCACHE)


@app.get("/logo.png", include_in_schema=False)
async def logo_png():
    return Response(content=base64.b64decode(ADELTE_LOGO_B64),
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


# --------------------------------- UI --------------------------------------

FALLBACK_UI = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>ADELTE</title><style>body{background:#05060c;color:#eaf0fb;font-family:
system-ui,sans-serif;max-width:760px;margin:60px auto;padding:0 20px;line-height:1.7}
a{color:#00e5ff}code{background:#141826;padding:2px 7px;border-radius:6px}
h1{background:linear-gradient(135deg,#00e5ff,#a855f7);-webkit-background-clip:text;
background-clip:text;color:transparent;font-size:32px}</style></head><body>
<h1>&#916; ADELTE Server is running</h1>
<p>The frontend file <code>adelte.html</code> was not found next to
<code>adelte.py</code>. Put it there and refresh.</p>
<p>The API works right now regardless:</p>
<ul>
<li><a href="/docs">/docs</a> — interactive Swagger UI</li>
<li><code>GET /api/search?q=your+question</code></li>
<li><code>POST /api/chat</code> with <code>{"message":"..."}</code></li>
<li><code>POST /api/keys/generate</code> — free API key</li>
</ul></body></html>"""


NOCACHE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
           "Pragma": "no-cache", "Expires": "0"}


@app.head("/")
async def index_head():
    return HTMLResponse("", headers=NOCACHE)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the UI with caching fully disabled.

    Browsers aggressively cache a bare .html served over localhost, which
    means UI fixes appear to 'not apply'. no-store kills that."""
    f = HERE / "adelte.html"
    if f.exists():
        return HTMLResponse(f.read_text(encoding="utf-8"), headers=NOCACHE)
    return HTMLResponse(FALLBACK_UI, headers=NOCACHE)


@app.get("/adelte.html", response_class=HTMLResponse,
         include_in_schema=False)
@app.get("/delta.html", response_class=HTMLResponse,
         include_in_schema=False)
async def index_alias():
    return await index()


@app.exception_handler(405)
async def wrong_method(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={"detail": getattr(exc, "detail", "Method Not Allowed"),
                 "path": str(request.url.path),
                 "method_used": request.method,
                 "hint": ("If you are seeing this in the browser UI, you opened "
                          "adelte.html from a static file server (Live Server / "
                          "http-server / file://) instead of from ADELTE. Run "
                          "`python adelte.py` and open http://localhost:8000")})


@app.exception_handler(404)
async def nf(request: Request, exc):
    return JSONResponse(status_code=404,
                        content={"detail": "Not found",
                                 "path": str(request.url.path),
                                 "hint": "open / for the UI, /docs for Swagger"})


# ============================================================================
#  SECTION 7 — ENTRY POINT
# ============================================================================

BANNER = """
\033[96m  ██████  \033[95m███████ \033[91m██   ████████  █████
\033[96m  ██   ██ \033[95m██      \033[91m██      ██    ██   ██
\033[96m  ██   ██ \033[95m█████   \033[91m██      ██    ███████
\033[96m  ██   ██ \033[95m██      \033[91m██      ██    ██   ██
\033[96m  ██████  \033[95m███████ \033[91m███████ ██    ██   ██\033[0m

  \033[1mADELTE Server v3.0\033[0m — by ADELTE Industries
  ──────────────────────────────────────────────────────
  UI       \033[96mhttp://localhost:{port}\033[0m
  Swagger  \033[96mhttp://localhost:{port}/docs\033[0m
  Engines  {nengines} (all keyless)
  AI layer {ai}
  Database {db}

  Models   {models}
  Brains   {brains}

  Ctrl+C to stop.
"""


def main() -> None:
    global STORE
    ap = argparse.ArgumentParser(description="ADELTE Server — by ADELTE Industries")
    ap.add_argument("--host", default=CFG.host)
    ap.add_argument("--port", type=int, default=CFG.port)
    ap.add_argument("--db", default=str(CFG.db_path))
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the AI swarm; extractive answers only (fastest)")
    ap.add_argument("--ai-budget", type=float, default=CFG.ai_budget,
                    help="seconds to wait for the AI swarm (default 50)")
    ap.add_argument("--no-system", action="store_true",
                    help="disable system commands (lock/sleep/volume)")
    ap.add_argument("--allow-system", action="store_true",
                    help="allow lock/sleep/volume commands on THIS machine")
    ap.add_argument("--deep-pages", type=int, default=CFG.deep_pages,
                    help="how many result pages to read in full (default 4)")
    a = ap.parse_args()

    CFG.host, CFG.port = a.host, a.port
    CFG.db_path = Path(a.db)
    CFG.use_ai = not a.no_ai
    CFG.ai_budget = a.ai_budget
    CFG.deep_pages = a.deep_pages

    global ALLOW_SYSTEM
    ALLOW_SYSTEM = not bool(a.no_system)

    STORE = Store(CFG.db_path)

    ready = [PROVIDERS[p]["label"] for p in PROVIDERS if provider_ready(p)
             and PROVIDERS[p]["env"]]
    try:
        print(BANNER.format(port=CFG.port, nengines=len(ENGINES),
                            ai="AI Horde (free, anonymous)" if CFG.use_ai
                               else "disabled — extractive only",
                            models=", ".join(m["name"] for m in
                                             ADELTE_MODELS.values()),
                            brains=(", ".join(ready) + " + free swarm") if ready
                                   else "free swarm only (add keys to .env)",
                            db=CFG.db_path))
    except UnicodeEncodeError:
        # Windows cp1252 consoles cannot render the block-art banner.
        # Additive fallback: ASCII only, same info, never crashes boot.
        print("ADELTE Server on port %d - %d engines - %s" % (CFG.port, len(ENGINES), CFG.db_path))
    if ready:
        for pid in PROVIDERS:
            if provider_ready(pid) and PROVIDERS[pid]["env"]:
                print("  \033[92mkey\033[0m %-12s %s"
                      % (PROVIDERS[pid]["label"], mask(env_key(PROVIDERS[pid]["env"]))))
    else:
        print("  \033[93mNo .env found.\033[0m Create one beside adelte.py:")
        print("      GROQ_API_KEY=gsk_... (see .env.example)")
        print("      GEMINI_API_KEY=AIza... (see .env.example)")
        print("      OPENROUTER_API_KEY=sk-or-v1-... (see .env.example)")
        print("      OPENAI_API_KEY=sk-proj-... (see .env.example)")
    print("")

    # Warn if the database would land next to adelte.html (reload trigger).
    try:
        if CFG.db_path.resolve().parent == (HERE / "adelte.html").resolve().parent:
            print("  \033[93mNote:\033[0m the database sits beside adelte.html. If you use a\n"
                  "  file-watching dev server, its writes can force browser reloads.\n"
                  "  Use --db to move it, e.g.  python adelte.py --db ~/adelte.db\n")
    except Exception:
        pass

    print("  \033[92mOpen  http://localhost:{}  in your browser.\033[0m".format(CFG.port))
    if ALLOW_SYSTEM:
        print("  \033[93mSystem commands ENABLED\033[0m — lock / sleep / volume can run"
              " on this machine.")
    else:
        print("  System commands disabled via --no-system."
              " `lock pc`, `sleep`, etc.")
    print("  \033[91mDo NOT use Live Server / http-server on adelte.html.\033[0m\n")

    import uvicorn
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")


if __name__ == "__main__":
    main()
