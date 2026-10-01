#!/usr/bin/env python3
"""
===============================================================================
 CARGOLIS DESKTOP — your Siri, outside the browser
===============================================================================
 A system-wide voice companion for ADELTE. It lives in your system tray,
 listens on a global hotkey, sees your screen, searches the web and talks
 back — whether or not the browser is open.

 QUICK START (Windows)
     pip install -r requirements-desktop.txt
     python adelte.py                  # terminal 1: the brain (or use hosted URL)
     python cargolis.py                # terminal 2: the voice

 USAGE
     * Press Ctrl+Alt+A anywhere  -> Cargolis listens, answers out loud
     * Type in the console at any time (mic optional, voice optional)
     * Say "see my screen"         -> screenshots, reads it, explains/fixes it
     * Say "search ..."            -> 16-engine web search, top answer spoken
     * Say "open ..." / "lock ..." -> runs on THIS pc via the local server
     * Say "stop listening" / Ctrl+C to quit

 CONFIG
     ADELTE_URL env or --url flag. Default http://localhost:8000 (your own PC,
     so "lock pc" etc. act on YOUR machine). Point it at Render/Vercel only
     for chat/search — PC commands then act on the server, not your PC.

 Only the standard library is REQUIRED. Microphone, voice, hotkey and tray
 are optional extras that degrade gracefully to typed console use.
===============================================================================
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
import webbrowser

ADELTE_URL = os.environ.get("ADELTE_URL", "http://localhost:8000").rstrip("/")
SESSION_FILE = os.path.join(os.path.expanduser("~"), ".cargolis_session")


# ---------------------------------------------------------------------------
# Tiny HTTP client (stdlib only — no dependency can ever break this)
# ---------------------------------------------------------------------------

def api(method: str, path: str, body: dict | None = None,
        timeout: float = 60.0) -> dict:
    url = ADELTE_URL + path
    data = json.dumps(body or {}).encode() if method != "GET" else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return {"ok": False, "error": "%s: %s" % (type(e).__name__, str(e)[:160])}


def load_session() -> str:
    try:
        with open(SESSION_FILE, encoding="utf-8") as f:
            sid = f.read().strip()
            if sid:
                return sid
    except Exception:
        pass
    sid = str(uuid.uuid4())
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            f.write(sid)
    except Exception:
        pass
    return sid


SESSION_ID = load_session()


# ---------------------------------------------------------------------------
# Voice out (offline) — pyttsx3 if present, print otherwise
# ---------------------------------------------------------------------------

_SPEAKER = None
_SPEAK_LOCK = threading.Lock()


def speak(text: str) -> None:
    """Speak aloud when possible; always print so nothing is ever silent."""
    short = text if len(text) <= 900 else text[:900] + "… (full answer in console)"
    print("\n[Cargolis] " + text + "\n")
    global _SPEAKER
    try:
        if _SPEAKER is None:
            import pyttsx3  # type: ignore
            _SPEAKER = pyttsx3.init()
            _SPEAKER.setProperty("rate", 178)
        with _SPEAK_LOCK:
            _SPEAKER.say(short)
            _SPEAKER.runAndWait()
    except Exception:
        pass  # no voice engine — console output already shown


# ---------------------------------------------------------------------------
# Voice in — microphone if present, keyboard otherwise
# ---------------------------------------------------------------------------

def listen(timeout: float = 8.0) -> str:
    """Listen on the mic once; fall back to typed input."""
    try:
        import speech_recognition as sr  # type: ignore
        r = sr.Recognizer()
        with sr.Microphone() as src:
            print("[Cargolis] Listening… (speak now)")
            r.adjust_for_ambient_noise(src, duration=0.4)
            audio = r.listen(src, timeout=timeout, phrase_time_limit=25)
        try:
            return r.recognize_google(audio)
        except Exception:
            print("[Cargolis] Didn't catch that — type it instead.")
    except Exception as e:
        if "speech_recognition" in str(type(e).__name__).lower() or "No module" in str(e):
            pass
        else:
            print("[Cargolis] Mic unavailable (%s) — type instead." % type(e).__name__)
    try:
        return input("[You] ").strip()
    except (EOFError, KeyboardInterrupt):
        return ""


# ---------------------------------------------------------------------------
# Eyes — screenshot this PC and let the server read it
# ---------------------------------------------------------------------------

def screenshot_b64() -> str:
    """PNG bytes of the primary monitor, base64-encoded. '' if unavailable."""
    try:
        import mss  # type: ignore
        with mss.mss() as sct:
            shot = sct.grab(sct.monitors[1])
            from PIL import Image  # type: ignore
            img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass
    try:
        from PIL import ImageGrab  # type: ignore
        buf = io.BytesIO()
        ImageGrab.grab().save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


QUICK_OPEN = {
    "youtube": "https://www.youtube.com", "google": "https://www.google.com",
    "gmail": "https://mail.google.com", "github": "https://github.com",
    "maps": "https://maps.google.com", "drive": "https://drive.google.com",
    "whatsapp": "https://web.whatsapp.com", "netflix": "https://www.netflix.com",
    "spotify": "https://open.spotify.com", "news": "https://news.google.com",
}


def handle(text: str) -> bool:
    """One turn. Returns False when the user asked to quit."""
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    if low in ("quit", "exit", "stop listening", "goodbye siri", "bye"):
        speak("Later. I'll be in the tray when you need me.")
        return False

    # --- eyes: see this PC ---
    if any(k in low for k in ("see my screen", "look at my screen", "what is on my screen",
                              "read my screen", "see screen", "screenshot")):
        speak("Looking at your screen now.")
        img = screenshot_b64()
        if not img:
            speak("I couldn't capture the screen. Install requirements-desktop for eyes: pip install mss pillow.")
            return True
        res = api("POST", "/api/cargolis/analyse",
                  {"image": img, "task": "auto",
                   "question": t if len(t) > 20 else ""}, timeout=120.0)
        speak(res.get("answer", "My vision brain is busy — try again in a moment."))
        return True

    # --- fast local lane: open sites on THIS pc without a round trip ---
    if low.startswith(("open ", "launch ", "go to ")):
        target = low.split(None, 1)[1].strip()
        url = QUICK_OPEN.get(target.replace(".com", "").replace("the ", "").strip())
        if url:
            webbrowser.open(url)
            speak("Opening %s on your PC." % target)
            return True

    # --- search lane ---
    if low.startswith(("search ", "google ", "look up ", "find ")):
        q = t.split(None, 1)[1] if " " in t else t
        res = api("GET", "/api/search?" + urllib.parse.urlencode({"q": q, "deep": False}))
        hits = res.get("results", []) if isinstance(res, dict) else []
        if not hits:
            speak("Nothing found for %s." % q)
            return True
        top = hits[0]
        speak("Top result: %(title)s. %(snippet)s" % {
            "title": top.get("title", ""), "snippet": top.get("snippet", "")})
        print("    -> " + top.get("url", ""))
        return True

    # --- everything else: the cargolis brain (chat keeps memory via session) ---
    res = api("POST", "/api/chat",
              {"message": t, "session_id": SESSION_ID, "model": "adelte-cargolis"},
              timeout=120.0)
    answer = (res or {}).get("answer", "")
    if not answer:
        speak("The brain didn't answer — is the server running at %s? (%s)"
              % (ADELTE_URL, (res or {}).get("error", "unknown error")))
        return True
    speak(answer)
    srcs = (res or {}).get("sources", []) or []
    for s in srcs[:5]:
        print("    [%s](%s)" % (s.get("title", "")[:80], s.get("url", "")))
    return True


def talk_hotkey_loop() -> None:
    """Background: Ctrl+Alt+A anywhere → voice turn. No-op without pynput."""
    try:
        from pynput import keyboard  # type: ignore
    except Exception:
        print("[Cargolis] Hotkey off (pip install pynput for Ctrl+Alt+A). Console typing works.")
        return
    combo = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.KeyCode.from_char("a")}
    held: set = set()

    def on_press(k):
        if k in combo:
            held.add(k)
        if combo <= held:
            held.clear()
            print("\n[Cargolis] Hotkey — talk now.")
            handle(listen())

    def on_release(k):
        held.discard(k)

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as L:
            L.join()
    except Exception:
        pass


def tray_loop() -> None:
    """System tray icon with Talk / See screen / Quit. Console if unavailable."""
    try:
        import pystray  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        print("[Cargolis] Tray off (pip install pystray pillow for a tray icon).")
        return
    img = Image.new("RGB", (64, 64), (20, 22, 40))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill=(168, 85, 247))
    d.ellipse([22, 22, 42, 42], fill=(0, 229, 255))

    def _talk(icon, item):
        threading.Thread(target=lambda: handle(listen()), daemon=True).start()

    def _see(icon, item):
        threading.Thread(target=lambda: handle("see my screen"), daemon=True).start()

    menu = pystray.Menu(
        pystray.MenuItem("Talk (or press Ctrl+Alt+A)", _talk, default=True),
        pystray.MenuItem("See my screen", _see),
        pystray.MenuItem("Quit", lambda i, m: (i.stop(), sys.exit(0))),
    )
    try:
        pystray.Icon("cargolis", img, "Cargolis — always listening", menu).run()
    except Exception:
        pass


def main() -> None:
    global ADELTE_URL
    ap = argparse.ArgumentParser(description="Cargolis Desktop — Siri outside the browser")
    ap.add_argument("--url", default=ADELTE_URL, help="ADELTE server URL")
    ap.add_argument("--no-tray", action="store_true")
    ap.add_argument("--no-hotkey", action="store_true")
    ap.add_argument("--say", default="", help="speak one line and exit")
    a = ap.parse_args()
    ADELTE_URL = a.url.rstrip("/")

    try:
        h = api("GET", "/api/health", timeout=8)
        if h.get("status") == "ok":
            print("[Cargolis] Connected to %s (%s, %d engines)."
                  % (ADELTE_URL, h.get("version"), len(h.get("engines", []))))
        else:
            print("[Cargolis] Server answered oddly — continuing anyway.")
    except Exception:
        pass

    if a.say:
        handle(a.say)
        return

    if not a.no_hotkey:
        threading.Thread(target=talk_hotkey_loop, daemon=True).start()
    if not a.no_tray:
        threading.Thread(target=tray_loop, daemon=True).start()

    print("[Cargolis] Ready. Type, or press Ctrl+Alt+A anywhere to talk.")
    print("            Try: 'see my screen' · 'search quantum chips' · 'open youtube'")
    while True:
        try:
            if not handle(listen(timeout=3600)):
                break
        except KeyboardInterrupt:
            speak("Going quiet. Tray icon still here if you need me.")
            break


if __name__ == "__main__":
    main()
