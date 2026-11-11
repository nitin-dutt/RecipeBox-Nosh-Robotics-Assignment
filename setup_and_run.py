#!/usr/bin/env python3
"""One-command setup and launcher for the Recipe Box chatbot.

    python setup_and_run.py            (Windows: py setup_and_run.py)

It is safe to run again: every step is skipped when it is already done.

Steps
  1. check Python 3.11+ and Node.js 18+
  2. create .venv and install requirements.txt
  3. download the Recipe Box dataset (three JSON files) if Dataset/ does not have them
  4. build the database and the search index (one time, roughly 15-25 minutes on a laptop CPU)
  5. install the web UI packages
  6. create .env from .env.example and ask for the Gemini API key (optional)
  7. start the API (port 8000) and the web UI (port 5173) and open the browser

Only the Python standard library is used here, so it runs before anything is installed.
Options: --no-start (set up only), --skip-key (do not ask for a key), --reset-key (ask again).
"""
from __future__ import annotations

import getpass
import hashlib
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WIN = os.name == "nt"
VENV = ROOT / ".venv"
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
DATASET_DIR = ROOT / "Dataset"
DATASET_FILES = ["recipes_raw_nosource_ar.json", "recipes_raw_nosource_epi.json", "recipes_raw_nosource_fn.json"]
DATASET_URL = "https://eightportions.com/recipes_raw.zip"
DB = ROOT / "data" / "recipes.duckdb"
INGEST_DONE = ROOT / "data" / ".ingestion_done"
INDEX_META = ROOT / "data" / "index" / "meta.json"
API_PORT, WEB_PORT = 8000, 5173
STEP = 0


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(title: str) -> None:
    global STEP
    STEP += 1
    say(f"\n[{STEP}] {title}")


def fail(msg: str) -> None:
    say(f"\nERROR: {msg}")
    sys.exit(1)


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    """Run a command, streaming its output; stop the whole setup if it fails."""
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    if proc.returncode != 0:
        fail(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- 1. prerequisites
def check_prerequisites() -> str:
    step("Checking prerequisites")
    if sys.version_info < (3, 11):
        fail(f"Python 3.11 or newer is required (this is {sys.version.split()[0]}). Install it from python.org.")
    say(f"  Python {sys.version.split()[0]} ok")
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        fail("Node.js 18+ (with npm) is required for the web UI. Install it from https://nodejs.org and run this again.")
    out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
    major = int(re.sub(r"[^0-9.]", "", out).split(".")[0] or 0)
    if major < 18:
        fail(f"Node.js 18+ is required (found {out}).")
    say(f"  Node {out} ok")
    return npm


# --------------------------------------------------------------------------- 2. python packages
def install_python_packages() -> None:
    step("Python environment and packages")
    if not VENV_PY.exists():
        say("  creating .venv ...")
        run([sys.executable, "-m", "venv", str(VENV)])
    if subprocess.run([str(VENV_PY), "-m", "pip", "--version"], capture_output=True).returncode != 0:
        say("  this environment has no pip (for example one made by uv); adding it ...")
        run([str(VENV_PY), "-m", "ensurepip", "--upgrade"])
    stamp = VENV / ".requirements.sha256"
    want = file_hash(ROOT / "requirements.txt") + file_hash(ROOT / "pyproject.toml")
    if stamp.exists() and stamp.read_text() == want:
        say("  packages already installed")
        return
    say("  installing packages (a few minutes the first time) ...")
    run([str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements.txt"])
    run([str(VENV_PY), "-m", "pip", "install", "--disable-pip-version-check", "-q", "--no-deps", "-e", "."])
    stamp.write_text(want)


# --------------------------------------------------------------------------- 3. dataset
def ensure_dataset() -> None:
    step("Dataset")
    DATASET_DIR.mkdir(exist_ok=True)
    missing = [n for n in DATASET_FILES if not (DATASET_DIR / n).exists()]
    if not missing:
        say("  the three dataset files are already in Dataset/")
        return
    say(f"  downloading the Recipe Box dataset (about 53 MB) from {DATASET_URL}")
    tmp = Path(tempfile.mkdtemp(prefix="recipebox_"))
    zpath = tmp / "recipes_raw.zip"
    try:
        req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "recipebot-setup"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(zpath, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        say()
        with zipfile.ZipFile(zpath) as z:
            names = {Path(i.filename).name: i for i in z.infolist()}
            for n in missing:
                if n not in names:
                    fail(f"{n} is not inside the downloaded archive")
                with z.open(names[n]) as src, open(DATASET_DIR / n, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        say("  extracted to Dataset/")
    except Exception as exc:  # network or archive problem: explain the manual route
        fail(f"could not download the dataset ({exc}).\n  Download recipes_raw.zip from https://eightportions.com/datasets/Recipes/ "
             f"and put the three JSON files in {DATASET_DIR}, then run this again.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- 4. build
def build_data() -> None:
    step("Database and search index")
    if DB.exists() and INGEST_DONE.exists():
        say("  database already built")
    else:
        say("  cleaning the data, loading DuckDB and parsing ingredients (a few minutes) ...")
        DB.parent.mkdir(exist_ok=True)
        run([str(VENV_PY), "-m", "recipebot.ingestion"])
        INGEST_DONE.write_text("ok\n")
    if INDEX_META.exists():
        say("  search index already built")
    else:
        say("  building the search index; it downloads a small embedding model and takes about 10-20 minutes.")
        say("  It resumes if you stop it, so you can press Ctrl+C and run this file again.")
        run([str(VENV_PY), "-m", "recipebot.retrieval.build"])


# --------------------------------------------------------------------------- 5. web packages
def install_web_packages(npm: str) -> None:
    step("Web UI packages")
    web = ROOT / "web"
    stamp = web / "node_modules" / ".package-lock.sha256"
    want = file_hash(web / "package-lock.json")
    if stamp.exists() and stamp.read_text() == want:
        say("  already installed")
        return
    run([npm, "install", "--no-audit", "--no-fund"], cwd=web)
    stamp.write_text(want)


# --------------------------------------------------------------------------- 6. .env and API key
def read_env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*{re.escape(key)}\s*=\s*(.*)$", line)
        if m:
            return m.group(1).strip().strip("\"'")
    return ""


def write_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"\s*{re.escape(key)}\s*=", line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_env(skip_key: bool, reset_key: bool) -> None:
    step("Configuration (.env)")
    env, example = ROOT / ".env", ROOT / ".env.example"
    if not env.exists():
        shutil.copyfile(example, env)
        say("  created .env from .env.example")
    have = read_env_value(env, "GEMINI_API_KEY")
    if have and not reset_key:
        say("  GEMINI_API_KEY is already set")
        return
    if skip_key or not sys.stdin.isatty():
        say("  no API key set: the bot will use its rule-based planner (set GEMINI_API_KEY in .env later for the full experience)")
        return
    say("  The chatbot understands follow-ups and unusual phrasing much better with a Gemini model.")
    say("  A free key is enough: https://aistudio.google.com/apikey")
    say("  Paste the key below (it will not be shown), or press Enter to skip.")
    key = getpass.getpass("  GEMINI_API_KEY: ").strip()
    if key:
        write_env_value(env, "GEMINI_API_KEY", key)
        say("  saved to .env (this file is git-ignored)")
    else:
        say("  skipped: using the rule-based planner")


# --------------------------------------------------------------------------- 7. start
def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port(port: int, proc: subprocess.Popen, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            return False
        if not port_free(port):
            return True
        time.sleep(0.5)
    return False


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_servers(npm: str) -> None:
    step("Starting the servers")
    for port, name in ((API_PORT, "API"), (WEB_PORT, "web UI")):
        if not port_free(port):
            fail(f"port {port} (the {name}) is already in use. Close what is using it and run this again.")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WIN else 0
    api = subprocess.Popen([str(VENV_PY), "-m", "uvicorn", "recipebot.api.app:app", "--port", str(API_PORT)], cwd=ROOT,
                           creationflags=flags)
    procs = [api]
    try:
        say(f"  starting the API on port {API_PORT} (loading the index takes a few seconds) ...")
        if not wait_for_port(API_PORT, api, 180):
            fail("the API did not start; see the messages above.")
        web = subprocess.Popen([npm, "run", "dev", "--", "--host", "127.0.0.1", "--strictPort"], cwd=ROOT / "web",
                               creationflags=flags)
        procs.append(web)
        if not wait_for_port(WEB_PORT, web, 90):
            fail("the web UI did not start; see the messages above.")
        url = f"http://127.0.0.1:{WEB_PORT}"
        say(f"\n  Ready: {url}\n  Press Ctrl+C here to stop both servers.\n")
        webbrowser.open(url)
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        say("\n  a server exited; stopping the other one.")
    except KeyboardInterrupt:
        say("\n  stopping ...")
    finally:
        for p in procs:
            stop(p)


def main() -> None:
    args = set(sys.argv[1:])
    os.chdir(ROOT)
    say("Recipe Box chatbot: setup and run")
    npm = check_prerequisites()
    install_python_packages()
    ensure_dataset()
    build_data()
    install_web_packages(npm)
    configure_env("--skip-key" in args, "--reset-key" in args)
    if "--no-start" in args:
        say("\nSetup complete. Start later with: python setup_and_run.py")
        return
    start_servers(npm)


if __name__ == "__main__":
    main()
