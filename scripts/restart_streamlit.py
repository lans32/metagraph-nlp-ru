"""Перезапуск Streamlit-приложения с очисткой кэша.

Останавливает запущенный Streamlit на указанном порту, чистит дисковый
кэш (`~/.streamlit/cache`, `.streamlit/cache`) и `streamlit cache clear`,
затем стартует новый процесс в фоне и ждёт готовности порта.

Запуск (из корня проекта):

    .venv/Scripts/python.exe scripts/restart_streamlit.py
    # с другим портом / приложением:
    .venv/Scripts/python.exe scripts/restart_streamlit.py --port 8502
    .venv/Scripts/python.exe scripts/restart_streamlit.py --app src/metagraph_nlp/web/app.py
    # без очистки кэша:
    .venv/Scripts/python.exe scripts/restart_streamlit.py --no-clear-cache

Cross-platform: использует ОС-специфичные команды для поиска PID
(netstat на Windows, lsof на Unix). Не требует psutil.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_PORT = 8501
DEFAULT_APP = "src/metagraph_nlp/web/app.py"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _streamlit_executable() -> Path:
    """Путь к streamlit в .venv (Windows: .exe, Unix: без расширения)."""
    if os.name == "nt":
        return PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "streamlit"


def _find_pid_on_port(port: int) -> int | None:
    """Найти PID процесса, слушающего TCP-порт. None если порт свободен."""
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"], text=True, errors="ignore"
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        for line in out.splitlines():
            parts = line.split()
            # netstat -ano строки: Proto  Local  Foreign  State  PID
            if len(parts) < 5 or parts[0] != "TCP":
                continue
            local = parts[1]
            state = parts[3]
            if state != "LISTENING":
                continue
            if local.endswith(f":{port}"):
                try:
                    return int(parts[4])
                except ValueError:
                    continue
        return None

    # Unix: lsof -ti tcp:<port> -sTCP:LISTEN
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], text=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    out = out.strip()
    if not out:
        return None
    return int(out.splitlines()[0])


def _kill_pid(pid: int, *, timeout: float = 5.0) -> bool:
    """Завершить процесс. Сначала мягко (SIGTERM/Ctrl-C), потом жёстко.

    Возвращает True, если процесс пропал.
    """
    print(f"[restart] stopping streamlit (pid={pid})…", flush=True)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T"],
                check=False, capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[restart] soft kill failed: {e}", flush=True)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.2)

    # Жёстко
    print(f"[restart] forcing kill (pid={pid})…", flush=True)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False, capture_output=True,
            )
        else:
            os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    time.sleep(0.5)
    return not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True,
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _clear_streamlit_cache(streamlit_exe: Path) -> None:
    """Очистить дисковый кэш Streamlit и вызвать `streamlit cache clear`."""
    cache_dirs = [
        Path.home() / ".streamlit" / "cache",
        PROJECT_ROOT / ".streamlit" / "cache",
    ]
    for d in cache_dirs:
        if d.exists():
            print(f"[restart] removing {d}", flush=True)
            shutil.rmtree(d, ignore_errors=True)
        else:
            print(f"[restart] (none) {d}", flush=True)

    print("[restart] streamlit cache clear", flush=True)
    subprocess.run(
        [str(streamlit_exe), "cache", "clear"],
        check=False, capture_output=True,
    )


def _wait_port(port: int, *, timeout: float = 30.0) -> bool:
    """Ждать, пока на порт можно установить TCP-соединение."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _start_streamlit(streamlit_exe: Path, app_path: Path, port: int) -> int:
    """Запустить streamlit в фоне (detached). Возвращает PID."""
    log_path = PROJECT_ROOT / "artifacts" / "streamlit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")

    cmd = [
        str(streamlit_exe), "run", str(app_path),
        "--server.headless", "true",
        "--server.port", str(port),
    ]
    print(f"[restart] starting: {' '.join(cmd)}", flush=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=creationflags, close_fds=True,
        )
    else:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    print(f"[restart] streamlit pid={proc.pid}, log={log_path}", flush=True)
    return proc.pid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP-порт Streamlit (default: {DEFAULT_PORT}).")
    parser.add_argument("--app", type=str, default=DEFAULT_APP,
                        help=f"Путь к приложению (default: {DEFAULT_APP}).")
    parser.add_argument("--no-clear-cache", action="store_true",
                        help="Пропустить очистку кэша.")
    parser.add_argument("--wait-timeout", type=float, default=30.0,
                        help="Сколько секунд ждать готовности порта.")
    args = parser.parse_args()

    streamlit_exe = _streamlit_executable()
    if not streamlit_exe.exists():
        print(f"[restart] ERROR: streamlit not found at {streamlit_exe}",
              file=sys.stderr)
        return 1

    app_path = (PROJECT_ROOT / args.app).resolve()
    if not app_path.exists():
        print(f"[restart] ERROR: app not found at {app_path}", file=sys.stderr)
        return 1

    pid = _find_pid_on_port(args.port)
    if pid is None:
        print(f"[restart] no process on port {args.port}", flush=True)
    else:
        if not _kill_pid(pid):
            print(f"[restart] WARNING: pid {pid} still alive, continuing",
                  flush=True)

    if not args.no_clear_cache:
        _clear_streamlit_cache(streamlit_exe)
    else:
        print("[restart] skipping cache clear (--no-clear-cache)", flush=True)

    _start_streamlit(streamlit_exe, app_path, args.port)

    if _wait_port(args.port, timeout=args.wait_timeout):
        print(f"[restart] ready at http://localhost:{args.port}", flush=True)
        return 0
    print(f"[restart] WARNING: port {args.port} did not open within "
          f"{args.wait_timeout:.0f}s — see artifacts/streamlit.log", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
