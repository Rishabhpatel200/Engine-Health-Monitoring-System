#!/usr/bin/env python3
"""Launch the Ground Control Station dashboard.

    python3 scripts/serve.py [--port 8080] [--profile isr_sortie] [--scenario healthy]

On a hosting platform the port comes from $PORT and the server binds 0.0.0.0
automatically; every option can also be set from the environment:
HOST, PORT, AERODT_PROFILE, AERODT_SCENARIO, AERODT_HOURS, AERODT_RECORD,
AERODT_AUTOSTART, AERODT_DATA_DIR.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dt.paths import DATA_DIR, resolve, writable_dir
from dt.server import serve
from dt.simulator import SimConfig


def _flag(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def main():
    on_platform = "PORT" in os.environ
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0" if on_platform else "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    ap.add_argument("--profile", default=os.environ.get("AERODT_PROFILE", "isr_sortie"))
    ap.add_argument("--scenario", default=os.environ.get("AERODT_SCENARIO", "healthy"))
    ap.add_argument("--hours", type=int, default=int(os.environ.get("AERODT_HOURS", 420)))
    ap.add_argument("--model", default="data/models.json")
    ap.add_argument("--fdr", default="data/fdr.sqlite")
    ap.add_argument("--socketcan", action="store_true",
                    help="bind a real SocketCAN interface instead of the loopback")
    ap.add_argument("--channel", default="vcan0")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--no-autostart", action="store_true")
    a = ap.parse_args()

    # models ship with the code (read-only); only the recorder database moves
    # with AERODT_DATA_DIR, and falls back to a temp dir if that is read-only
    fdr = a.fdr if os.path.isabs(a.fdr) else os.path.join(DATA_DIR, os.path.basename(a.fdr))
    fdr = os.path.join(writable_dir(os.path.dirname(fdr)), os.path.basename(fdr))

    cfg = SimConfig(profile=a.profile, scenario=a.scenario, initial_hours=a.hours,
                    record=(not a.no_record) and _flag("AERODT_RECORD", True),
                    fdr_path=fdr, prefer_socketcan=a.socketcan)
    serve(a.host, a.port, cfg, a.model,
          autostart=(not a.no_autostart) and _flag("AERODT_AUTOSTART", True))


if __name__ == "__main__":
    main()
