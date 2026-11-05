# Deploying AeroDT

AeroDT is a **long-running Python web service** (a simulator thread plus an HTTP/SSE
server). It needs a host that runs a persistent process. It will **not** work on
serverless platforms (Vercel, Netlify Functions, AWS Lambda) because those kill the
process after each request and do not support long-lived SSE streams.

Works on: Render, Railway, Fly.io, Heroku, Google Cloud Run, any VPS or Docker host.

## Start command (all platforms)

    python3 scripts/serve.py

The server reads `PORT` from the environment and binds `0.0.0.0` automatically.
No build step and no dependencies to install.

## Render
Push the repo, create a Blueprint from `render.yaml` (or a Web Service with start
command `python3 scripts/serve.py`, health check path `/healthz`).

## Railway / Heroku
`Procfile` and `railway.json` are included. Deploy the repo as-is.

## Docker / Cloud Run / Fly.io
    docker build -t aero-dt .
    docker run -p 8080:8080 aero-dt

## Environment variables (all optional)
| Variable | Default | Meaning |
|---|---|---|
| PORT | 8080 | listen port (set by the platform) |
| HOST | 0.0.0.0 on a platform, else 127.0.0.1 | bind address |
| AERODT_PROFILE | isr_sortie | mission profile |
| AERODT_SCENARIO | healthy | fault scenario |
| AERODT_HOURS | 420 | engine hours at start |
| AERODT_RECORD | true | flight data recorder on/off |
| AERODT_AUTOSTART | true | start the simulation on boot |
| AERODT_DATA_DIR | ./data | where the recorder database is written |

## Notes
- If the data directory is read-only the recorder automatically falls back to a
  temp directory. On free tiers with an ephemeral disk, recorded sorties are lost on
  restart; attach a persistent disk and point `AERODT_DATA_DIR` at it to keep them.
- Health check: `GET /healthz`.
- Run a single instance: simulator state lives in memory.
