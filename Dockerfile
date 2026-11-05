FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app
COPY . /app

# Run as a non-root user; the flight recorder writes to /app/data
RUN useradd -m aerodt && chown -R aerodt /app
USER aerodt

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python3 -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','8080'),timeout=4)"

CMD ["python3", "scripts/serve.py"]
