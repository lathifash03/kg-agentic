# Bullseye, not the default bookworm `python:3.11-slim`.
#
# Bookworm ships glibc 2.36, whose pthread implementation calls `clone3()`.
# Docker's default seccomp profile before ~23.0 returns EPERM for that syscall
# instead of ENOSYS, so glibc never falls back to `clone()` and every thread
# creation dies with "RuntimeError: can't start new thread" - the build fails
# during `pip install`, long before any of our code runs. Reproduced on Docker
# 20.10.2. Bullseye's glibc 2.31 uses `clone()` and builds everywhere.
#
# Safe to move back to `python:3.11-slim` once every host running this image is
# on Docker >= 23.
FROM python:3.11-slim-bullseye

# PYTHONUNBUFFERED so uvicorn's logs reach `docker logs` immediately rather than
# sitting in a pipe buffer - the difference between diagnosing a failed start
# and staring at an empty log.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY kg_agent ./kg_agent
COPY scripts ./scripts

# Runs as a non-root user: this image is reachable over the tailnet, and it is
# pointed at a graph owned by someone else. Nothing it does needs root.
RUN useradd --create-home --uid 10001 kgagent && chown -R kgagent:kgagent /app
USER kgagent

EXPOSE 8000

# Verifies the app answers, not merely that the process exists. Start period is
# generous because the first request opens the Neo4j driver.
HEALTHCHECK --interval=30s --timeout=8s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=5)"

# API sebagai default; untuk CLI:
#   docker compose run --rm agent python -m kg_agent.cli --query "..."
# Untuk smoke test dari dalam container:
#   docker compose exec agent python scripts/smoke_endpoint.py --url http://localhost:8000
CMD ["uvicorn", "kg_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
