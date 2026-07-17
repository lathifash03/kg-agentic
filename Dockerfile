FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY kg_agent ./kg_agent

EXPOSE 8000

# API sebagai default; untuk CLI:
#   docker compose run --rm agent python -m kg_agent.cli --query "..."
CMD ["uvicorn", "kg_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
