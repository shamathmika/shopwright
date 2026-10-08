FROM python:3.12-slim
WORKDIR /app
COPY docker/requirements-agent.txt .
RUN pip install --no-cache-dir -r requirements-agent.txt
COPY shopwright ./shopwright
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "shopwright.api:app", "--host", "0.0.0.0", "--port", "8000"]
