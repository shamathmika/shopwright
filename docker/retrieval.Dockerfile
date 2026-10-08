FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY docker/requirements-retrieval.txt .
RUN pip install --no-cache-dir -r requirements-retrieval.txt
ENV HF_HOME=/opt/hf PYTHONPATH=/app
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
COPY data/catalog.parquet data/reviews.parquet data/embeddings.npy data/flat.index ./data/
COPY data/bm25 ./data/bm25
COPY shopwright ./shopwright
EXPOSE 8000
CMD ["uvicorn", "shopwright.retrieval_api:app", "--host", "0.0.0.0", "--port", "8000"]
