# SN21 miner image: the model only (stdlib Python, no network, read-only root).
# Build from the repository root: docker build -t "$SN21_IMAGE_REPO:candidate" tracks/main
FROM python:3.12.7-slim-bookworm
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
COPY model/predict.py model/params.json /model/
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/model/predict.py"]
