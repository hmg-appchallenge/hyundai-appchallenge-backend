FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8 \
    HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
RUN mkdir -p agent logs server/tmp && python -m server --download

EXPOSE 7860
CMD ["python", "-m", "server", "--mode", "real", "--host", "0.0.0.0", "--port", "7860"]
