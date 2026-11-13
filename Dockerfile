FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Buat non-root user (standar Hugging Face Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependensi
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin source code
COPY --chown=user:user . .

# Hugging Face Spaces port
EXPOSE 7860

# Jalankan FastAPI REST API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
