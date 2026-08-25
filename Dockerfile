FROM python:3.11-slim

WORKDIR /opt/cloud/services/maas-tokenizer

ENV PROD_ENV=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/cloud/services/maas-tokenizer/src:/opt/cloud/services/maas-tokenizer \
    TOKENIZER_LOG_PATH=/opt/cloud/logs/maas-tokenizer/access.log \
    TOKENIZER_LOG_MAX_BYTES=104857600 \
    TOKENIZER_LOG_BACKUP_COUNT=5 \
    TOKENIZER_LOG_REQUEST_BODY=false \
    TOKENIZER_LOG_REQUEST_BODY_MAX_BYTES=65536 \
    TOKENIZER_QUEUE_SIZE=100 \
    TOKENIZER_QUEUE_TIMEOUT_SECONDS=2

# 安装生产依赖
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# 创建运行用户和日志目录
RUN groupadd --gid 1000 service \
    && useradd --uid 1000 --gid 1000 --create-home service \
    && mkdir -p /opt/cloud/logs/maas-tokenizer \
    && chown -R service:service /opt/cloud/logs

# 复制服务运行所需文件
COPY --chown=service:service models/ ./models/
COPY --chown=service:service model_assets/ ./model_assets/
COPY --chown=service:service src/ ./src/
COPY --chown=service:service vendor/ ./vendor/

USER service

EXPOSE 8080

# 单进程、单 worker
CMD ["python", "-m", "uvicorn", "maas_tokenizer.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
