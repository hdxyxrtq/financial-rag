FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（chromadb 等需要编译 C 扩展）
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/chroma_db data/eval

EXPOSE 8501 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# 默认启动 Streamlit
# API 模式：docker run ... python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
