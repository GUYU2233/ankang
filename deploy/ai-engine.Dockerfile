FROM python:3.11-slim

WORKDIR /app
COPY ai-engine/requirements.txt /app/ai-engine/requirements.txt
RUN pip install --no-cache-dir -r /app/ai-engine/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY ai-engine /app/ai-engine
WORKDIR /app/ai-engine
EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]