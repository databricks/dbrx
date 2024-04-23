FROM mosaicml/llm-foundry:2.2.1_cu121_flash2-latest

ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

COPY requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt

COPY . .
