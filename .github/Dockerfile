FROM node:18

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY .github/site/package*.json ./
COPY .github/requirements.txt ./

RUN ls -la
RUN npm install
RUN python3 -m venv venv && \
    . /app/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY .github/site ./
COPY .github/site/src/api ./api
COPY .github/assets ./src/pages/notebooks/.github/assets
COPY README.md ./src/pages/notebooks/
COPY bricks/analytical ./analytical

ENV PYTHONPATH=/app:/app/analytical:$PYTHONPATH

EXPOSE 4321 8000
CMD ["sh", "-c", "npm run dev -- --host 0.0.0.0 --port 4321 & . /app/venv/bin/activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"]