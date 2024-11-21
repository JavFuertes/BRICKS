FROM node:18

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY .github/site/package*.json ./
RUN npm install

COPY .github/site ./
COPY setup.py ./
COPY .github/requirements.txt ./

RUN python3 -m venv venv

RUN /app/venv/bin/pip install --upgrade pip && \
    /app/venv/bin/pip install git+https://github.com/JavFuertes/BRICKS.git@main

RUN /app/venv/bin/pip install -r requirements.txt && \
    /app/venv/bin/pip install -e .

COPY .github/site ./
COPY .github/site/src/api ./api
COPY .github/assets ./src/pages/notebooks/.github/assets
COPY README.md ./src/pages/notebooks/

ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 4321 8000 8050 8051 8052
CMD ["sh", "-c", "npm run dev -- --host 0.0.0.0 --port 4321 & . /app/venv/bin/activate && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"]