#!/bin/sh
npm install && \
pip install -e /app/bricks && \
(npm run dev -- --host 0.0.0.0 & \
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload)