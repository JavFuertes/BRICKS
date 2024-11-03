FROM node:18-alpine AS frontend-build
WORKDIR /app/site

COPY site/package*.json ./
RUN npm ci

COPY site/ .
RUN npm run build

FROM python:3.9 AS backend

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "site.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
