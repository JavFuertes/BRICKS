FROM node:18-alpine AS frontend-build
WORKDIR /app/site

COPY site/package*.json ./
RUN npm ci

COPY site/ .
RUN npm run build

FROM python:3.9 AS backend

WORKDIR /app

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000
CMD ["uvicorn", ".github.site.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
