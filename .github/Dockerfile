# Frontend build stage
FROM node:18-alpine AS frontend-build
WORKDIR /app/site
COPY site/package*.json ./
RUN npm ci
COPY site/ .
RUN npm run build

# Backend stage
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 8000
CMD ["uvicorn", "site.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]