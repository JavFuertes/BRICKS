FROM node:18-alpine AS frontend-build
WORKDIR /app/site

COPY site/package*.json ./
RUN npm ci

COPY site/ .
RUN npm run build
