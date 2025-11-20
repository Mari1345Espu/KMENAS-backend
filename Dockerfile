# Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

COPY app.py .

# Render asigna dinámicamente el puerto en $PORT
ENV PORT=5000

EXPOSE 5000

CMD gunicorn app:app --workers 4 --bind 0.0.0.0:$PORT
