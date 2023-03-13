# app/Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt update -y && apt install -y curl ffmpeg libsm6 libxext6

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]