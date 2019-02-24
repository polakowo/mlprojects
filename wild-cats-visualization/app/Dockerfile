FROM python:3.6-slim-stretch

RUN apt-get update && apt-get install -y python3-dev gcc \
    && apt-get install -y libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . app/

EXPOSE 8000

CMD ["python", "app/server.py", "serve"]