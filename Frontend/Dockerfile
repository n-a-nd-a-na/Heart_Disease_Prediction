FROM python:3.11

WORKDIR /app

COPY Frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Frontend/ .

EXPOSE 8501

CMD ["streamlit","run","app.py","--server.address=0.0.0.0"]