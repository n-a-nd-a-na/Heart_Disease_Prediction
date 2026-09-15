FROM python:3.11

WORKDIR /app

COPY Backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Backend/ Backend/

COPY model /app/model

EXPOSE 8000

CMD ["uvicorn","Backend.main:app","--host","0.0.0.0","--port","8000"]