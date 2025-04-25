FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 443
CMD ["python","-m","streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
