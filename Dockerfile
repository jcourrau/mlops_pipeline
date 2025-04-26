FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501

ARG GIT_COMMIT
ARG BUILD_TIME

ENV GIT_COMMIT=$GIT_COMMIT
ENV BUILD_TIME=$BUILD_TIME

# arrancamos Streamlit en 0.0.0.0:8501
CMD ["python","-m","streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
