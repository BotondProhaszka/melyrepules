FROM python:3.10.0-slim

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#update apt-get
RUN apt-get update

CMD ["python", "gradio_interface.py"]
