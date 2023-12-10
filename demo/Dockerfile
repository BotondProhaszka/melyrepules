FROM python:3.7-slim

ENV PORT 7860

EXPOSE $PORT

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#update apt-get
RUN apt-get update

CMD ["python", "gradio_interface.py"]
