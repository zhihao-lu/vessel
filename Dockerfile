FROM python:3.9.7-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

# To fix error with opencv in docker
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update -y && apt-get install -y gcc && apt-get install -y git
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/ultralytics/yolov3 yolov3

# Need to copy folders out to prevent import errors
COPY yolov3/models models
COPY yolov3/utils utils

COPY main.py main.py
COPY exp86/weights/best.pt best.pt

ENTRYPOINT [ "python3", "main.py" ]