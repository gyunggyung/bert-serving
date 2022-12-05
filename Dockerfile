FROM tensorflow/tensorflow

RUN apt-get update
RUN apt-get install -y git-all
RUN pip install --upgrade pip
RUN pip install tensorflow-hub
RUN pip install tensorflow_text
RUN pip install flask

COPY app.py app.py
COPY templates templates

ENTRYPOINT ["python", "app.py"]
