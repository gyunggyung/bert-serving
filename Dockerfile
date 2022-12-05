FROM tensorflow/tensorflow

RUN apt-get update
RUN apt install -y git-all

RUN pip install tensorflow-hub\
pip install tensorflow_text\
pip install flask 

WORKDIR /serving

COPY app.py app.py
COPY templates templates

ENTRYPOINT ["python app.py"]
