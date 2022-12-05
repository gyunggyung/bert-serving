FROM tensorflow/tensorflow

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py
COPY templates templates

ENTRYPOINT ["python", "app.py"]
