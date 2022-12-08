FROM tensorflow/tensorflow

RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py
COPY download_model.py download_model.py
COPY templates templates
RUN download_model.py

ENTRYPOINT ["python", "app.py"]
