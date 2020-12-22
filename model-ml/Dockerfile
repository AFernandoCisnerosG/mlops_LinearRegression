FROM python:3

COPY . .

RUN pip install -r requirements.txt

WORKDIR /usr/src/mlops

CMD [ "python", "./model-ml/model.py"]