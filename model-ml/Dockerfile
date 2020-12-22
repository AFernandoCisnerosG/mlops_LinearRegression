FROM python:3

WORKDIR /usr/src/mlops

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "./model-ml/model.py"]