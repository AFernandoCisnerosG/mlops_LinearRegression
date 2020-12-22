FROM python:3

WORKDIR /usr/src/mlops

ADD ./requirements.txt /usr/src/mlops/requirements.txt
ADD ./model.py /usr/src/mlops/model.py
ADD ./test_suma.py /usr/src/test_suma.py

RUN pip install -r requirements.txt

ADD . /usr/src/mlops

CMD [ "python", "./model-ml/model.py"]