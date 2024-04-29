FROM python:3.7

ADD MOBILENET.py /
ADD img.jpeg /

RUN pip install --upgrade pip
RUN pip install pystrich
RUN pip install tensorflow==2.7
RUN pip install matplotlib



CMD [ "python", "./MOBILENET.py" ]

