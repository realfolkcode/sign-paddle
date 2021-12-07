FROM python:3.8

COPY . /sign
WORKDIR /sign

RUN python -m pip install paddlepaddle-gpu==2.2.1 -i https://mirror.baidu.com/pypi/simple
RUN pip install pgl
RUN pip install -U scikit-learn
RUN pip install tqdm

RUN mkdir logs
RUN mkdir models

RUN chmod +x scripts/commands.sh
RUN chmod +x scripts/download.sh

ENTRYPOINT ["/bin/bash", "-c", "./scripts/commands.sh"]

