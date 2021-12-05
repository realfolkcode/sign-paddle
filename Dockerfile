FROM python:3.8

COPY . /sign
WORKDIR /sign

RUN python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
RUN pip install pgl
RUN pip install -U scikit-learn
RUN pip install tqdm

RUN mkdir logs

RUN chmod +x scripts/commands.sh
RUN chmod +x scripts/download.sh
WORKDIR scripts

ENTRYPOINT ["/bin/bash", "-c", "./commands.sh"]
