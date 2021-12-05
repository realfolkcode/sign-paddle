FROM python:3.8

COPY . /sign
WORKDIR /sign

RUN python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
RUN pip install pgl
RUN pip install -U scikit-learn
RUN pip install tqdm

RUN chmod +x scripts/
WORKDIR scripts

ENTRYPOINT ["/bin/bash", "-c", "./commands.sh"]

