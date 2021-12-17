FROM paddlepaddle/paddle:2.2.1-gpu-cuda11.2-cudnn8

COPY . /sign
WORKDIR /sign

RUN pip install pgl
RUN pip install -U scikit-learn
RUN pip install tqdm

RUN chmod +x scripts/commands.sh
RUN chmod +x scripts/download.sh

ENTRYPOINT ["/bin/bash", "-c", "./scripts/commands.sh"]

