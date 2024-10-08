# Stage 1: Download the model
FROM python:3.12 as model-downloader

WORKDIR /model

RUN python -m pip install modelscope[framework] -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN python -m pip install poetry -i https://mirrors.aliyun.com/pypi/simple


# Install necessary packages
COPY download_model.py .
RUN python download_model.py
# RUN pip install --no-cache-dir -r requirements.txt