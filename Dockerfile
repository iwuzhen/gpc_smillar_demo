# Stage 1: Download the model
FROM python:3.12-slim as model-downloader

WORKDIR /model

RUN python -m pip install transformers torch
# RUN python -m pip install poetry -i https://mirrors.aliyun.com/pypi/simple

# Install necessary packages
COPY download_model.py .
RUN python download_model.py
# RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
COPY --from=model-downloader /model /app/model

RUN python -m pip install poetry

WORKDIR /app
COPY . .

RUN poetry install

EXPOSE 80

CMD ["poetry", "run", "streamlit", "run", "main.py", "--server.port", "80"]

