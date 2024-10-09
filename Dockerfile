FROM python:3.12-slim

RUN python -m pip install poetry

WORKDIR /app
COPY . .

RUN poetry install

EXPOSE 80

CMD ["poetry", "run", "streamlit", "run", "main.py", "--server.port", "80"]

