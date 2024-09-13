
FROM python:3.11

WORKDIR /app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY . /app

EXPOSE 8080
CMD ["fastapi", "run", "main.py", "--proxy-headers", "--port", "8080"]