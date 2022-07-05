FROM python:3.8-slim
RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean
WORKDIR /usr/src/app
COPY requirements.* .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["./entrypoint.sh"]

