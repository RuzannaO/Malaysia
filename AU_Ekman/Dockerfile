FROM python:3.7-buster

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip
RUN apt-get -y install cmake

COPY requirements_mark.txt /app/requirements_mark.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements_mark.txt
COPY . /app

ENTRYPOINT ["python"]
CMD ["offline.py"]