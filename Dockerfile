FROM python:3.6
MAINTAINER gopalsvs venkatasaigopal01@gmail.com
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "drowsiness_detection:app"]
