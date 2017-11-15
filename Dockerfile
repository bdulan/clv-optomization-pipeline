FROM python:3

COPY . /app
WORKDIR /app

# RUN pip install pandas json cx_oracle sqlalchemy json
RUN pip install pandas

CMD [ "python", "./main.py" ]
