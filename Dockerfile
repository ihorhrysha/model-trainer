FROM python:3.7 

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt 

# Copy source code
COPY manage.py manage.py
COPY migrations migrations
COPY trainer_app trainer_app

EXPOSE 5000 
ENTRYPOINT [ "python" ] 
CMD [ "manage.py", "run" ] 
