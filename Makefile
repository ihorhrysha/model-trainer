.PHONY: clean install tests run all

clean:
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.log' -delete

install: 
	pip install -r requirements.txt

tests:
	python manage.py test

run:
	python manage.py run

all: clean install tests run