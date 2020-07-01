.PHONY: clean install tests run all

clean:
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.log' -delete

install:
	pip install -U pip
	pip install -r requirements.txt

tests:
	flask utils test

run:
	flask run

all: clean install tests run