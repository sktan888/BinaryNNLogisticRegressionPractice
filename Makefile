install:
		pip install --upgrade pip &&\
			pip install -r requirements.txt

test:
		python -m pytest --nbval helloAI.ipynb 
		python -m pytest -vv --cov=hello test_hello.py 

format:
		black *.py myLib/*.py

lint:
		pylint --disable=R,C *.py myLib/*.py

refactor: format lint

all: install lint test format