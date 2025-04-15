.PHONY: run_backend run_frontend cleanup all install

install:
	pip install -r requirements.txt

cleanup:
	python cleanup.py

run_backend:
	python main.py

run_frontend:
	streamlit run frontend.py

start: cleanup
	python main.py & sleep 2 && streamlit run frontend.py
	
all: cleanup
	clear & python main.py & sleep 2 && streamlit run frontend.py