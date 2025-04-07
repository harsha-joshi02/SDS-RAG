.PHONY: run_backend run_frontend cleanup all install

install:
	pip install -r requirements.txt

cleanup:
	python3 cleanup.py

run_backend:
	python3 main.py

run_frontend:
	streamlit run frontend.py

start: cleanup
	python3 main.py & sleep 2 && streamlit run frontend.py
	
all: cleanup
	clear & python3 main.py & sleep 2 && streamlit run frontend.py