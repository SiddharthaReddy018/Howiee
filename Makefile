.PHONY: setup run test app

# Default target when someone just types 'make'
all: setup run

# Install requirements
setup:
	pip install -r requirements.txt

# Run the end-to-end pipeline (what the grader will do)
run:
	./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# Run the test suite
test:
	pytest tests/ -v

# Launch the Streamlit frontend
app:
	streamlit run src/app.py
