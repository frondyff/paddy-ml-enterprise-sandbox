PYTHON ?= python

train:
	$(PYTHON) -m paddy.train --data data/raw/paddydataset.csv --outdir results

eval:
	$(PYTHON) -m paddy.evaluate --data data/raw/paddydataset.csv --model_path results/model.joblib --outdir results

semi:
	$(PYTHON) -m paddy.semisupervised --data data/raw/paddydataset.csv --outdir results

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q
