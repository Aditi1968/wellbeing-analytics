
.PHONY: run seed etl analytics clean

run:
	streamlit run app.py

seed:
	python scripts/seed_sqlite.py

etl:
	python scripts/etl_pipeline.py --db data/health360.db

analytics:
	python scripts/analytics.py --db data/health360.db

clean:
	rm -f data/health360.db
	rm -rf __pycache__ */__pycache__
