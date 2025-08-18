.PHONY: run dev fmt lint type check docker-build

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	UVICORN_RELOAD=true uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

fmt:
	ruff check . --fix

lint:
	ruff check .

type:
	mypy app

check: lint type

docker-build:
	docker build -t semantic-bot -f Dockerfile.optimized .
