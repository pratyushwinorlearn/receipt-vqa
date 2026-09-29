PY ?= python
ADAPTER ?=
SPLIT ?= test

.PHONY: install data baseline overfit debug train eval app test lint clean

install:
	pip install -e ".[dev]"

data:
	$(PY) -m receiptqa.data.download --config configs/data.yaml
	$(PY) -m receiptqa.data.build_qa --config configs/data.yaml

baseline:
	$(PY) -m receiptqa.eval.evaluate --config configs/eval.yaml --split val --modes base_plain base_brief

overfit:
	$(PY) -m receiptqa.train.train --config configs/train_256m_debug.yaml --overfit 16

debug:
	$(PY) -m receiptqa.train.train --config configs/train_256m_debug.yaml

train:
	$(PY) -m receiptqa.train.train --config configs/train_500m_qlora.yaml

eval:
	@test -n "$(ADAPTER)" || (echo "usage: make eval ADAPTER=outputs/runs/<run_id>/adapter [SPLIT=test]"; exit 1)
	$(PY) -m receiptqa.eval.evaluate --config configs/eval.yaml --split $(SPLIT) --adapter $(ADAPTER) --modes base_plain base_brief adapter_plain

app:
	ADAPTER_PATH=$(ADAPTER) $(PY) app/app.py

test:
	$(PY) -m pytest -q

lint:
	ruff check . && ruff format --check .

clean:
	rm -rf outputs/runs/* outputs/results/* .pytest_cache .ruff_cache
