# MLBend: setup and verification shortcuts. Run `make setup` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python

.PHONY: setup check laws clean lint

setup:            ## Bend, Lean and the Python venv
	scripts/setup.sh

check:            ## the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
	$(PY) reference/check_all.py

laws:             ## print every law statement of every package
	$(PY) reference/list_laws.py

clean:
	rm -rf reference/__pycache__ .scratch

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate
	bunx bend-lint --jobs max
