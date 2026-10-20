# MLBend: setup and verification shortcuts. Run `make setup` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
BENDENV := $(HOME)/.bend/bin/bendenv
BEND := $(or $(BEND_BIN),$(HOME)/.bend/bin/bend)

BEND_VERSION := 2.0.37
BEND_REF := ../bend-ref

.PHONY: setup check laws sync-bend clean lint env env-check

setup:            ## Bend, Lean and the Python venv
	scripts/setup.sh

check:            ## the proofs, the kernel (--verdict), the rejected programs, the NumPy comparison
	$(PY) reference/check_all.py

laws:             ## print every law statement of every package
	$(PY) reference/list_laws.py

env:              ## the environment: one lock, one env, for every package here
	$(BENDENV) lock
	$(BENDENV) sync

env-check:        ## the environment against the lock, and the example over it
	$(BENDENV) check
	$(BENDENV) run -- $(BEND) examples/stacks/main.bend

sync-bend:       ## a read-only copy of the pinned Bend source, for reading the compiler
	rm -rf $(BEND_REF)
	git clone --depth 1 --branch v$(BEND_VERSION) https://github.com/bendlang/bend.git $(BEND_REF)

clean:
	rm -rf reference/__pycache__ .scratch

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate
	bunx bend-lint --jobs max
