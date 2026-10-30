# MLBend: setup and verification shortcuts. Run `make setup` once, then `make check`.
export BEND_NO_TELEMETRY := 1
export PATH := $(HOME)/.bend/bin:$(HOME)/.elan/bin:$(PATH)
PY := reference/.venv/bin/python
BENDENV := $(HOME)/.bend/bin/bendenv
BEND := $(or $(BEND_BIN),$(HOME)/.bend/bin/bend)

BEND_VERSION := 2.0.38
BEND_REF := ../bend-ref

.PHONY: setup check laws sync-bend clean lint env env-check release experiment experiments results bench plumbing flow figures select plots sweep sweep-agent

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
	$(BENDENV) run -- $(BEND) examples/transformers/attention/main.bend

experiment:       ## run one task and write a report card: make experiment TASK=iris-crossval
	@test -n "$(TASK)" || (echo "usage: make experiment TASK=<task> (see reference/experiment.py --list)" && false)
	$(PY) reference/experiment.py --task $(TASK) $(if $(TRAINER),--trainer $(TRAINER),)

experiments:      ## every task, with its report cards, saved under results/
	$(PY) reference/experiment.py --all --repeat 2

results:          ## regenerate results/RESULTS.md from the latest reports
	$(PY) reference/experiment.py --summary

bench:            ## a compute-only kernel benchmark, saved under results/
	$(PY) reference/bench_matmul.py

plumbing:         ## the time split of one ViT training step, saved under results/
	$(PY) reference/bench_plumbing.py

figures:          ## rebuild the README figures from results/, saved under docs/figures/
	$(PY) reference/make_readme_figures.py

flow:             ## train the 2D flow matching example, saved under results/flow-matching/
	@mkdir -p .scratch
	BEND_LIB=$(CURDIR)/.bend/env $(BEND) -o .scratch/flow.bin examples/diffusion/flow_matching/main.bend
	.scratch/flow.bin > .scratch/flow.log
	$(PY) reference/flow_results.py .scratch/flow.log

select:           ## model selection and a classification report (kNN)
	$(PY) reference/select_models.py

plots:            ## write examples/<name>/README.md and plot.svg for every example
	$(PY) reference/make_example_readmes.py

issues:          ## render issues-hit.md as issues-hit.html (bend-lang theme)
	$(PY) reference/make_issues_html.py

sweep:            ## start the W&B sweep defined in reference/wandb_sweep.yaml
	$(PY) -m wandb sweep reference/wandb_sweep.yaml

sweep-agent:      ## run an agent for a started sweep: make sweep-agent SWEEP=<id>
	@test -n "$(SWEEP)" || (echo "usage: make sweep-agent SWEEP=<id>" && false)
	$(PY) -m wandb agent $(SWEEP)

release:          ## release one package to the hub and tag it here: make release PKG=hf
	@test -n "$(PKG)" || (echo "usage: make release PKG=<package directory>" && false)
	scripts/release.sh $(PKG)

sync-bend:       ## a read-only copy of the pinned Bend source, for reading the compiler
	rm -rf $(BEND_REF)
	git clone --depth 1 --branch v$(BEND_VERSION) https://github.com/bendlang/bend.git $(BEND_REF)

clean:
	rm -rf reference/__pycache__ .scratch

lint:             ## bend-lint's performance rules over the Bend code: a report, not a gate
	bunx bend-lint --jobs max
