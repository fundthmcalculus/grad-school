# Scope: this repository only. The pinned submodules (tribble-fis,
# tribble-cluster, tribble-opt) are independent uv projects with their own
# formatting state and CI — formatting them from here would rewrite files
# inside a submodule checkout (AGENTS.md non-negotiable 1), and black would
# also descend into their .venv directories (hundreds of MB of vendored
# code). Mirrors CI, whose lint job checks out without submodules.
.PHONY: format deploy

# black is the gate (CI gates black only; flake8/mypy run there with
# continue-on-error). flake8 and mypy are run for reporting but must not
# fail the target: the repo carries hundreds of pre-existing findings in
# graded coursework and legacy research scripts, and mypy cannot run
# whole-tree at all (module-name collisions across the coursework
# directories).
format:
	black ./ --extend-exclude 'tribble-(fis|cluster|opt)/'
	flake8 ./ --exclude '.git,__pycache__,.venv,.tox,.eggs,.hatch,build,dist,tribble-fis,tribble-cluster,tribble-opt' || echo 'flake8: informational (pre-existing findings) -- see the list above'
	mypy ./ --exclude 'tribble-(fis|cluster|opt)' || echo 'mypy: informational (pre-existing findings) -- see the list above'
