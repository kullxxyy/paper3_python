# Paper 3 Refactored Python Code

This folder is a refactored copy of `../paper3_python_v4.py`.

The original file has not been modified. The statistical logic was split into
smaller modules, and a class-based entry point was added so the analysis can be
run either from Python code or from the command line.

## Folder Structure

```text
paper3_refactored/
  main.py                    # command-line entry point
  pipeline.py                # Paper3Pipeline class
  config.py                  # paths and run-mode configuration
  data_prep.py               # original prepare_data()
  friction.py                # market-friction indices
  panel.py                   # household panel and FE helper
  nonfarm_outcomes.py        # nonfarm outcome construction
  analyses/
    baseline.py              # baseline DID and friction heterogeneity
    event_study.py           # friction event study and figures
    bootstrap.py             # cluster inference and wild bootstrap
    decomposition.py         # farm decomposition
    nonfarm.py               # nonfarm labor analysis
    placebo.py               # placebo timing tests
```

## Install Dependencies

From `/Users/lenovo/PhD papers/Paper 3_new2/python_code`, install the packages
used by the original script:

```bash
python3 -m pip install -r paper3_python/requirements.txt
```

If you use a conda environment or a PyCharm interpreter, run the same command
inside that environment.

## Run From Command Line

The recommended command is run from the parent `python_code` directory:

Run everything:

```bash
python3 -m paper3_python.main --mode all
```

If the terminal is already inside `paper3_refactored`, use:

```bash
python3 -m pip install -r requirements.txt
python3 -m main --mode all
```

Run only one block:

```bash
python3 -m paper3_python.main --mode baseline
python3 -m paper3_python.main --mode event
python3 -m paper3_python.main --mode bootstrap
python3 -m paper3_python.main --mode decomposition
python3 -m paper3_python.main --mode nonfarm
python3 -m paper3_python.main --mode placebo
```

Available modes:

```text
prep, baseline, event, bootstrap, decomposition, nonfarm, placebo, all
```

Default paths match the original script:

```text
data:    /Users/lenovo/PhD papers/Paper 3_new2/data/CHNS_data_analysis
results: /Users/lenovo/PhD papers/Paper 3_new2/python_code/results
logs:    /Users/lenovo/PhD papers/Paper 3_new2/python_code/logs
```

You can override them:

```bash
python3 -m paper3_python.main \
  --mode event \
  --data-dir "/path/to/CHNS_data_analysis" \
  --output-dir "/path/to/results" \
  --log-dir "/path/to/logs"
```

## Run From Python

```python
from paper3_python import Paper3Config, Paper3Pipeline

config = Paper3Config(run_mode="baseline")
pipeline = Paper3Pipeline(config)
state = pipeline.run()

baseline_results = state["baseline"]
```

You can also call analysis blocks directly:

```python
pipeline = Paper3Pipeline()
pipeline.prepare()
pipeline.run_baseline()
pipeline.run_event_study()
```

The pipeline keeps intermediate objects in `pipeline.state`, replacing the
global `STATE` dictionary from the original single-file script.
