# Reproducing QuantCI

## Environment

- Python 3.14.10
- Windows PowerShell
- 16 GB RAM
- Intel Core i5
- Local CPU-based quantum simulation

## Data Preparation

```bash
python -m src.data.inspect_archive
python -m src.data.extract_dataset
python -m src.data.validate_dataset
python -m src.data.load_dataset
python -m src.data.analyze_dataset
python -m src.data.split_dataset
```

## Feature Engineering

```bash
python -m src.features.feature_engineering
python -m src.features.build_matrices
python -m src.data.temporal_embargo
```

## Machine Learning

```bash
python -m src.ml.baseline
python -m src.ml.logistic_regression
python -m src.ml.random_forest
python -m src.ml.final_evaluation
```

## Classical Optimization

```bash
python -m src.optimization.candidates
python -m src.optimization.objective
python -m src.optimization.greedy
python -m src.optimization.ortools_solver
```

## Quantum Optimization

```bash
python -m src.quantum.qubo
python -m src.quantum.exact_solver
python -m src.quantum.ising
python -m src.quantum.qaoa_solver
python -m src.quantum.candidate_pruning
python -m src.quantum.qaoa_experiments
```

## Final Evaluation

Generate the slack-repair and random-sampling experiment artifacts before executing the unified benchmark.

```bash
python -m src.evaluation.unified_benchmark
python -m src.evaluation.heldout_evaluation
```

## Automated Testing

```bash
python -m pytest tests -v
```

## Dashboard

```bash
python -m streamlit run src/demo/app.py
```

## Reproducibility Notes

Random seeds are fixed where supported.

Solver runtimes may vary depending on hardware, operating system, thread scheduling and installed dependency versions.

Quantum measurement outcomes are stochastic.

The raw dataset must be acquired separately before executing the pipeline.