import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from ortools.sat.python import cp_model
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.config import RANDOM_SEED, MAX_WORKERS


def test_numpy():
    values = np.array([1, 2, 3, 4])

    assert values.sum() == 10


def test_pandas():
    dataframe = pd.DataFrame(
        {
            "test_id": ["test_1", "test_2"],
            "failed": [0, 1],
        }
    )

    assert len(dataframe) == 2
    assert dataframe["failed"].sum() == 1


def test_tfidf():
    messages = [
        "fix authentication error",
        "update database connection",
        "fix database timeout",
    ]

    vectorizer = TfidfVectorizer()

    features = vectorizer.fit_transform(messages)

    assert features.shape[0] == 3
    assert features.shape[1] > 0


def test_machine_learning():
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
    )

    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert np.array_equal(predictions, y)


def test_ortools():
    model = cp_model.CpModel()

    x = model.NewBoolVar("x")
    y = model.NewBoolVar("y")

    model.Add(2 * x + 3 * y <= 3)

    model.Maximize(4 * x + 5 * y)

    solver = cp_model.CpSolver()

    solver.parameters.random_seed = RANDOM_SEED
    solver.parameters.num_search_workers = MAX_WORKERS

    status = solver.Solve(model)

    assert status == cp_model.OPTIMAL
    assert solver.Value(x) == 0
    assert solver.Value(y) == 1
    assert solver.ObjectiveValue() == 5


def test_quantum_simulator():
    circuit = QuantumCircuit(2, 2)

    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])

    simulator = AerSimulator(
        max_parallel_threads=MAX_WORKERS,
        seed_simulator=RANDOM_SEED,
    )

    compiled_circuit = transpile(
        circuit,
        simulator,
        seed_transpiler=RANDOM_SEED,
    )

    result = simulator.run(
        compiled_circuit,
        shots=1024,
    ).result()

    counts = result.get_counts()

    assert sum(counts.values()) == 1024
    assert set(counts.keys()) == {"00", "11"}