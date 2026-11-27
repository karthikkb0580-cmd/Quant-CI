import json
import time

import numpy as np

from src.config import (
    MAX_QAOA_CANDIDATES,
    QUBO_BUDGET_SECONDS,
    QUBO_MAX_EXACT_VARIABLES,
    QUBO_RESULTS_DIR,
    QUBO_EXACT_SOLUTION_FILE,
)

from src.optimization.candidates import (
    load_predictions,
    prepare_candidates,
    get_run_ids,
)

from src.optimization.objective import (
    create_problem,
    evaluate_selection,
)

from src.quantum.qubo import (
    create_qubo,
    decode_state,
    qubo_energy,
)


def solve_exact_qubo(qubo):
    if qubo.size > QUBO_MAX_EXACT_VARIABLES:
        raise ValueError(
            "QUBO exceeds exact enumeration limit."
        )

    start = time.perf_counter()

    total_states = 2 ** qubo.size

    best_energy = float("inf")

    best_state = None

    feasible_states = 0

    batch_size = 4096

    shifts = np.arange(
        qubo.size,
        dtype=np.uint64,
    )

    for lower in range(
        0,
        total_states,
        batch_size,
    ):
        upper = min(
            lower + batch_size,
            total_states,
        )

        integers = np.arange(
            lower,
            upper,
            dtype=np.uint64,
        )

        states = (
            (
                integers[:, None]
                >> shifts
            ) & 1
        ).astype(float)

        energies = np.einsum(
            "bi,ij,bj->b",
            states,
            qubo.matrix,
            states,
            optimize=True,
        ) + qubo.offset

        used_units = (
            states[:, :qubo.test_count]
            @ qubo.duration_units
        )

        slack_units = (
            states[:, qubo.test_count:]
            @ qubo.slack_weights
        )

        feasible_states += int(
            np.count_nonzero(
                used_units + slack_units
                == qubo.budget_units
            )
        )

        index = int(
            np.argmin(energies)
        )

        energy = float(
            energies[index]
        )

        if energy < best_energy:
            best_energy = energy

            best_state = states[
                index
            ].astype(np.int8)

    decoded = decode_state(
        qubo,
        best_state,
    )

    if not decoded["constraint_satisfied"]:
        raise RuntimeError(
            "Exact solution violates QUBO constraint."
        )

    selection = decoded["selection"]

    result = evaluate_selection(
        qubo.original_problem,
        selection,
    )

    if not result["feasible"]:
        raise RuntimeError(
            "Original budget constraint violated."
        )

    result.update(
        {
            "algorithm": "exact_qubo_enumeration",
            "energy": qubo_energy(
                qubo,
                best_state,
            ),
            "binary_state": best_state.tolist(),
            "selection": selection.tolist(),
            "slack_bits": (
                decoded["slack_bits"].tolist()
            ),
            "slack_units": decoded[
                "slack_units"
            ],
            "used_duration_units": decoded[
                "used_duration_units"
            ],
            "budget_units": qubo.budget_units,
            "constraint_satisfied": decoded[
                "constraint_satisfied"
            ],
            "states_evaluated": total_states,
            "constraint_satisfying_states": (
                feasible_states
            ),
            "optimal_qubo_solution": True,
            "solver_seconds": (
                time.perf_counter() - start
            ),
        }
    )

    return selection, result


def main():
    QUBO_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = load_predictions(
        "validation"
    )

    run_id = get_run_ids(
        predictions
    )[0]

    candidates = prepare_candidates(
        predictions,
        run_id,
        MAX_QAOA_CANDIDATES,
    )

    problem = create_problem(
        candidates,
        QUBO_BUDGET_SECONDS,
    )

    qubo = create_qubo(
        problem
    )

    _, result = solve_exact_qubo(
        qubo
    )

    result["run_id"] = run_id

    QUBO_EXACT_SOLUTION_FILE.write_text(
        json.dumps(
            result,
            indent=4,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("=" * 70)
    print("QuantCI - Exact QUBO Solver")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=4,
        )
    )

    print("=" * 70)
    print("EXACT QUBO OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()