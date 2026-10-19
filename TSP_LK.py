# Based on an implementation by kikicastroneto
# LK for TSP

import json
import math
import multiprocessing as mp
import os
import random
import tempfile
import time
import traceback
import uuid
from pathlib import Path


FILENAME = "fnl4461.tsp"
TIME_LIMIT = 300  # seconds
RANDOM_SEED = 42

BACKTRACKING = (5, 5)
REDUCTION_LEVEL = 4
REDUCTION_CYCLE = 4


INSTANCE_PATH = Path(__file__).resolve().parent / FILENAME


# TSPLIB READER

def read_tsplib(filename):
    """Read a TSPLIB instance and return its name, nodes and costs."""

    with open(filename, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    header = {}
    sections = {}
    current_section = None

    for line in lines:
        if line.upper() == "EOF":
            break

        upper = line.upper()

        if upper in {
            "NODE_COORD_SECTION",
            "EDGE_WEIGHT_SECTION",
            "DISPLAY_DATA_SECTION",
            "DEPOT_SECTION",
            "FIXED_EDGES_SECTION",
        }:
            current_section = upper
            sections.setdefault(current_section, [])
            continue

        if current_section is not None:
            sections[current_section].append(line)
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()
        else:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                header[parts[0].strip().upper()] = parts[1].strip()

    dimension = int(header["DIMENSION"])
    edge_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    edge_format = header.get("EDGE_WEIGHT_FORMAT", "").upper()
    name = header.get("NAME", Path(filename).stem)

    # Read coordinates, if provided.
    coordinates = {}

    for line in sections.get("NODE_COORD_SECTION", []):
        parts = line.split()
        if len(parts) >= 3:
            node_id = int(parts[0])
            coordinates[node_id] = (float(parts[1]), float(parts[2]))

    # Use the file's node order for internal indices.
    if coordinates:
        external_ids = list(coordinates.keys())

        if len(external_ids) != dimension:
            raise ValueError(
                f"Expected {dimension} nodes, found {len(external_ids)}."
            )


        external_ids.sort()
    else:
        external_ids = list(range(1, dimension + 1))

    if len(external_ids) != dimension:
        raise ValueError("The number of node IDs does not match DIMENSION.")

    # Create Node2D objects. Tsp/Tour assigns internal IDs 0..n-1.
    from lk_heuristic.models.node import Node2D

    nodes = []
    for external_id in external_ids:
        x, y = coordinates.get(external_id, (0.0, 0.0))
        nodes.append(Node2D(x, y))

    # Explicit distance matrix.
    if edge_type == "EXPLICIT":
        values = []
        for line in sections.get("EDGE_WEIGHT_SECTION", []):
            values.extend(float(value) for value in line.split())

        matrix = [[0.0] * dimension for _ in range(dimension)]
        k = 0

        def next_value():
            nonlocal k
            if k >= len(values):
                raise ValueError("EDGE_WEIGHT_SECTION is incomplete.")
            value = values[k]
            k += 1
            return value

        if edge_format == "FULL_MATRIX":
            for i in range(dimension):
                for j in range(dimension):
                    matrix[i][j] = next_value()

        elif edge_format == "LOWER_DIAG_ROW":
            for i in range(dimension):
                for j in range(i + 1):
                    value = next_value()
                    matrix[i][j] = value
                    matrix[j][i] = value

        elif edge_format == "UPPER_DIAG_ROW":
            for i in range(dimension):
                for j in range(i, dimension):
                    value = next_value()
                    matrix[i][j] = value
                    matrix[j][i] = value

        elif edge_format == "LOWER_ROW":
            for i in range(dimension):
                for j in range(i):
                    value = next_value()
                    matrix[i][j] = value
                    matrix[j][i] = value

        elif edge_format == "UPPER_ROW":
            for i in range(dimension):
                for j in range(i + 1, dimension):
                    value = next_value()
                    matrix[i][j] = value
                    matrix[j][i] = value

        else:
            raise ValueError(
                f"Unsupported EXPLICIT format: {edge_format}"
            )

        if k != len(values):
            raise ValueError(
                f"Unexpected extra edge weights: {len(values) - k}"
            )

        def cost_function(a, b):
            return matrix[a.id][b.id]

        return name, nodes, cost_function

    # Coordinate-based distance functions.
    if edge_type not in {"EUC_2D", "CEIL_2D", "ATT"}:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}. "
            "Supported types: EUC_2D, CEIL_2D, ATT, EXPLICIT."
        )

    if len(coordinates) != dimension:
        raise ValueError(
            f"{edge_type} requires NODE_COORD_SECTION with {dimension} nodes."
        )

    def cost_function(a, b):
        dx = a.x - b.x
        dy = a.y - b.y
        distance_squared = dx * dx + dy * dy

        if edge_type == "EUC_2D":
            # TSPLIB nearest-integer convention.
            return int(math.sqrt(distance_squared) + 0.5)

        if edge_type == "CEIL_2D":
            return math.ceil(math.sqrt(distance_squared))

        if edge_type == "ATT":
            rij = math.sqrt(distance_squared / 10.0)
            tij = int(rij + 0.5)
            return tij + 1 if tij < rij else tij

    return name, nodes, cost_function


# RESULT CHECKPOINT

def save_checkpoint(path, data):
    """Atomically save the most recent valid best tour."""
    temporary_path = path + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    os.replace(temporary_path, path)


def get_tour_order(tour):
    """Return the tour as internal node IDs, or raise if it is invalid."""
    start = tour.nodes[0]
    current = start
    order = []
    visited = set()

    while current.id not in visited:
        visited.add(current.id)
        order.append(current.id)

        if current.succ is None or current.pred is None:
            raise ValueError("A node has no predecessor or successor.")

        if current.succ.pred != current:
            raise ValueError("Successor/predecessor links are inconsistent.")

        current = current.succ

        if len(order) > len(tour.nodes):
            raise ValueError("Tour contains too many nodes.")

    if current != start:
        raise ValueError("The route does not close into one cycle.")

    if len(order) != len(tour.nodes):
        raise ValueError("The route does not visit every node exactly once.")

    if len(set(order)) != len(tour.nodes):
        raise ValueError("The route contains repeated nodes.")

    return order


# WORKER PROCESS


def run_lk_worker(instance_path, checkpoint_path, seed):
    started = time.perf_counter()

    try:
        random.seed(seed)

        from lk_heuristic.models.tsp import Tsp

        name, nodes, cost_function = read_tsplib(instance_path)

        tsp = Tsp(
            nodes=nodes,
            cost_function=cost_function,
            shuffle=True,
            backtracking=BACKTRACKING,
            reduction_level=REDUCTION_LEVEL,
            reduction_cycle=REDUCTION_CYCLE,
            tour_type="cycle",
        )

        tsp.tour.set_cost(tsp.cost_matrix)

        best_cost = tsp.tour.cost
        best_order = get_tour_order(tsp.tour)
        completed_calls = 0

        save_checkpoint(checkpoint_path, {
            "status": "running",
            "instance": name,
            "best_cost": best_cost,
            "best_order": best_order,
            "completed_calls": completed_calls,
            "worker_runtime_seconds": 0.0,
        })

        while True:
            # Save the cost before this LK1 call.
            previous_cost = tsp.tour.cost

            # Run the LK1 improvement procedure.
            tsp.lk1_improve()

            # Recalculate the tour cost after the procedure.
            tsp.tour.set_cost(tsp.cost_matrix)

            current_cost = tsp.tour.cost
            current_order = get_tour_order(tsp.tour)
            completed_calls += 1

            # Keep the best tour found so far.
            if current_cost < best_cost:
                best_cost = current_cost
                best_order = current_order

            save_checkpoint(checkpoint_path, {
                "status": "running",
                "instance": name,
                "best_cost": best_cost,
                "best_order": best_order,
                "completed_calls": completed_calls,
                "worker_runtime_seconds": time.perf_counter() - started,
            })

            # Allow enough calls for the configured reduction cycle
            # to reach the double-bridge stage.
            if (
                completed_calls >= REDUCTION_CYCLE
                and current_cost >= previous_cost
            ):
                break

        save_checkpoint(checkpoint_path, {
            "status": "completed",
            "instance": name,
            "best_cost": best_cost,
            "best_order": best_order,
            "completed_calls": completed_calls,
            "worker_runtime_seconds": time.perf_counter() - started,
        })

    except Exception:
        save_checkpoint(checkpoint_path, {
            "status": "error",
            "error": traceback.format_exc(),
            "completed_calls": locals().get("completed_calls", 0),
            "worker_runtime_seconds": time.perf_counter() - started,
        })


# MAIN PROGRAM

def main():
    if not INSTANCE_PATH.is_file():
        raise FileNotFoundError(
            f"TSPLIB file not found: {INSTANCE_PATH}\n"
            "Place the .tsp file next to run_lk.py or change FILENAME."
        )

    checkpoint_path = os.path.join(
        tempfile.gettempdir(),
        f"lk_checkpoint_{uuid.uuid4().hex}.json",
    )

    process = mp.Process(
        target=run_lk_worker,
        args=(str(INSTANCE_PATH), checkpoint_path, RANDOM_SEED),
    )

    started = time.perf_counter()
    process.start()

    process.join(TIME_LIMIT)
    timed_out = process.is_alive()

    if timed_out:
        process.terminate()
        process.join()
        # terminate() is the hard cutoff; kill() is a fallback.
        if process.is_alive():
            process.kill()
            process.join()

    elapsed = time.perf_counter() - started

    print("=" * 55)
    print("Lin-Kernighan (LK1) — TSPLIB experiment")
    print("=" * 55)
    print(f"Instance: {FILENAME}")
    print(f"Time limit: {TIME_LIMIT} seconds")
    print(f"Measured wall-clock time: {elapsed:.3f} seconds")

    result = None
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                result = json.load(file)
        finally:
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass

    if result is None:
        print("Status: TIME LIMIT / NO CHECKPOINT")
        print("No saved valid tour was available.")
        return

    if result.get("status") == "error":
        print("Status: ERROR")
        print(result.get("error", "Unknown error"))
        return

    if timed_out:
        print("Status: TIME LIMIT REACHED")
        print("The result below is the best saved tour before termination.")
    elif result.get("status") == "completed":
        print("Status: COMPLETED")
    else:
        print("Status: PROCESS ENDED WITHOUT NORMAL COMPLETION")

    print(f"Best saved tour length: {result['best_cost']}")
    print(f"Completed LK calls: {result.get('completed_calls', 0)}")

    order = result.get("tour_order")
    if order is not None:
        # Internal IDs start at 0. The repeated first node closes the cycle.
        closed_order = order + [order[0]]
        print("Tour (internal node IDs):")
        print(" -> ".join(map(str, closed_order)))

    print("=" * 55)


if __name__ == "__main__":
    mp.freeze_support()
    main()
