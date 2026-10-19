# Based on an implementation by jarvis73
# ACO for TSP

import math
import random
import time
import multiprocessing as mp
import queue
import numpy as np


# CONFIGURATION

INSTANCE_FILE = "fnl4461.tsp"

NUM_ANTS = 50
MAX_ITERATIONS = 200
ALPHA = 1.0
BETA = 5.0
RHO = 0.5
TIME_LIMIT = 300

# Dense matrices require substantial memory for large instances.
MAX_NODES = 5000


# TSPLIB READER

def read_tsplib(filename):
    with open(filename, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    headers = {}
    sections = {}
    current_section = None

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        upper = line.upper()

        if upper == "EOF":
            break

        if upper.endswith("_SECTION"):
            current_section = upper
            sections[current_section] = []
            continue

        if current_section is not None:
            sections[current_section].append(line)
            continue

        if ":" in line:
            key, value = line.split(":", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts

        headers[key.strip().upper()] = value.strip()

    n = int(headers["DIMENSION"])
    edge_type = headers.get("EDGE_WEIGHT_TYPE", "").upper()
    edge_format = headers.get("EDGE_WEIGHT_FORMAT", "").upper()
    problem_type = headers.get("TYPE", "TSP").upper()

    if problem_type == "ATSP":
        raise ValueError(
            "ATSP instances are not supported by this symmetric-TSP implementation."
        )

    if n > MAX_NODES:
        raise ValueError(
            f"This dense ACO implementation supports at most "
            f"{MAX_NODES} vertices. The instance has {n}."
        )

    ids = list(range(1, n + 1))

    if edge_type == "EXPLICIT":
        tokens = []
        for line in sections.get("EDGE_WEIGHT_SECTION", []):
            tokens.extend(line.split())

        values = np.array([float(x) for x in tokens], dtype=np.float32)
        matrix = np.zeros((n, n), dtype=np.float32)

        def fill_symmetric(i, j, value):
            matrix[i, j] = value
            matrix[j, i] = value

        k = 0

        if edge_format == "FULL_MATRIX":
            if len(values) < n * n:
                raise ValueError("Incomplete FULL_MATRIX data.")
            matrix = values[:n * n].reshape((n, n)).copy()

        elif edge_format in ("LOWER_DIAG_ROW", "UPPER_DIAG_ROW",
                             "LOWER_ROW", "UPPER_ROW",
                             "LOWER_DIAG_COL", "UPPER_DIAG_COL",
                             "LOWER_COL", "UPPER_COL"):

            if edge_format == "LOWER_DIAG_ROW":
                pairs = ((i, j) for i in range(n) for j in range(i + 1))
            elif edge_format == "UPPER_DIAG_ROW":
                pairs = ((i, j) for i in range(n) for j in range(i, n))
            elif edge_format == "LOWER_ROW":
                pairs = ((i, j) for i in range(n) for j in range(i))
            elif edge_format == "UPPER_ROW":
                pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
            elif edge_format == "LOWER_DIAG_COL":
                pairs = ((i, j) for j in range(n) for i in range(j, n))
            elif edge_format == "UPPER_DIAG_COL":
                pairs = ((i, j) for j in range(n) for i in range(j + 1))
            elif edge_format == "LOWER_COL":
                pairs = ((i, j) for j in range(n) for i in range(j + 1, n))
            else:
                pairs = ((i, j) for j in range(n) for i in range(j))

            for i, j in pairs:
                if k >= len(values):
                    raise ValueError("Incomplete explicit edge-weight data.")
                fill_symmetric(i, j, values[k])
                k += 1
        else:
            raise ValueError(
                f"Unsupported EDGE_WEIGHT_FORMAT: {edge_format}"
            )

        return ids, matrix

    if edge_type not in (
        "EUC_2D", "CEIL_2D", "ATT", "GEO",
        "MAN_2D", "MAX_2D", "EUC_3D",
        "CEIL_3D", "MAN_3D", "MAX_3D"
    ):
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {edge_type}. "
            "SPECIAL and custom distance functions are not implemented."
        )

    coord_lines = sections.get("NODE_COORD_SECTION", [])
    coordinates = {}

    for line in coord_lines:
        parts = line.split()
        if len(parts) >= 3:
            node_id = int(parts[0])
            coordinates[node_id] = [float(x) for x in parts[1:]]

    ids = sorted(coordinates)

    if len(ids) != n:
        raise ValueError(
            f"Expected {n} coordinates, found {len(ids)}."
        )

    coords = np.array([coordinates[i] for i in ids], dtype=np.float64)

    if edge_type == "GEO":
        if coords.shape[1] < 2:
            raise ValueError("GEO requires two coordinates per vertex.")

        def geo_rad(value):
            degrees = int(value)
            minutes = value - degrees
            return math.pi * (degrees + 5.0 * minutes / 3.0) / 180.0

        geo = np.array([
            [geo_rad(value) for value in row[:2]]
            for row in coords
        ])

        matrix = np.zeros((n, n), dtype=np.float32)
        radius = 6378.388

        for i in range(n):
            for j in range(i + 1, n):
                lat1, lon1 = geo[i]
                lat2, lon2 = geo[j]

                q1 = math.cos(lon1 - lon2)
                q2 = math.cos(lat1 - lat2)
                q3 = math.cos(lat1 + lat2)

                arg = 0.5 * (
                    (1.0 + q1) * q2 - (1.0 - q1) * q3
                )
                arg = max(-1.0, min(1.0, arg))
                distance = int(radius * math.acos(arg) + 1.0)

                matrix[i, j] = distance
                matrix[j, i] = distance

        return ids, matrix

    dimensions = 3 if edge_type.endswith("_3D") else 2
    if coords.shape[1] < dimensions:
        raise ValueError(
            f"{edge_type} requires {dimensions} coordinates per vertex."
        )

    coords = coords[:, :dimensions]
    delta = coords[:, None, :] - coords[None, :, :]
    absolute = np.abs(delta)

    if edge_type.startswith("MAX"):
        distances = np.max(absolute, axis=2)
        matrix = np.floor(distances + 0.5)

    elif edge_type.startswith("MAN"):
        distances = np.sum(absolute, axis=2)
        matrix = np.floor(distances + 0.5)

    else:
        squared = np.sum(delta * delta, axis=2)

        if edge_type == "ATT":
            raw = np.sqrt(squared / 10.0)
            rounded = np.floor(raw + 0.5)
            matrix = np.where(rounded < raw, rounded + 1, rounded)

        else:
            distances = np.sqrt(squared)

            if edge_type.startswith("CEIL"):
                matrix = np.ceil(distances)
            else:
                # TSPLIB EUC_2D / EUC_3D rounding
                matrix = np.floor(distances + 0.5)

    return ids, matrix.astype(np.float32)


# ANT SYSTEM (AS)

def nearest_neighbor_cost(distance):
    n = len(distance)
    unseen = set(range(1, n))
    path = [0]
    current = 0
    total = 0.0

    while unseen:
        next_node = min(
            unseen, key=lambda node: distance[current, node]
        )
        total += float(distance[current, next_node])
        path.append(next_node)
        unseen.remove(next_node)
        current = next_node

    total += float(distance[current, path[0]])
    return total


def choose_next_node(current, unseen, pheromone, heuristic):
    unseen_array = np.asarray(unseen, dtype=np.int64)

    probabilities = (
        pheromone[current, unseen_array] ** ALPHA
        * heuristic[current, unseen_array] ** BETA
    )

    probabilities = np.nan_to_num(
        probabilities, nan=0.0, posinf=0.0, neginf=0.0
    )

    total = float(probabilities.sum())

    if total <= 0.0:
        return random.choice(unseen)

    probabilities /= total

    return random.choices(
        unseen, weights=probabilities.tolist(), k=1
    )[0]


def publish_best(result_queue, cost, path, iteration):
    message = ("best", float(cost), list(path), iteration)

    try:
        result_queue.put_nowait(message)
    except queue.Full:
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            result_queue.put_nowait(message)
        except queue.Full:
            pass


def run_aco(instance_file, result_queue):
    try:
        ids, distance = read_tsplib(instance_file)
        n = len(ids)

        if n < 3:
            raise ValueError("The instance must contain at least 3 vertices.")

        algorithm_start = time.perf_counter()

        nn_cost = nearest_neighbor_cost(distance)
        if nn_cost <= 0:
            raise ValueError("The nearest-neighbor tour has invalid cost.")

        pheromone_init = NUM_ANTS / nn_cost

        pheromone = np.full(
            (n, n), pheromone_init, dtype=np.float32
        )

        heuristic = np.divide(
            1.0,
            distance,
            out=np.zeros_like(distance, dtype=np.float32),
            where=distance > 0
        )

        best_cost = float("inf")
        best_path = None
        best_iteration = 0

        for iteration in range(1, MAX_ITERATIONS + 1):
            all_paths = []
            all_costs = []

            # Each ant constructs a complete tour.
            for _ in range(NUM_ANTS):
                start = random.randrange(n)
                path = [start]
                unseen = list(range(n))
                unseen.remove(start)
                current = start

                while unseen:
                    next_node = choose_next_node(
                        current, unseen, pheromone, heuristic
                    )

                    path.append(next_node)
                    unseen.remove(next_node)
                    current = next_node

                path_array = np.asarray(path, dtype=np.int64)
                next_array = np.roll(path_array, -1)

                tour_cost = float(
                    distance[path_array, next_array].sum()
                )

                all_paths.append(path)
                all_costs.append(tour_cost)

                if tour_cost < best_cost:
                    best_cost = tour_cost
                    best_path = path.copy()
                    best_iteration = iteration

                    publish_best(
                        result_queue, best_cost,
                        best_path, best_iteration
                    )

            # Global pheromone update, following Ant System.
            deposition = np.zeros_like(pheromone)

            for path, tour_cost in zip(all_paths, all_costs):
                if tour_cost <= 0:
                    continue

                path_array = np.asarray(path, dtype=np.int64)
                next_array = np.roll(path_array, -1)

                deposition[path_array, next_array] += 1.0 / tour_cost

            pheromone = (1.0 - RHO) * pheromone + deposition

        runtime = time.perf_counter() - algorithm_start

        result_queue.put((
            "done", best_cost, best_path,
            best_iteration, runtime, ids
        ))

    except Exception as exc:
        try:
            result_queue.put(("error", str(exc)))
        except Exception:
            pass


# MAIN: TIME LIMIT AND FINAL OUTPUT ONLY

def main():
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)

    process = context.Process(
        target=run_aco,
        args=(INSTANCE_FILE, result_queue)
    )

    wall_start = time.perf_counter()
    process.start()

    best = None
    final_result = None
    error_message = None

    while process.is_alive():
        elapsed = time.perf_counter() - wall_start

        if elapsed >= TIME_LIMIT:
            process.terminate()
            process.join()
            break

        try:
            message = result_queue.get(
                timeout=min(0.2, TIME_LIMIT - elapsed)
            )
        except queue.Empty:
            continue

        if message[0] == "best":
            best = message

        elif message[0] == "done":
            final_result = message
            best = ("best", message[1], message[2], message[3])

        elif message[0] == "error":
            error_message = message[1]
            break

    process.join(timeout=1)

    # Read any final message that remains in the queue.
    while True:
        try:
            message = result_queue.get_nowait()
        except queue.Empty:
            break

        if message[0] == "best":
            best = message
        elif message[0] == "done":
            final_result = message
            best = ("best", message[1], message[2], message[3])
        elif message[0] == "error":
            error_message = message[1]

    elapsed = time.perf_counter() - wall_start

    if error_message:
        print(f"ERROR: {error_message}")
        print(f"Runtime: {elapsed:.3f} seconds")

    elif final_result is not None:
        _, cost, path, iteration, runtime, ids = final_result
        tour = [ids[i] for i in path]
        tour.append(tour[0])

        print(f"Instance: {INSTANCE_FILE}")
        print(f"Tour cost: {cost:.2f}")
        print(f"Best iteration: {iteration}")
        print(f"Tour: {tour}")
        print(f"Runtime: {runtime:.3f} seconds")

    elif best is not None:
        _, cost, path, iteration = best
        print(f"Instance: {INSTANCE_FILE}")
        print("Status: TIME LIMIT REACHED")
        print(f"Best tour cost found: {cost:.2f}")
        print(f"Best iteration: {iteration}")
        print(f"Best tour (internal vertex indices): {path}")
        print(f"Runtime: {elapsed:.3f} seconds")

    else:
        print(f"Instance: {INSTANCE_FILE}")
        print("Status: TIME LIMIT REACHED; no complete tour reported.")
        print(f"Runtime: {elapsed:.3f} seconds")

    result_queue.close()


if __name__ == "__main__":
    main()
