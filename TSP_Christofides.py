# Based on an implementation by dsrahul30
# Christofides algorithm for TSP

import multiprocessing
import time
import math
import itertools
import traceback

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from munkres import Munkres


# SETTINGS

INSTANCE_FILE = "fnl4461.tsp"
TIME_LIMIT = 300  # seconds


# READ TSPLIB INSTANCE

def read_tsplib(filename):
    header = {}
    coordinates = {}
    edge_weights = []
    section = None

    with open(filename, "r", encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line == "EOF":
                if line == "EOF":
                    break
                continue

            if line.endswith("_SECTION"):
                section = line
                continue

            if section == "NODE_COORD_SECTION":
                parts = line.split()
                coordinates[int(parts[0])] = tuple(
                    float(value) for value in parts[1:]
                )

            elif section == "EDGE_WEIGHT_SECTION":
                edge_weights.extend(float(x) for x in line.split())

            elif section is None:
                if ":" in line:
                    key, value = line.split(":", 1)
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) != 2:
                        continue
                    key, value = parts

                header[key.strip().upper()] = value.strip()

    dimension = int(header["DIMENSION"])
    problem_type = header.get("TYPE", "TSP").upper()
    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "").upper()
    edge_weight_format = header.get("EDGE_WEIGHT_FORMAT", "").upper()

    if problem_type != "TSP":
        raise ValueError(
            f"Unsupported problem TYPE: {problem_type}. "
            "Christofides implementation requires symmetric TSP."
        )

    return (
        coordinates,
        dimension,
        edge_weight_type,
        edge_weight_format,
        edge_weights
    )


# CALCULATE DISTANCE

def calculate_distance(coord1, coord2, edge_weight_type):
    x1, y1, *rest1 = coord1
    x2, y2, *rest2 = coord2

    if edge_weight_type == "GEO":
        pi = 3.141592
        radius = 6378.388

        def geo_to_radians(value):
            degrees = int(value)
            minutes = value - degrees
            return pi * (degrees + 5.0 * minutes / 3.0) / 180.0

        latitude1 = geo_to_radians(x1)
        longitude1 = geo_to_radians(y1)
        latitude2 = geo_to_radians(x2)
        longitude2 = geo_to_radians(y2)

        q1 = math.cos(longitude1 - longitude2)
        q2 = math.cos(latitude1 - latitude2)
        q3 = math.cos(latitude1 + latitude2)

        value = radius * math.acos(
            max(-1.0, min(1.0, 0.5 * (
                (1.0 + q1) * q2 - (1.0 - q1) * q3
            )))
        )

        return int(value + 1.0)

    differences = [abs(x1 - x2), abs(y1 - y2)]

    if rest1 and rest2:
        differences.append(abs(rest1[0] - rest2[0]))

    euclidean = math.sqrt(sum(
        (a - b) ** 2 for a, b in zip(coord1, coord2)
    ))

    if edge_weight_type in ("EUC_2D", "EUC_3D"):
        return int(euclidean + 0.5)

    elif edge_weight_type in ("CEIL_2D", "CEIL_3D"):
        return math.ceil(euclidean)

    elif edge_weight_type in ("MAN_2D", "MAN_3D"):
        return int(sum(differences) + 0.5)

    elif edge_weight_type in ("MAX_2D", "MAX_3D"):
        return int(max(differences) + 0.5)

    elif edge_weight_type == "ATT":
        rij = math.sqrt(
            ((x1 - x2) ** 2 + (y1 - y2) ** 2) / 10.0
        )
        tij = int(rij)
        return tij + 1 if tij < rij else tij

    else:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}"
        )


# CREATE DISTANCE MATRIX FROM COORDINATES

def create_distance_matrix(coordinates, edge_weight_type):
    node_ids = sorted(coordinates)
    n = len(node_ids)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            distance = calculate_distance(
                coordinates[node_ids[i]],
                coordinates[node_ids[j]],
                edge_weight_type
            )
            matrix[i][j] = distance
            matrix[j][i] = distance

    return matrix


# CREATE EXPLICIT DISTANCE MATRIX

def create_explicit_matrix(edge_weights, dimension, edge_weight_format):
    n = dimension
    matrix = np.zeros((n, n), dtype=float)
    index = 0

    if edge_weight_format == "FULL_MATRIX":
        expected = n * n
        if len(edge_weights) != expected:
            raise ValueError(
                f"Expected {expected} edge weights, "
                f"but found {len(edge_weights)}."
            )

        matrix = np.array(edge_weights, dtype=float).reshape((n, n))

        if not np.allclose(matrix, matrix.T):
            raise ValueError(
                "FULL_MATRIX is not symmetric. "
                "Christofides requires symmetric distances."
            )

        return matrix

    formats = {
        "LOWER_DIAG_ROW": ("row", "lower", True),
        "UPPER_DIAG_ROW": ("row", "upper", True),
        "LOWER_ROW": ("row", "lower", False),
        "UPPER_ROW": ("row", "upper", False),
        "LOWER_DIAG_COL": ("col", "lower", True),
        "UPPER_DIAG_COL": ("col", "upper", True),
        "LOWER_COL": ("col", "lower", False),
        "UPPER_COL": ("col", "upper", False),
    }

    if edge_weight_format not in formats:
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}"
        )

    order, triangle, diagonal = formats[edge_weight_format]

    if order == "row":
        for i in range(n):
            if triangle == "lower":
                js = range(i + 1) if diagonal else range(i)
            else:
                js = range(i, n) if diagonal else range(i + 1, n)

            for j in js:
                if index >= len(edge_weights):
                    raise ValueError("Not enough edge weights.")

                value = edge_weights[index]
                matrix[i][j] = value
                matrix[j][i] = value
                index += 1

    else:
        for j in range(n):
            if triangle == "lower":
                iset = range(j, n) if diagonal else range(j + 1, n)
            else:
                iset = range(j + 1) if diagonal else range(j)

            for i in iset:
                if index >= len(edge_weights):
                    raise ValueError("Not enough edge weights.")

                value = edge_weights[index]
                matrix[i][j] = value
                matrix[j][i] = value
                index += 1

    if index != len(edge_weights):
        raise ValueError(
            f"Expected {index} edge weights, "
            f"but found {len(edge_weights)}."
        )

    return matrix


# ORIGINAL CHRISTOFIDES FUNCTIONS

def _csr_gen_triples(A):
    """Converts a SciPy sparse matrix into weighted edge triples."""
    graph = nx.Graph()
    nrows = A.shape[0]
    data, indices, indptr = A.data, A.indices, A.indptr

    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            graph.add_edge(i, indices[j], weight=data[j])

    return list(graph.edges(data="weight"))


def _odd_vertices_of_MST(M, number_of_nodes):
    """Returns the vertices having odd degree in the MST."""
    odd_vertices = [0 for _ in range(number_of_nodes)]

    for u, v, d in M:
        odd_vertices[u] += 1
        odd_vertices[v] += 1

    return [
        vertex for vertex, degree in enumerate(odd_vertices)
        if degree % 2 == 1
    ]


def Munkres_cost(indexes, bipartite_graph):
    """Returns the cost of the edges in the assignment."""
    cost = 0

    for index in indexes:
        cost += bipartite_graph[index[0]][index[1]]

    return cost


def min_Munkres(M, bipartite_graphs):
    """Finds the minimum-cost assignment among the bipartitions."""
    munkres = Munkres()
    minimum = np.inf
    min_index = None
    min_munkres_indexes = None

    for index, bipartite_graph in enumerate(bipartite_graphs[0]):
        indexes = munkres.compute(bipartite_graph)
        cost = Munkres_cost(indexes, bipartite_graph)

        if min_index is None or cost < minimum:
            minimum = cost
            min_index = index
            min_munkres_indexes = indexes

    if min_munkres_indexes is None or not np.isfinite(minimum):
        raise ValueError("Could not find a finite perfect matching.")

    result = []

    for row, col in min_munkres_indexes:
        result.append([
            bipartite_graphs[1][min_index][0][row],
            bipartite_graphs[1][min_index][1][col]
        ])

    return result


def bipartite_Graph(M, bipartite_set, odd_vertices):
    """Builds the bipartite cost matrices used by Munkres."""
    bipartite_graphs = []
    vertex_sets = []

    for vertex_set1 in bipartite_set:
        vertex_set1 = list(sorted(vertex_set1))
        vertex_set2 = [
            vertex for vertex in odd_vertices
            if vertex not in vertex_set1
        ]

        matrix = [
            [np.inf for _ in range(len(vertex_set2))]
            for _ in range(len(vertex_set1))
        ]

        for i in range(len(vertex_set1)):
            for j in range(len(vertex_set2)):
                u = vertex_set1[i]
                v = vertex_set2[j]

                if u < v:
                    matrix[i][j] = M[u][v]
                else:
                    matrix[i][j] = M[v][u]

        bipartite_graphs.append(matrix)
        vertex_sets.append([vertex_set1, vertex_set2])

    return [bipartite_graphs, vertex_sets]


def create_Multigraph(M, MST, indexes, odd_vertices):
    """Creates a MultiGraph from the MST and perfect matching."""
    multigraph = nx.MultiGraph()

    for u, v, d in MST:
        multigraph.add_edge(u, v, weight=d)

    for pair in indexes:
        multigraph.add_edge(
            pair[0], pair[1],
            weight=M[pair[0]][pair[1]]
        )

    return multigraph


def Euler_Tour(multigraph):
    """Uses Fleury's algorithm to find an Euler tour."""
    tour = []
    temp_graph = nx.MultiGraph()
    graph_nodes = list(nx.nodes(multigraph))
    current_node = graph_nodes[0]
    tour.append(current_node)

    while nx.number_of_edges(multigraph) > 0:
        for edge in multigraph.edges(current_node):
            temp_graph = multigraph.copy()
            temp_graph.remove_edge(edge[0], edge[1], key=None)

            if nx.is_connected(temp_graph):
                tour.append(edge[1])
                current_node = edge[1]
                multigraph.remove_edge(edge[0], edge[1], key=None)
                break
        else:
            tour.append(edge[1])
            current_node = edge[1]
            multigraph.remove_edge(edge[0], edge[1], key=None)
            multigraph.remove_nodes_from(list(nx.isolates(multigraph)))

    return tour


def shortcut_Euler_Tour(tour):
    """Shortcuts repeated vertices in the Euler tour."""
    result = []

    for vertex in tour:
        if vertex not in result:
            result.append(vertex)

    result.append(tour[0])
    return result


def cost(christofides_tour, M):
    """Returns the cost of a tour."""
    travel_cost = 0
    christofides_tour = christofides_tour[:-1]
    previous_vertex = christofides_tour[-1]

    for current_vertex in christofides_tour:
        if previous_vertex > current_vertex:
            travel_cost += M[current_vertex][previous_vertex]
        else:
            travel_cost += M[previous_vertex][current_vertex]

        previous_vertex = current_vertex

    return travel_cost



def compute(M):
    """Runs the Christofides algorithm using NetworkX matching."""

    # Step 1: Compute a minimum spanning tree.
    mst = _csr_gen_triples(
        minimum_spanning_tree(csr_matrix(M))
    )

    # Step 2: Find vertices with odd degree in the MST.
    odd_vertices = _odd_vertices_of_MST(mst, M.shape[0])

    # Step 3: Build a complete graph on the odd-degree vertices.
    matching_graph = nx.Graph()
    matching_graph.add_nodes_from(odd_vertices)

    for i, u in enumerate(odd_vertices):
        for v in odd_vertices[i + 1:]:
            matching_graph.add_edge(
                u, v, weight=float(M[u, v])
            )

    # Step 4: Find a minimum-weight perfect matching.
    matching = nx.min_weight_matching(
        matching_graph, weight="weight"
    )

    indexes = [
        list(sorted(edge)) for edge in matching
    ]
    indexes.sort()

    if len(indexes) * 2 != len(odd_vertices):
        raise ValueError(
            "The matching is not a perfect matching."
        )

    # Step 5: Combine the MST and the matching.
    multigraph = create_Multigraph(
        M, mst, indexes, odd_vertices
    )
    saved_multigraph = multigraph.copy()

    # Step 6: Find an Eulerian tour.
    euler_tour = Euler_Tour(multigraph)

    # Step 7: Shortcut repeated vertices.
    solution = shortcut_Euler_Tour(euler_tour)
    travel_cost = cost(solution, M)

    return {
        "Christofides_Solution": solution,
        "Travel_Cost": travel_cost,
        "MST": mst,
        "Odd_Vertices": odd_vertices,
        "Indexes": indexes,
        "Multigraph": list(
            saved_multigraph.edges(data="weight")
        ),
        "Euler_Tour": euler_tour
    }

# WORKER: READ INSTANCE AND RUN ALGORITHM

def run_worker(filename, connection):
    try:
        (
            coordinates,
            dimension,
            edge_weight_type,
            edge_weight_format,
            edge_weights
        ) = read_tsplib(filename)

        if edge_weight_type == "EXPLICIT":
            matrix = create_explicit_matrix(
                edge_weights, dimension, edge_weight_format
            )
        else:
            if len(coordinates) != dimension:
                raise ValueError(
                    f"Expected {dimension} coordinates, "
                    f"but found {len(coordinates)}."
                )

            matrix = create_distance_matrix(
                coordinates, edge_weight_type
            )

        result = compute(matrix)

        mst_cost = sum(weight for _, _, weight in result["MST"])
        matching_cost = sum(
            matrix[u][v] for u, v in result["Indexes"]
        )

        node_ids = sorted(coordinates) if coordinates else list(
            range(1, dimension + 1)
        )

        display_tour = [
            node_ids[index]
            for index in result["Christofides_Solution"]
        ]

        connection.send({
            "status": "OK",
            "dimension": dimension,
            "edge_weight_type": edge_weight_type,
            "edge_weight_format": edge_weight_format,
            "tour": display_tour,
            "cost": result["Travel_Cost"],
            "mst_cost": mst_cost,
            "matching_cost": matching_cost,
            "odd_vertices_count": len(result["Odd_Vertices"])
        })

    except Exception:
        connection.send({
            "status": "ERROR",
            "error": traceback.format_exc()
        })

    finally:
        connection.close()


# MAIN WITH HARD TIME LIMIT

if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("Instance:", INSTANCE_FILE)
    print("Maximum runtime:", TIME_LIMIT, "seconds")
    print("Running Christofides algorithm...")

    parent_connection, child_connection = multiprocessing.Pipe(
        duplex=False
    )

    process = multiprocessing.Process(
        target=run_worker,
        args=(INSTANCE_FILE, child_connection)
    )

    start_time = time.perf_counter()
    process.start()
    child_connection.close()

    if parent_connection.poll(TIME_LIMIT):
        result = parent_connection.recv()
        process.join()
        elapsed = time.perf_counter() - start_time

        print("\nRuntime: {:.6f} seconds".format(elapsed))

        if result["status"] == "ERROR":
            print("\nERROR:")
            print(result["error"])

        else:
            print("\n========== RESULTS ==========")
            print("Instance:", INSTANCE_FILE)
            print("Number of vertices:", result["dimension"])
            print("Edge weight type:", result["edge_weight_type"])
            print("Edge weight format:", result["edge_weight_format"])
            print("Christofides tour:", result["tour"])
            print("Tour cost:", result["cost"])
            print("MST cost:", result["mst_cost"])
            print("Matching cost:", result["matching_cost"])
            print(
                "Number of odd vertices:",
                result["odd_vertices_count"]
            )

    else:
        process.terminate()
        process.join()
        elapsed = time.perf_counter() - start_time

        print("\nSTATUS: TIME LIMIT REACHED")
        print("Maximum runtime: {} seconds".format(TIME_LIMIT))
        print("Elapsed time: {:.6f} seconds".format(elapsed))
        print("No completed solution was returned.")

    parent_connection.close()
