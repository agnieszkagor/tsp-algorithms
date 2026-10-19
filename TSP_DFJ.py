# Based on an implementation by setyotw
# DFJ for TSP

import time
import multiprocessing as mp
import pyomo.environ as pyo
import tsplib95

# Instance & time limit

INSTANCE_FILE = "fnl4461.tsp"
TIME_LIMIT = 300  # seconds


# DFJ SOLVER

def solve_dfj(instance_file, time_limit, result_queue):

    #Load TSPLIB instance
    problem = tsplib95.load(instance_file)
    nodes = list(problem.get_nodes())

    # Cost matrix
    cost = {
        (i, j): problem.get_weight(i, j)
        for i in nodes
        for j in nodes
        if i != j
    }

    arcs = list(cost.keys())

    # Initialize the model
    model = pyo.ConcreteModel()

    # Initialize decision variables
    model.x = pyo.Var(arcs, within=pyo.Binary)

    # Define objective function
    model.objective = pyo.Objective(
        expr=sum(cost[i, j] * model.x[i, j] for i, j in arcs),
        sense=pyo.minimize
    )

    # Define constraints
    model.constraints = pyo.ConstraintList()

    # Restrict the inbound arc to 1
    for j in nodes:
        model.constraints.add(
            sum(model.x[i, j] for i in nodes if i != j) == 1
        )

    # Restrict the outbound arc to 1
    for i in nodes:
        model.constraints.add(
            sum(model.x[i, j] for j in nodes if i != j) == 1
        )

    # Call the solver - Gurobi
    solver = pyo.SolverFactory("gurobi")
    solver.options["Threads"] = 1

    start = time.perf_counter()
    best_length = None
    optimal = False

    # Iteratively add DFJ subtour elimination constraints
    while True:
        remaining = time_limit - (time.perf_counter() - start)

        if remaining <= 0:
            break

        solver.options["TimeLimit"] = remaining

        results = solver.solve(model, tee=False)

        # Extract selected arcs
        selected = []

        for i, j in arcs:
            value = pyo.value(model.x[i, j], exception=False)

            if value is not None and value > 0.5:
                selected.append((i, j))

        # A complete assignment has one outgoing arc per node
        if len(selected) != len(nodes):
            break

        successor = dict(selected)

        # Find cycles in the current solution
        unvisited = set(nodes)
        cycles = []

        while unvisited:
            current = next(iter(unvisited))
            cycle = []

            while current in unvisited:
                cycle.append(current)
                unvisited.remove(current)
                current = successor[current]

            cycles.append(cycle)

        # A single cycle is a valid TSP tour
        if len(cycles) == 1 and len(cycles[0]) == len(nodes):
            best_length = sum(cost[i, j] for i, j in selected)

            optimal = (
                results.solver.termination_condition
                == pyo.TerminationCondition.optimal
            )
            break

        # Add DFJ constraints for the detected subtours
        for cycle in cycles:
            if len(cycle) < len(nodes):
                model.constraints.add(
                    sum(
                        model.x[i, j]
                        for i in cycle
                        for j in cycle
                        if i != j
                    ) <= len(cycle) - 1
                )

    result_queue.put((best_length, optimal))


# MAIN: GLOBAL TIMEOUT

if __name__ == "__main__":
    start_time = time.perf_counter()

    queue = mp.Queue()
    process = mp.Process(
        target=solve_dfj,
        args=(INSTANCE_FILE, TIME_LIMIT, queue)
    )

    process.start()
    process.join(TIME_LIMIT)

    if process.is_alive():
        process.terminate()
        process.join()
        best_length = None
        optimal = False
    elif not queue.empty():
        best_length, optimal = queue.get()
    else:
        best_length = None
        optimal = False

    runtime = time.perf_counter() - start_time

    # Print results 
    if best_length is None:
        print("Tour length: no feasible tour found")
    elif optimal:
        print(f"Optimal tour length: {best_length:.2f}")
    else:
        print(f"Best tour length (not proven optimal): {best_length:.2f}")

    print(f"Running time: {runtime:.3f} seconds")