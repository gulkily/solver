# begin solver.py # marker line, do not remove this comment

"""
This script implements an optimized SAT (Boolean Satisfiability) solver and runs experiments on DIMACS CNF files.

Key components:

1. OptimizedSATSolver class:
   - Implements a CDCL (Conflict-Driven Clause Learning) SAT solver
   - Uses techniques like watched literals, unit propagation, and conflict analysis
   - Employs heuristics for variable ordering and clause learning

2. SATVerifier class:
   - Verifies the correctness of SAT solver solutions

3. DIMACS CNF file parsing:
   - Reads and processes DIMACS CNF format files
   - Extracts variables and clauses from the input

4. Parallel experiment runner:
   - Solves multiple SAT problems in parallel using multiprocessing
   - Handles timeouts to prevent infinite loops
   - Collects and aggregates results

5. Results processing and visualization:
   - Calculates average runtimes and solve rates for different problem sizes
   - Saves results to a JSON file
   - Generates graphs using an external script (graph_generator.py)

6. Logging:
   - Implements detailed logging for debugging and performance analysis

Usage:
python solver.py <data_directory>

The script processes all .cnf files in the specified directory, runs the SAT solver on each,
and produces a summary of results including average runtimes and solve rates for different
problem sizes. It's designed to be used for benchmarking and analyzing the performance of
the SAT solver across various problem instances.

This implementation is particularly useful for computer science students and professors
studying algorithm design, constraint satisfaction problems, and parallel computing.
It provides a practical example of how theoretical concepts in SAT solving can be
implemented and evaluated in a real-world setting.

Dependencies:
- Python 3.x
- tqdm (for progress bars)
- matplotlib (for graph generation, used in graph_generator.py)

Note: This script uses multiprocessing and signal handling, which may have platform-specific
behavior. It has been primarily tested on Unix-like systems.
"""

import sys
import json
import time
import multiprocessing
from collections import defaultdict, deque
import subprocess
import os
import re
from typing import Union
from tqdm import tqdm
import signal
import logging

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SATVerifier:
    """
    Verifies the correctness of SAT solver solutions.
    """
    def __init__(self, variables, clauses, assignment):
        self.variables = variables
        self.clauses = clauses
        self.assignment = assignment

    def verify_solution(self):
        """
        Verifies if the assignment satisfies all clauses.
        Returns: (bool, str) - (is_valid, error_message)
        """
        # Check if all variables are assigned
        for var in self.variables:
            if var not in self.assignment:
                return False, f"Variable {var} is not assigned"

        # Check if all clauses are satisfied
        for i, clause in enumerate(self.clauses):
            if not self._is_clause_satisfied(clause):
                return False, f"Clause {i} ({clause}) is not satisfied"

        return True, "Solution is valid"

    def _is_clause_satisfied(self, clause):
        """
        Checks if a single clause is satisfied by the assignment.
        """
        return any(
            (lit > 0 and self.assignment[abs(lit)]) or
            (lit < 0 and not self.assignment[abs(lit)])
            for lit in clause
        )

class OptimizedSATSolver:
    def __init__(self):
        self.variables = []
        self.clauses = []
        self.assignment = {}
        self.variable_order = []
        self.clause_data = defaultdict(list)
        self.watched_literals = defaultdict(list)
        self.score_decay = 0.95
        self.variable_scores = defaultdict(int)
        self.activity_scores = defaultdict(int)
        self.learned_clauses = set()

    def preprocess(self):
        unit_clauses = [c for c in self.clauses if len(c) == 1]
        for clause in unit_clauses:
            lit = clause[0]
            self.assignment[abs(lit)] = lit > 0
            self.variable_scores[abs(lit)] += 1

        for clause in self.clauses:
            for lit in clause:
                self.variable_scores[abs(lit)] += 1
                self.activity_scores[abs(lit)] += 1

        self.variable_order = sorted(self.variables, 
                                     key=lambda v: self.variable_scores[v] / (self.activity_scores[v] + 1), 
                                     reverse=True)

        for i, clause in enumerate(self.clauses):
            if len(clause) >= 2:
                self.watched_literals[clause[0]].append(i)
                self.watched_literals[clause[1]].append(i)

    def update_watched_literals(self, false_lit):
        impacted_clauses = list(self.watched_literals[-false_lit])
        for clause_idx in impacted_clauses:
            clause = self.clauses[clause_idx]
            other_watch = next((lit for lit in clause[:2] if lit != -false_lit), None)
            
            if other_watch is None or self.is_satisfied(clause):
                continue

            for lit in clause[2:]:
                if abs(lit) not in self.assignment:
                    self.watched_literals[lit].append(clause_idx)
                    self.watched_literals[-false_lit].remove(clause_idx)
                    clause[0], clause[1] = other_watch, lit
                    break
            else:
                if other_watch not in self.assignment:
                    self.assignment[abs(other_watch)] = other_watch > 0
                    if not self.unit_propagation(other_watch):
                        return False
                elif self.assignment[abs(other_watch)] != (other_watch > 0):
                    return False
        return True

    def unit_propagation(self, lit):
        propagation_queue = deque([lit])
        while propagation_queue:
            current_lit = propagation_queue.popleft()
            if not self.update_watched_literals(current_lit):
                return False
            for clause_idx in self.watched_literals[current_lit]:
                clause = self.clauses[clause_idx]
                if self.is_satisfied(clause):
                    continue
                unassigned = [l for l in clause if abs(l) not in self.assignment]
                if not unassigned:
                    return False
                if len(unassigned) == 1:
                    new_lit = unassigned[0]
                    self.assignment[abs(new_lit)] = new_lit > 0
                    propagation_queue.append(new_lit)
        return True

    def is_satisfied(self, clause):
        return any((lit > 0 and self.assignment.get(abs(lit), False)) or
                   (lit < 0 and not self.assignment.get(abs(lit), True))
                   for lit in clause)

    def solve(self):
        self.preprocess()
        result = self._backtrack()
        
        if result:
            # Verify the solution
            verifier = SATVerifier(self.variables, self.clauses, self.assignment)
            is_valid, message = verifier.verify_solution()
            
            if not is_valid:
                logging.error(f"Solution verification failed: {message}")
                return False
            
            logging.info("Solution verified successfully")
        
        return result

    def _backtrack(self):
        stack = []
        current_level = 0
        max_iterations = 10000000  # Limit iterations to prevent infinite loops

        for iteration in range(max_iterations):
            logging.debug(f"Iteration {iteration}, Current level: {current_level}, Stack size: {len(stack)}, Assignments: {len(self.assignment)}")

            if len(self.assignment) == len(self.variables):
                logging.info("Solution found")
                return True

            if current_level >= len(stack):
                unassigned = [v for v in self.variable_order if v not in self.assignment]
                if not unassigned:
                    logging.warning("No unassigned variables left, but solution not found.")
                    return False
                var = unassigned[0]
                stack.append((var, False))
                logging.debug(f"Trying new variable: {var}")

            assert current_level >= 0, f"current_level became negative: {current_level}"

            var, value = stack[current_level]

            if value and var in self.assignment:
                del self.assignment[var]
                stack.pop()
                current_level -= 1
                logging.debug(f"Backtracking: var={var}, new level={current_level}")
                continue

            self.assignment[var] = value
            logging.debug(f"Assigning: var={var}, value={value}")

            if self.unit_propagation(var if value else -var):
                current_level += 1
                logging.debug(f"Propagation successful, new level={current_level}")
            else:
                del self.assignment[var]
                conflict_clause = self._analyze_conflict(stack[:current_level+1])
                if conflict_clause:
                    self.learn_clause(conflict_clause)
                    backjump_level = self._find_backjump_level(conflict_clause, current_level)
                    logging.debug(f"Conflict detected. Learned clause: {conflict_clause}. Backjumping to level {backjump_level}")
                    while current_level > backjump_level:
                        var_to_unassign = stack.pop()[0]
                        if var_to_unassign in self.assignment:
                            del self.assignment[var_to_unassign]
                        current_level -= 1
                else:
                    if value:
                        stack.pop()
                        current_level -= 1
                        logging.debug(f"Both values tried for var={var}, backtracking to level={current_level}")
                    else:
                        stack[current_level] = (var, True)
                        logging.debug(f"Trying opposite value for var={var}")

        logging.warning(f"Reached maximum iterations ({max_iterations}) without finding a solution.")
        return False

    def _analyze_conflict(self, partial_assignment):
        # Implement conflict analysis here
        # This is a placeholder implementation
        return tuple(sorted(-lit for var, val in partial_assignment for lit in [var, -var] if val != (lit > 0)))

    def _find_backjump_level(self, conflict_clause, current_level):
        # Implement backjumping logic here
        # This is a placeholder implementation
        return max(0, current_level - 1)

    def learn_clause(self, clause):
        if clause not in self.learned_clauses:
            self.learned_clauses.add(clause)
            self.clauses.append(list(clause))
            if len(clause) >= 2:
                self.watched_literals[clause[0]].append(len(self.clauses) - 1)
                self.watched_literals[clause[1]].append(len(self.clauses) - 1)

def parse_dimacs_cnf(file_path):
    variables = set()
    clauses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('c'):
                continue
            if line.startswith('p'):
                _, _, num_vars, _ = line.split()
                variables = set(range(1, int(num_vars) + 1))
            else:
                clause = [int(x) for x in line.split()[:-1]]
                if not clause:
                    logging.warning(f"Empty clause found in {file_path}")
                    continue
                clauses.append(clause)
                variables.update(abs(x) for x in clause)
    if not variables or not clauses:
        raise ValueError(f"Invalid DIMACS CNF file: {file_path}")
    return list(variables), clauses

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Solver timed out")

def solve_problem(file_path, timeout=150):  # 150 second timeout
    variables, clauses = parse_dimacs_cnf(file_path)
    solver = OptimizedSATSolver()
    solver.variables = variables
    solver.clauses = clauses

    start_time = time.time()
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = solver.solve()
    except TimeoutException:
        logging.warning(f"Solver timed out for {file_path}")
        result = None
    finally:
        signal.alarm(0)

    end_time = time.time()
    runtime = end_time - start_time
    
    logging.info(f"Problem: {file_path}, Runtime: {runtime}, Result: {result}")
    return runtime, result is not None  # Return if solved (not timed out)

def solve_problem_wrapper(file_path):
    return solve_problem(file_path)

def extract_n_value(filename):
    # Try to extract n value from filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    # If no number found, return the filename
    return filename

def custom_sort_key(value: Union[int, str]):
    return (isinstance(value, str), value)

def solve_and_update(args):
    file_path, progress_queue = args
    try:
        runtime, solved = solve_problem(file_path)
        progress_queue.put((file_path, runtime, solved))
        return file_path, runtime, solved
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())  # Log the full stack trace
        progress_queue.put((file_path, 0, False))
        return file_path, 0, False

def run_experiments_parallel(data_directory, num_processes=None):
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    cnf_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.cnf')]
    cnf_files.sort(key=lambda x: custom_sort_key(extract_n_value(os.path.basename(x))))

    results = defaultdict(list)
    total_files = len(cnf_files)

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=total_files, desc="Processing files") as pbar:
            async_results = [pool.apply_async(solve_and_update, ((file_path, progress_queue),)) for file_path in cnf_files]

            completed = 0
            while completed < total_files:
                file_path, runtime, solved = progress_queue.get()
                completed += 1
                try:
                    n = extract_n_value(os.path.basename(file_path))
                    results[n].append((runtime, solved))
                    pbar.update(1)
                    pbar.set_postfix({
                        "Current file": os.path.basename(file_path),
                        "Runtime": f"{runtime:.2f}s",
                        "Solved": solved
                    })
                except Exception as e:
                    logging.error(f"Error processing results for {file_path}: {str(e)}")

            # Ensure all processes are done
            for res in async_results:
                res.get()

    n_values = sorted(results.keys(), key=custom_sort_key)
    average_runtimes = []
    solve_rates = []

    logging.info("\nResults summary:")
    for n in n_values:
        n_results = results[n]
        if n_results:
            runtimes, solved = zip(*n_results)
            avg_runtime = sum(runtimes) / len(n_results)
            solve_rate = sum(solved) / len(n_results)
            average_runtimes.append(avg_runtime)
            solve_rates.append(solve_rate)
            logging.info(f"n={n}: Avg runtime={avg_runtime:.6f}s, Solve rate={solve_rate:.2%}")
        else:
            logging.warning(f"n={n}: No valid results")
            average_runtimes.append(0)
            solve_rates.append(0)

    return n_values, average_runtimes, solve_rates

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python solver.py <data_directory>")
        sys.exit(1)

    data_directory = sys.argv[1]
    num_processes = multiprocessing.cpu_count()
    
    logging.info(f"\nRunning parallel experiments for DIMACS CNF files using {num_processes} processes:")
    n_values, runtimes, solve_rates = run_experiments_parallel(data_directory, num_processes)

    results = {
        "n_values": n_values,
        "runtimes": runtimes,
        "solve_rates": solve_rates,
        "num_processes": num_processes,
        "data_directory": data_directory
    }
    
    results_file = f"solver_results_dimacs_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(results, f)
    logging.info(f"Results saved to {results_file}")

    subprocess.run(["python", "graph_generator.py", results_file])

# Helper function to extract variables and clauses from a DIMACS CNF file
def extract_variables_and_clauses(file_path):
    variables = set()
    clauses = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('c'):
                continue
            if line.startswith('p'):
                _, _, num_vars, _ = line.split()
                variables = set(range(1, int(num_vars) + 1))
            else:
                clause = [int(x) for x in line.split()[:-1]]
                clauses.append(clause)
                variables.update(abs(x) for x in clause)
    return list(variables), clauses

# end solver.py # marker line, do not remove this comment 
