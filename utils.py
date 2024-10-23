import re
from typing import Union
import signal
import time
from sat_solver import OptimizedSATSolver
from cnf_parser import parse_dimacs_cnf

def extract_n_value(filename):
    # Try to extract n value from filename
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    # If no number found, return the filename
    return filename

def custom_sort_key(value: Union[int, str]):
    return (isinstance(value, str), value)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Solver timed out")

def solve_problem(file_path, timeout=10):  # 10 second timeout
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
        print(f"Solver timed out for {file_path}")
        result = None
    finally:
        signal.alarm(0)

    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"Problem: {file_path}, Runtime: {runtime}, Result: {result}")
    return runtime, result is not None  # Return if solved (not timed out)

def solve_problem_wrapper(file_path):
    return solve_problem(file_path)
