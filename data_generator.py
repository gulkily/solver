# begin data_generator.py
 
import random
import time
import os
import subprocess

def random_3sat(n, m, balanced=True, structured=False, distribution='uniform'):
    variables = list(range(1, n + 1))
    clauses = []
    for _ in range(m):
        if distribution == 'uniform':
            clause = random.sample(variables, min(3, n))
        elif distribution == 'power_law':
            clause = [random.choices(variables, weights=[1/i for i in range(1, n+1)])[0] for _ in range(3)]
        elif distribution == 'community':
            community_size = max(3, n // 10)
            community = random.randint(0, (n - 1) // community_size)
            start = community * community_size + 1
            end = min(start + community_size, n + 1)
            clause = random.sample(range(start, end), min(3, end - start))
        
        if structured:
            clause = [random.choice([-1, 1]) * x for x in clause]
        else:
            clause = [x if random.random() < 0.5 else -x for x in clause]
        
        if balanced and len(clause) < 3:
            clause += [-x for x in clause[:3-len(clause)]]
        elif len(clause) < 3:
            clause += random.sample(variables, 3 - len(clause))
        
        clauses.append(clause)
    return clauses

def generate_dimacs_cnf(n, m, problem_type='random', balanced=True, structured=False, distribution='uniform'):
    clauses = random_3sat(n, m, balanced, structured, distribution)
    
    lines = [f"c DIMACS CNF file for 3-SAT problem"]
    lines.append(f"c Generated on {time.ctime()}")
    lines.append(f"c Parameters: n={n}, m={m}, type={problem_type}, balanced={balanced}, structured={structured}, distribution={distribution}")
    lines.append(f"p cnf {n} {m}")
    
    for clause in clauses:
        lines.append(" ".join(map(str, clause)) + " 0")
    
    return "\n".join(lines)

def generate_dataset(max_n, num_runs, problem_types, balanced, structured, distributions):
    random_seed = int(time.time())
    random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    n_values = [1, 2, 3] + list(range(5, max_n + 1, max(1, max_n // 25)))
    if n_values[-1] != max_n:
        n_values.append(max_n)

    os.makedirs("dimacs_cnf_files", exist_ok=True)
    
    for n in n_values:
        for problem_type in problem_types:
            m = n if problem_type == 'random' else int(n * 4.3)
            for distribution in distributions:
                for run in range(num_runs):
                    cnf_content = generate_dimacs_cnf(n, m, problem_type, balanced, structured, distribution)
                    filename = f"dimacs_cnf_files/sat_problem_n{n}_{problem_type}_{distribution}_run{run}.cnf"
                    with open(filename, "w") as f:
                        f.write(cnf_content)
                    print(f"Generated {filename}")

    return "dimacs_cnf_files"

if __name__ == "__main__":
    max_n = 200
    num_runs = 10
    problem_types = ['random', 'structured']
    balanced = True
    structured = False
    distributions = ['uniform', 'power_law', 'community']
    
    data_directory = generate_dataset(max_n, num_runs, problem_types, balanced, structured, distributions)
    
    # Call the solver script
    subprocess.run(["python", "solver.py", data_directory])

# end data_generator.py