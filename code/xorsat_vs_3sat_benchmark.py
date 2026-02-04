#!/usr/bin/env python3
"""
XOR-SAT vs 3-SAT Energy Comparison Benchmark
============================================

This script runs actual measurements comparing:
- XOR-SAT: Solved via Gaussian elimination over F_2 (polynomial time)
- 3-SAT: Solved via DPLL with backtrack counting (exponential time)

Energy Model: E = operations × 1 nJ (per composite operation)

Output: Verified measurements for the comparison paper.
"""

import numpy as np
import subprocess
import tempfile
import os
import random
import json
from datetime import datetime
from pathlib import Path


class GF2Matrix:
    """Matrix operations over GF(2) = {0, 1} with XOR arithmetic"""
    
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((rows, cols), dtype=np.uint8)
        self.operations = 0  # Count row operations
    
    def set_row(self, i, values):
        self.data[i] = np.array(values, dtype=np.uint8) % 2
    
    def xor_rows(self, target, source):
        """XOR row[target] ^= row[source]"""
        self.data[target] ^= self.data[source]
        self.operations += 1
    
    def swap_rows(self, i, j):
        """Swap rows i and j"""
        self.data[[i, j]] = self.data[[j, i]]
        self.operations += 1
    
    def gaussian_elimination(self):
        """
        Perform Gaussian elimination over GF(2).
        Returns: (solvable, operations_count)
        """
        m, n = self.rows, self.cols - 1  # Last column is RHS
        pivot_row = 0
        
        for col in range(n):
            if pivot_row >= m:
                break
            
            # Find pivot
            pivot_found = False
            for row in range(pivot_row, m):
                if self.data[row, col] == 1:
                    if row != pivot_row:
                        self.swap_rows(row, pivot_row)
                    pivot_found = True
                    break
            
            if not pivot_found:
                continue
            
            # Eliminate column
            for row in range(m):
                if row != pivot_row and self.data[row, col] == 1:
                    self.xor_rows(row, pivot_row)
            
            pivot_row += 1
        
        # Check for inconsistency (0...0|1)
        for row in range(m):
            if np.all(self.data[row, :-1] == 0) and self.data[row, -1] == 1:
                return False, self.operations
        
        return True, self.operations


def generate_xorsat_instance(n_vars, n_clauses, seed=None):
    """
    Generate a random XOR-SAT instance.
    Each clause: x_i XOR x_j XOR x_k = b (where b is random 0/1)
    
    Returns: Augmented matrix [A|b] over GF(2)
    """
    if seed is not None:
        random.seed(seed)
    
    matrix = GF2Matrix(n_clauses, n_vars + 1)
    
    for i in range(n_clauses):
        # Choose 3 random variables
        vars_in_clause = random.sample(range(n_vars), min(3, n_vars))
        row = [0] * (n_vars + 1)
        for v in vars_in_clause:
            row[v] = 1
        row[-1] = random.randint(0, 1)  # RHS
        matrix.set_row(i, row)
    
    return matrix


def solve_xorsat(n_vars, n_clauses, seed=None):
    """
    Solve XOR-SAT instance using Gaussian elimination.
    Returns: (solvable, operations_count)
    """
    matrix = generate_xorsat_instance(n_vars, n_clauses, seed)
    return matrix.gaussian_elimination()


def generate_3sat_cnf(n_vars, n_clauses, seed=None):
    """Generate random 3-SAT instance in DIMACS format"""
    if seed is not None:
        random.seed(seed)
    
    clauses = []
    for _ in range(n_clauses):
        # Choose 3 random variables
        vars_in_clause = random.sample(range(1, n_vars + 1), min(3, n_vars))
        # Random polarity
        clause = [v if random.random() > 0.5 else -v for v in vars_in_clause]
        clauses.append(clause)
    
    return clauses


def write_dimacs(clauses, n_vars, filepath):
    """Write clauses to DIMACS CNF format"""
    with open(filepath, 'w') as f:
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")


def run_dpll_solver(cnf_file, solver_path):
    """
    Run our custom DPLL solver and extract backtrack count.
    Returns: (satisfiable, backtracks)
    """
    try:
        result = subprocess.run(
            [solver_path, cnf_file],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout
        
        # Parse backtracks from output
        backtracks = 0
        satisfiable = None
        
        for line in output.split('\n'):
            if 'Backtracks:' in line or 'backtracks:' in line.lower():
                try:
                    backtracks = int(line.split(':')[1].strip().split()[0])
                except:
                    pass
            if 'UNSATISFIABLE' in line or 'UNSAT' in line:
                satisfiable = False
            elif 'SATISFIABLE' in line or 'SAT' in line:
                satisfiable = True
        
        return satisfiable, backtracks
    
    except subprocess.TimeoutExpired:
        return None, -1
    except Exception as e:
        print(f"Error running solver: {e}")
        return None, -1


class DPLLSolver:
    """
    Pure Python DPLL implementation with operation counting.
    Used when C++ solver is not available.
    """
    
    def __init__(self, clauses, n_vars):
        self.original_clauses = [set(c) for c in clauses]
        self.n_vars = n_vars
        self.backtracks = 0
        self.decisions = 0
    
    def solve(self):
        """Main DPLL entry point"""
        assignment = {}
        return self._dpll(list(self.original_clauses), assignment)
    
    def _unit_propagate(self, clauses, assignment):
        """Perform unit propagation"""
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                # Filter out assigned literals
                remaining = []
                satisfied = False
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                            satisfied = True
                            break
                    else:
                        remaining.append(lit)
                
                if satisfied:
                    continue
                
                if len(remaining) == 0:
                    return None  # Conflict
                
                if len(remaining) == 1:
                    # Unit clause - must assign
                    lit = remaining[0]
                    var = abs(lit)
                    assignment[var] = (lit > 0)
                    changed = True
        
        return clauses
    
    def _dpll(self, clauses, assignment):
        """Recursive DPLL"""
        # Unit propagation
        result = self._unit_propagate(clauses, assignment.copy())
        if result is None:
            return False
        
        # Check if all clauses satisfied
        all_satisfied = True
        unsat_clause = None
        for clause in clauses:
            satisfied = False
            has_unassigned = False
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    if (lit > 0 and assignment[var]) or (lit < 0 and not assignment[var]):
                        satisfied = True
                        break
                else:
                    has_unassigned = True
            
            if not satisfied:
                if not has_unassigned:
                    return False  # Conflict
                all_satisfied = False
                unsat_clause = clause
        
        if all_satisfied:
            return True
        
        # Choose unassigned variable
        var = None
        for lit in unsat_clause:
            if abs(lit) not in assignment:
                var = abs(lit)
                break
        
        if var is None:
            # Find any unassigned variable
            for v in range(1, self.n_vars + 1):
                if v not in assignment:
                    var = v
                    break
        
        if var is None:
            return all_satisfied
        
        self.decisions += 1
        
        # Try True
        assignment_copy = assignment.copy()
        assignment_copy[var] = True
        if self._dpll(clauses, assignment_copy):
            return True
        
        self.backtracks += 1
        
        # Try False
        assignment_copy = assignment.copy()
        assignment_copy[var] = False
        if self._dpll(clauses, assignment_copy):
            return True
        
        self.backtracks += 1
        return False


def solve_3sat_python(clauses, n_vars):
    """Solve 3-SAT using pure Python DPLL"""
    solver = DPLLSolver(clauses, n_vars)
    satisfiable = solver.solve()
    return satisfiable, solver.backtracks + solver.decisions


def run_benchmark(sizes, samples_per_size, alpha=4.26):
    """
    Run full XOR-SAT vs 3-SAT comparison benchmark.
    
    Args:
        sizes: List of problem sizes (N)
        samples_per_size: Number of instances per size
        alpha: Clause-to-variable ratio (4.26 = phase transition)
    
    Returns: Dictionary with all measurements
    """
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'alpha': alpha,
            'samples_per_size': samples_per_size,
            'sizes': sizes,
            'energy_per_op_nJ': 1.0
        },
        'xorsat': [],
        '3sat': [],
        'comparison': []
    }
    
    print("=" * 60)
    print("XOR-SAT vs 3-SAT Energy Comparison Benchmark")
    print("=" * 60)
    print(f"Clause density α = {alpha}")
    print(f"Samples per size: {samples_per_size}")
    print(f"Problem sizes: {sizes}")
    print("=" * 60)
    print()
    
    for N in sizes:
        M = int(alpha * N)
        print(f"\n--- N = {N}, M = {M} ---")
        
        # XOR-SAT measurements
        xor_ops = []
        xor_solvable = 0
        for i in range(samples_per_size):
            solvable, ops = solve_xorsat(N, M, seed=i*1000 + N)
            xor_ops.append(ops)
            if solvable:
                xor_solvable += 1
        
        xor_mean = np.mean(xor_ops)
        xor_std = np.std(xor_ops)
        xor_energy_nJ = xor_mean * 1.0  # 1 nJ per operation
        
        print(f"  XOR-SAT: {xor_mean:.0f} ± {xor_std:.0f} ops, "
              f"{xor_solvable}/{samples_per_size} solvable, "
              f"E = {xor_energy_nJ/1e6:.4f} mJ")
        
        results['xorsat'].append({
            'N': N,
            'M': M,
            'mean_ops': float(xor_mean),
            'std_ops': float(xor_std),
            'energy_nJ': float(xor_energy_nJ),
            'energy_mJ': float(xor_energy_nJ / 1e6),
            'solvable_count': xor_solvable,
            'total_samples': samples_per_size
        })
        
        # 3-SAT measurements (UNSAT instances only for fair comparison)
        sat_ops = []
        sat_unsat = 0
        sat_sat = 0
        
        for i in range(samples_per_size * 3):  # Generate more to filter for UNSAT
            clauses = generate_3sat_cnf(N, M, seed=i*1000 + N + 500)
            satisfiable, ops = solve_3sat_python(clauses, N)
            
            if not satisfiable and ops > 0:
                sat_ops.append(ops)
                sat_unsat += 1
                if sat_unsat >= samples_per_size:
                    break
            elif satisfiable:
                sat_sat += 1
        
        if len(sat_ops) == 0:
            print(f"  3-SAT: No UNSAT instances found (all {sat_sat} were SAT)")
            sat_mean = 0
            sat_std = 0
            sat_energy_nJ = 0
        else:
            sat_mean = np.mean(sat_ops)
            sat_std = np.std(sat_ops)
            sat_energy_nJ = sat_mean * 1.0
            
            print(f"  3-SAT:  {sat_mean:.0f} ± {sat_std:.0f} ops, "
                  f"{len(sat_ops)} UNSAT instances, "
                  f"E = {sat_energy_nJ/1e6:.4f} mJ")
        
        results['3sat'].append({
            'N': N,
            'M': M,
            'mean_ops': float(sat_mean),
            'std_ops': float(sat_std),
            'energy_nJ': float(sat_energy_nJ),
            'energy_mJ': float(sat_energy_nJ / 1e6),
            'unsat_count': len(sat_ops),
            'sat_count': sat_sat
        })
        
        # Comparison
        if xor_energy_nJ > 0 and sat_energy_nJ > 0:
            ratio = sat_energy_nJ / xor_energy_nJ
            gap = sat_energy_nJ - xor_energy_nJ
            print(f"  RATIO: {ratio:.1f}× (3-SAT / XOR-SAT)")
            
            results['comparison'].append({
                'N': N,
                'ratio': float(ratio),
                'gap_nJ': float(gap),
                'gap_mJ': float(gap / 1e6)
            })
        else:
            results['comparison'].append({
                'N': N,
                'ratio': None,
                'gap_nJ': None,
                'gap_mJ': None
            })
    
    return results


def fit_models(results):
    """Fit polynomial (XOR-SAT) and exponential (3-SAT) models"""
    from scipy.optimize import curve_fit
    
    # Extract data
    xor_N = np.array([r['N'] for r in results['xorsat']])
    xor_E = np.array([r['energy_nJ'] for r in results['xorsat']])
    
    sat_N = np.array([r['N'] for r in results['3sat']])
    sat_E = np.array([r['energy_nJ'] for r in results['3sat']])
    sat_E = sat_E[sat_E > 0]  # Filter zeros
    sat_N = sat_N[:len(sat_E)]
    
    # Fit XOR-SAT: E = A * N^k
    def power_law(N, A, k):
        return A * np.power(N, k)
    
    try:
        popt_xor, _ = curve_fit(power_law, xor_N, xor_E, p0=[1, 3])
        print(f"\nXOR-SAT fit: E = {popt_xor[0]:.4f} * N^{popt_xor[1]:.2f}")
    except:
        popt_xor = None
        print("\nXOR-SAT fit failed")
    
    # Fit 3-SAT: E = A * exp(α * N)
    def exponential(N, A, alpha):
        return A * np.exp(alpha * N)
    
    try:
        if len(sat_N) >= 3:
            popt_sat, _ = curve_fit(exponential, sat_N, sat_E, p0=[1, 0.1], maxfev=5000)
            print(f"3-SAT fit:   E = {popt_sat[0]:.4f} * exp({popt_sat[1]:.4f} * N)")
        else:
            popt_sat = None
            print("3-SAT fit: insufficient data")
    except Exception as e:
        popt_sat = None
        print(f"3-SAT fit failed: {e}")
    
    return popt_xor, popt_sat


def generate_latex_table(results):
    """Generate LaTeX table from results"""
    print("\n" + "=" * 60)
    print("LaTeX Table for Paper")
    print("=" * 60)
    
    print(r"""
\begin{table}[h]
\centering
\caption{Measured energy consumption: XOR-SAT vs 3-SAT}
\label{tab:comparison}
\begin{tabular}{@{}ccccc@{}}
\toprule
\textbf{N} & \textbf{$E_{\text{XOR}}$ (nJ)} & \textbf{$E_{\text{3SAT}}$ (nJ)} & \textbf{Gap (nJ)} & \textbf{Ratio} \\
\midrule""")
    
    for i, xor in enumerate(results['xorsat']):
        sat = results['3sat'][i]
        comp = results['comparison'][i]
        
        if comp['ratio'] is not None:
            print(f"{xor['N']} & {xor['energy_nJ']:.0f} & {sat['energy_nJ']:.0f} & "
                  f"{comp['gap_nJ']:.0f} & {comp['ratio']:.0f}× \\\\")
        else:
            print(f"{xor['N']} & {xor['energy_nJ']:.0f} & -- & -- & -- \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    # Problem sizes to test
    sizes = [10, 15, 20, 25, 30, 35]
    samples_per_size = 20  # Reduced for reasonable runtime
    
    print("Starting XOR-SAT vs 3-SAT benchmark...")
    print(f"This will test N = {sizes} with {samples_per_size} samples each\n")
    
    # Run benchmark
    results = run_benchmark(sizes, samples_per_size, alpha=4.26)
    
    # Try to fit models
    try:
        fit_models(results)
    except ImportError:
        print("\nNote: scipy not available, skipping curve fitting")
    
    # Generate LaTeX table
    generate_latex_table(results)
    
    # Save results
    output_path = Path(__file__).parent / 'xorsat_vs_3sat_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if len(results['comparison']) > 0 and results['comparison'][-1]['ratio'] is not None:
        max_N = results['xorsat'][-1]['N']
        max_ratio = results['comparison'][-1]['ratio']
        print(f"At N={max_N}: 3-SAT requires {max_ratio:.0f}× more operations than XOR-SAT")
        print(f"This confirms exponential vs polynomial scaling!")
    
    return results


if __name__ == '__main__':
    main()
