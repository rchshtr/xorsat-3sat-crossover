#!/usr/bin/env python3
"""
Analyze the actual comparison between XOR-SAT and 3-SAT operations.
Uses real measured data from verified_unsat_results.txt.
"""

import numpy as np

# Real 3-SAT data from verified_unsat_results.txt (backtracks)
sat_data = {
    20: 41.2,
    22: 61.4,
    24: 67.4,
    26: 94.4,
    28: 142.4,
    30: 149.8
}

# XOR-SAT: Gaussian elimination over GF(2)
# For N vars with M clauses, actual operations depend on matrix rank
# At alpha=4.26, the matrix is overdetermined (M > N)
# Operations ≈ N² for rank computation + back-substitution

def xorsat_ops_estimate(N):
    """
    Estimate XOR-SAT operations for Gaussian elimination.
    For N variables with M = 4.26N clauses over GF(2):
    - Create augmented matrix: M × (N+1)
    - Gaussian elimination: O(M × N) operations
    - Each row operation involves XORing N+1 elements
    - But in GF(2), this is very fast (bit operations)
    
    Counting "composite operations" similar to 3-SAT:
    - Each pivot search: ~M comparisons
    - Each row elimination: ~M XORs
    - Total: ~N pivots × M = O(N²) at α=4.26
    """
    M = int(4.26 * N)
    # Conservative: count each row XOR as 1 operation
    # Worst case: N pivots, each eliminates up to M-1 rows
    # Average case with early termination: ~N × N/2 = N²/2
    return N * N / 2


def xorsat_ops_measured(N):
    """
    Use actual measured operations from our benchmark.
    From xorsat_vs_3sat_benchmark.py output:
    N=10: 172 ops, N=15: 332 ops, N=20: 530 ops
    N=25: 740 ops, N=30: 1000 ops, N=35: 1309 ops
    
    Fits E = 3.97 * N^1.63 (between N² and N³)
    """
    measured = {10: 172, 15: 332, 20: 530, 25: 740, 30: 1000, 35: 1309}
    if N in measured:
        return measured[N]
    else:
        # Interpolate using power law
        return 3.97 * (N ** 1.63)


print("=" * 70)
print("HONEST ANALYSIS: XOR-SAT vs 3-SAT Operation Counts")
print("=" * 70)
print()

print("REAL 3-SAT DATA (from verified_unsat_results.txt):")
print("  Instances: UNSAT, verified with minisat")
print("  Solver: Custom DPLL with backtrack counting")
print("  Constraint density: α = 4.26 (phase transition)")
print()

print("XOR-SAT DATA (from xorsat_vs_3sat_benchmark.py):")
print("  Algorithm: Gaussian elimination over GF(2)")
print("  Same constraint density: α = 4.26")
print()

print("-" * 70)
print(f"{'N':>4} | {'XOR-SAT (ops)':>14} | {'3-SAT (ops)':>14} | {'Ratio':>10} | {'Winner':>10}")
print("-" * 70)

for N in sorted(sat_data.keys()):
    xor_ops = xorsat_ops_measured(N)
    sat_ops = sat_data[N]
    ratio = sat_ops / xor_ops
    winner = "3-SAT" if ratio < 1 else "XOR-SAT"
    print(f"{N:4} | {xor_ops:14.0f} | {sat_ops:14.1f} | {ratio:10.3f}x | {winner:>10}")

print("-" * 70)
print()

print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print()
print("1. At N=20-30, 3-SAT uses FEWER operations than XOR-SAT!")
print("   - Ratio ranges from 0.08x to 0.15x (3-SAT is 6-12x FASTER)")
print()
print("2. This contradicts the paper's claim of 10^6x gap at N=35")
print()
print("3. WHY? Several reasons:")
print("   a) XOR-SAT phase transition is at α=1, not α=4.26")
print("      - At α=4.26, XOR-SAT is massively over-constrained")
print("      - Most instances are UNSAT and detected quickly")
print("   b) Our DPLL solver with unit propagation is efficient")
print("      - Unit propagation prunes search space dramatically")
print("      - UNSAT instances are proven with few backtracks")
print("   c) Small problem sizes (N≤30) don't show exponential blowup")
print("      - The 'hard' regime for 3-SAT starts around N>50")
print()
print("4. HONEST CONCLUSION:")
print("   - The energy gap claim in the paper is NOT SUPPORTED by data")
print("   - We should either:")
print("      A) Remove the 10^6x claim entirely, OR")
print("      B) Generate much harder instances at larger N, OR")
print("      C) Reframe as theoretical scaling (not measured)")
print()

# Fit exponential to 3-SAT data
N_vals = np.array(list(sat_data.keys()))
E_vals = np.array(list(sat_data.values()))

from scipy.optimize import curve_fit

def exponential(N, A, beta):
    return A * np.exp(beta * N)

def polynomial(N, A, k):
    return A * (N ** k)

try:
    popt_exp, _ = curve_fit(exponential, N_vals, E_vals, p0=[1, 0.1])
    popt_poly, _ = curve_fit(polynomial, N_vals, E_vals, p0=[1, 2])
    
    print("=" * 70)
    print("MODEL FITTING")
    print("=" * 70)
    print()
    print(f"3-SAT exponential fit: E = {popt_exp[0]:.4f} × exp({popt_exp[1]:.4f} × N)")
    print(f"3-SAT polynomial fit:  E = {popt_poly[0]:.4f} × N^{popt_poly[1]:.2f}")
    print()
    
    # Compute residuals
    exp_pred = exponential(N_vals, *popt_exp)
    poly_pred = polynomial(N_vals, *popt_poly)
    
    rss_exp = np.sum((E_vals - exp_pred) ** 2)
    rss_poly = np.sum((E_vals - poly_pred) ** 2)
    
    print(f"Residual sum of squares (exponential): {rss_exp:.2f}")
    print(f"Residual sum of squares (polynomial):  {rss_poly:.2f}")
    print()
    
    if rss_exp < rss_poly:
        print("Exponential fits better, but...")
    else:
        print("Polynomial fits better!")
    
    print(f"With only {len(N_vals)} data points over a narrow range,")
    print("we cannot confidently distinguish exponential from polynomial scaling.")
    
except Exception as e:
    print(f"Fitting failed: {e}")

print()
print("=" * 70)
print("RECOMMENDATION FOR PAPER")
print("=" * 70)
print()
print("The current paper claims:")
print('  "At N=35, 3-SAT requires 10^6x more energy than XOR-SAT"')
print()
print("This is FALSE based on our actual measurements.")
print()
print("Options:")
print("  1. HONEST VERSION: Report actual measured ratios (~0.1x to 0.2x)")
print("  2. THEORETICAL VERSION: Discuss asymptotic scaling, not measured gap")
print("  3. DIFFERENT COMPARISON: Compare SAT vs UNSAT at same N (harder)")
print("  4. LARGER N: Run benchmarks at N=50, 60, 70 where gap may emerge")
print()
