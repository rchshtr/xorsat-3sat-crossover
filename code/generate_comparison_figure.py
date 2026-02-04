#!/usr/bin/env python3
"""
Generate crossover point figure for XOR-SAT vs 3-SAT paper.
Shows the three phases: 3-SAT cheaper, crossover, exponential blowup.
"""

import matplotlib.pyplot as plt
import numpy as np

# Actual measured data
N_measured = np.array([20, 22, 24, 26, 28, 30])

# XOR-SAT: Gaussian elimination (from xorsat_vs_3sat_benchmark.py)
xor_measured = np.array([530, 612, 706, 804, 907, 1000])

# 3-SAT: DPLL with backtrack counting (from verified_unsat_results.txt)
sat_measured = np.array([41.2, 61.4, 67.4, 94.4, 142.4, 149.8])

# Model fits
def xor_model(N):
    return 3.97 * N**1.63

def sat_model(N):
    return 3.67 * np.exp(0.126 * N)

# Extended range for prediction
N_extended = np.linspace(10, 200, 400)
xor_predicted = xor_model(N_extended)
sat_predicted = sat_model(N_extended)

# Find crossover point
crossover_idx = np.argmin(np.abs(sat_predicted - xor_predicted))
crossover_N = N_extended[crossover_idx]

# Create figure with three-phase visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Background shading for phases
ax.axvspan(10, 50, alpha=0.15, color='green', label='Phase I: 3-SAT Cheaper')
ax.axvspan(50, 70, alpha=0.15, color='yellow', label='Phase II: Crossover')
ax.axvspan(70, 200, alpha=0.15, color='red', label='Phase III: Exponential Blowup')

# Plot curves
ax.semilogy(N_extended, xor_predicted, 'b-', linewidth=2.5, 
             label=f'XOR-SAT (P): $O(N^{{1.63}})$')
ax.semilogy(N_extended, sat_predicted, 'r-', linewidth=2.5, 
             label=f'3-SAT (NP): $O(e^{{0.126N}})$')

# Mark measured data
ax.scatter(N_measured, xor_measured, s=80, c='blue', marker='o', zorder=5, edgecolors='black')
ax.scatter(N_measured, sat_measured, s=80, c='red', marker='s', zorder=5, edgecolors='black')

# Mark crossover
ax.axvline(x=crossover_N, color='green', linestyle='--', linewidth=2, alpha=0.8)
ax.annotate(f'CROSSOVER\n$N \\approx {crossover_N:.0f}$', 
             xy=(crossover_N, sat_predicted[crossover_idx]), 
             xytext=(crossover_N + 15, sat_predicted[crossover_idx] * 0.5),
             fontsize=11, ha='left', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Mark key milestones with gap ratios
milestones = [
    (20, '0.08×', 'green'),
    (80, '17×', 'orange'),
    (150, '42,000×', 'red'),
    (200, '$10^7$×', 'darkred')
]
for n, label, color in milestones:
    ratio_point = sat_model(n)
    ax.scatter([n], [ratio_point], s=100, c=color, zorder=6, edgecolors='black', linewidth=2)
    offset = 3 if n < 100 else 0.3
    ax.annotate(f'{label}', xy=(n, ratio_point), 
                 xytext=(n+5, ratio_point*offset), fontsize=10, fontweight='bold', color=color)

# Labels and title
ax.set_xlabel('Problem Size (N)', fontsize=13)
ax.set_ylabel('Operations (log scale)', fontsize=13)
ax.set_title('The Crossover Point: Scale-Dependent Complexity Barrier\n'
             '3-SAT is CHEAPER at small N, then EXPONENTIALLY HARDER at large N', 
             fontsize=13, fontweight='bold')

# Legend
ax.legend(loc='upper left', fontsize=10)

# Grid and limits
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(10, 205)
ax.set_ylim(10, 1e13)

# Add phase labels at top
ax.text(30, 5e12, 'Phase I\n3-SAT Wins', ha='center', fontsize=11, fontweight='bold', color='darkgreen')
ax.text(60, 5e12, 'Crossover', ha='center', fontsize=11, fontweight='bold', color='olive')
ax.text(135, 5e12, 'Phase III: Exponential Blowup', ha='center', fontsize=11, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('energy_comparison.png', dpi=300, bbox_inches='tight')
print("Generated: energy_comparison.png")

# Print summary
print(f"\n=== CROSSOVER ANALYSIS ===")
print(f"Crossover point: N ≈ {crossover_N:.0f}")
print(f"\nPhase I (3-SAT cheaper):")
print(f"  N=20: ratio = {sat_model(20)/xor_model(20):.2f}×")
print(f"  N=40: ratio = {sat_model(40)/xor_model(40):.2f}×")
print(f"\nPhase II (crossover):")
print(f"  N=50: ratio = {sat_model(50)/xor_model(50):.2f}×")
print(f"  N=60: ratio = {sat_model(60)/xor_model(60):.1f}×")
print(f"\nPhase III (exponential blowup):")
print(f"  N=80: ratio = {sat_model(80)/xor_model(80):.0f}×")
print(f"  N=100: ratio = {sat_model(100)/xor_model(100):.0f}×")
print(f"  N=150: ratio = {sat_model(150)/xor_model(150):,.0f}×")
print(f"  N=200: ratio = {sat_model(200)/xor_model(200):,.0f}×")
