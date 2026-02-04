.PHONY: all install benchmark analyze figure clean verify

# Default: run everything
all: install benchmark analyze figure

# Install dependencies
install:
	pip install -r requirements.txt

# Run benchmarks (generates data/xorsat_vs_3sat_results.json)
benchmark:
	python code/xorsat_vs_3sat_benchmark.py

# Quick benchmark (fewer samples)
benchmark-quick:
	python code/xorsat_vs_3sat_benchmark.py --samples 5

# Run analysis and curve fitting
analyze:
	python code/analyze_xorsat_comparison.py

# Generate figure
figure:
	python code/generate_comparison_figure.py

# Verify reproduction matches paper
verify:
	@echo "=== Verification Test ==="
	@python3 -c "\
import json; \
data = json.load(open('data/xorsat_vs_3sat_results.json')); \
xor20 = next(x for x in data['xorsat'] if x['N'] == 20)['mean_ops']; \
sat20 = next(x for x in data['3sat'] if x['N'] == 20)['mean_ops']; \
ratio = sat20/xor20; \
print(f'XOR-SAT N=20: {xor20:.0f} ops (expected ~530)'); \
print(f'3-SAT N=20: {sat20:.0f} ops (expected ~41)'); \
print(f'Ratio: {ratio:.2f}× (expected ~0.08)'); \
ok = 400 < xor20 < 650 and 30 < sat20 < 70 and 0.05 < ratio < 0.15; \
print('✓ PASS' if ok else '✗ FAIL')"

# Clean generated files
clean:
	rm -f data/xorsat_vs_3sat_results.json
	rm -f figures/energy_comparison.png
