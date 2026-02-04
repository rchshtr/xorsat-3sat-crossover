# Reproducibility Guide

This document provides step-by-step instructions to reproduce all results in the paper:

**"The Scale-Dependent Complexity Barrier: Measuring the XOR-SAT to 3-SAT Crossover"**

## Quick Start (5 minutes)

```bash
# Clone repository
git clone https://github.com/rchshtr/xorsat-3sat-crossover.git
cd xorsat-3sat-crossover

# Install dependencies
pip install -r requirements.txt

# Reproduce Table 1 measurements
python code/xorsat_vs_3sat_benchmark.py

# Generate Figure 1
python code/generate_comparison_figure.py

# Run analysis and curve fitting
python code/analyze_xorsat_comparison.py
```

## Expected Results

### Table 1: Energy Scaling Comparison (N=20-30)

After running the benchmark, you should see:

| N | XOR-SAT (ops) | 3-SAT (ops) | Ratio |
|---|---------------|-------------|-------|
| 20 | ~530 | ~41 | 0.08× |
| 25 | ~740 | ~80 | 0.11× |
| 30 | ~1000 | ~150 | 0.15× |

**Note:** Exact values may vary ±10% due to random instance generation, but ratios should be consistent.

### Model Fits

The analysis script will output:
- XOR-SAT: E = 3.97 × N^1.63 (polynomial)
- 3-SAT: E = 3.67 × e^(0.126N) (exponential)
- Crossover point: N ≈ 51

## Detailed Reproduction Steps

### Step 1: Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Benchmarks

```bash
# Full benchmark (takes ~5 minutes)
python code/xorsat_vs_3sat_benchmark.py

# Quick test (10 samples per size)
python code/xorsat_vs_3sat_benchmark.py --samples 10
```

Output is saved to `data/xorsat_vs_3sat_results.json`

### Step 3: Verify Key Measurements

Check the JSON output for these values:

```python
import json
with open('data/xorsat_vs_3sat_results.json') as f:
    data = json.load(f)

# XOR-SAT at N=20 should be ~530 ops
xorsat_n20 = next(x for x in data['xorsat'] if x['N'] == 20)
print(f"XOR-SAT N=20: {xorsat_n20['mean_ops']:.0f} ops")

# 3-SAT at N=20 should be ~41 ops  
sat3_n20 = next(x for x in data['3sat'] if x['N'] == 20)
print(f"3-SAT N=20: {sat3_n20['mean_ops']:.0f} ops")

# Ratio should be ~0.08
ratio = sat3_n20['mean_ops'] / xorsat_n20['mean_ops']
print(f"Ratio: {ratio:.2f}×")
```

### Step 4: Generate Figure

```bash
python code/generate_comparison_figure.py
```

This creates `figures/energy_comparison.png` matching Figure 1 in the paper.

### Step 5: Run Analysis

```bash
python code/analyze_xorsat_comparison.py
```

This outputs:
- Model fit parameters
- Crossover point calculation
- Statistical analysis

## Verification Checklist

After running all steps, verify:

- [ ] `data/xorsat_vs_3sat_results.json` exists with measurements
- [ ] XOR-SAT N=20 ≈ 530 ops (±50)
- [ ] 3-SAT N=20 ≈ 41 ops (±10)
- [ ] Ratio at N=20 ≈ 0.08 (±0.02)
- [ ] Ratio at N=30 ≈ 0.15 (±0.03)
- [ ] `figures/energy_comparison.png` shows crossover pattern
- [ ] Crossover point ≈ N=50 (±5)

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install -r requirements.txt
```

### Different results than paper
- Results may vary ±10% due to random instances
- Ensure you're using UNSAT instances (hardest cases)
- Check that α = 4.26 is being used

### Benchmark takes too long
```bash
# Reduce samples (less accurate but faster)
python code/xorsat_vs_3sat_benchmark.py --samples 5
```

## Data Format

### xorsat_vs_3sat_results.json

```json
{
  "metadata": {
    "timestamp": "...",
    "alpha": 4.26,
    "samples_per_size": 20,
    "sizes": [10, 15, 20, 25, 30, 35]
  },
  "xorsat": [
    {"N": 20, "M": 85, "mean_ops": 529.5, "std_ops": 53.4, ...},
    ...
  ],
  "3sat": [
    {"N": 20, "M": 85, "mean_ops": 41.2, "std_ops": 18.7, ...},
    ...
  ]
}
```

### verified_unsat_results.txt

Contains verified UNSAT instances with backtrack counts:
```
N=20: 41.2 backtracks average (10 instances)
N=22: 61.4 backtracks average (10 instances)
...
```

## Contact

For questions about reproduction:
- Email: rchshtr@entelec.ai
- Issues: https://github.com/rchshtr/xorsat-3sat-crossover/issues
