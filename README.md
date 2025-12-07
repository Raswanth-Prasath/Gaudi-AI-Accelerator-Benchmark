# BERT SST-2 Benchmark: A100 vs Intel Gaudi

This repository contains benchmark scripts for comparing NVIDIA A100 GPU and Intel Gaudi AI Accelerator performance on BERT fine-tuning for sentiment classification.

## Quick Start

### 1. Run A100 Benchmark
```bash
cd <path>
sbatch run_a100.sh
```

### 2. Run Gaudi Benchmarks
```bash
cd <path>

# Lazy mode (recommended)
sbatch run_gaudi_lazy.sh

# Eager mode
sbatch run_gaudi_eager.sh
```

### 3. Compare Results
```bash
cd <path>
python compare_benchmarks.py
```


## Gaudi Execution Modes

- **Lazy Mode**: Graph-based execution for optimal performance. Requires `htcore.mark_step()`.
- **Eager Mode**: Immediate execution like standard PyTorch. Easier debugging.


## Output Files

After running, results are saved as:
- `benchmark_results_a100.json`
- `benchmark_results_gaudi_lazy.json`
- `benchmark_results_gaudi_eager.json`
