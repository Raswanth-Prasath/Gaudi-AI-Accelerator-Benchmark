"""
=============================================================
BERT Fine-tuning on SST-2 Dataset - Intel Gaudi Benchmark
=============================================================
This script fine-tunes a pre-trained BERT model on the SST-2
sentiment classification dataset using Intel Gaudi HPU with
EAGER MODE enabled.

Eager Mode: Operations execute immediately as they are called,
            similar to standard PyTorch behavior.
=============================================================
Author: Raswanth
Date: December 2025
Hardware: Intel Gaudi AI Accelerator (HPU)
Mode: Eager
=============================================================
"""

# =============================================================
# 1. Environment Setup for Gaudi Eager Mode
# =============================================================
import os

# CRITICAL: Disable Lazy Mode for Eager execution
os.environ["PT_HPU_LAZY_MODE"] = "0"

# =============================================================
# 2. Importing Libraries
# =============================================================
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Gaudi-specific imports
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu as hpu

# =============================================================
# 3. Configuration
# =============================================================
CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "seed": 42,
    "device": "hpu",
    "mode": "eager"
}

# =============================================================
# 4. Set Random Seeds for Reproducibility
# =============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.hpu.is_available():
        import habana_frameworks.torch.hpu.random as hpu_random
        hpu_random.manual_seed(seed)

set_seed(CONFIG["seed"])

# =============================================================
# 5. Device Configuration
# =============================================================
if not torch.hpu.is_available():
    raise RuntimeError("Intel Gaudi HPU is not available. "
                       "Please ensure you are running on a Gaudi node.")

device = torch.device(CONFIG["device"])
print(f"{'='*60}")
print(f"BERT SST-2 Benchmark - Intel Gaudi HPU (EAGER MODE)")
print(f"{'='*60}")
print(f"Using device: {device}")
print(f"Eager Mode: ENABLED (PT_HPU_LAZY_MODE=0)")
print(f"{'='*60}\n")

# =============================================================
# 6. Load and Preprocess SST-2 Dataset
# =============================================================
print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])

def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=CONFIG["max_length"]
    )

print("Tokenizing dataset...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Create DataLoaders
train_loader = DataLoader(
    tokenized_datasets["train"],
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=4
)

print(f"Training samples: {len(tokenized_datasets['train'])}")
print(f"Validation samples: {len(tokenized_datasets['validation'])}")

# =============================================================
# 7. Load Pre-trained BERT Model
# =============================================================
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=2
)
model.to(device)

# =============================================================
# 8. Optimizer and Scheduler Setup
# =============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
total_steps = len(train_loader) * CONFIG["num_epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# =============================================================
# 9. Benchmark Metrics Storage
# =============================================================
benchmark_results = {
    "platform": "Intel Gaudi HPU",
    "mode": "eager",
    "model": CONFIG["model_name"],
    "batch_size": CONFIG["batch_size"],
    "epochs": CONFIG["num_epochs"],
    "training_times": [],
    "epoch_times": [],
    "throughput_samples_per_sec": [],
    "latency_per_batch_ms": [],
    "validation_accuracy": [],
    "total_training_time": 0,
    "final_accuracy": 0
}

# =============================================================
# 10. Training Loop with Benchmarking (Eager Mode)
# =============================================================
print("\n" + "="*60)
print("Starting Training (EAGER MODE)...")
print("="*60 + "\n")

total_start_time = time.time()

for epoch in range(CONFIG["num_epochs"]):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()
    batch_times = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")

    for batch in pbar:
        batch_start = time.time()

        # Move batch to HPU device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # NOTE: In eager mode, mark_step() is not required but can be
        # used optionally for synchronization points
        # htcore.mark_step()  # Optional in eager mode

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Synchronize for accurate timing
        hpu.synchronize()
        batch_end = time.time()

        batch_time = (batch_end - batch_start) * 1000  # Convert to ms
        batch_times.append(batch_time)

        running_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{running_loss / (len(batch_times)):.4f}',
            'batch_ms': f'{batch_time:.1f}'
        })

    epoch_time = time.time() - epoch_start_time
    avg_batch_time = np.mean(batch_times)
    throughput = (len(train_loader) * CONFIG["batch_size"]) / epoch_time

    benchmark_results["epoch_times"].append(epoch_time)
    benchmark_results["throughput_samples_per_sec"].append(throughput)
    benchmark_results["latency_per_batch_ms"].append(avg_batch_time)

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  - Time: {epoch_time:.2f}s")
    print(f"  - Avg Loss: {running_loss / len(train_loader):.4f}")
    print(f"  - Throughput: {throughput:.2f} samples/sec")
    print(f"  - Avg Batch Latency: {avg_batch_time:.2f} ms")

    # Validation after each epoch
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    benchmark_results["validation_accuracy"].append(accuracy)
    print(f"  - Validation Accuracy: {accuracy * 100:.2f}%\n")

total_training_time = time.time() - total_start_time
benchmark_results["total_training_time"] = total_training_time
benchmark_results["final_accuracy"] = benchmark_results["validation_accuracy"][-1]

# =============================================================
# 11. Inference Latency Benchmark
# =============================================================
print("="*60)
print("Measuring Inference Latency (EAGER MODE)...")
print("="*60)

model.eval()
inference_times = []
num_inference_runs = 100

# Warmup
with torch.no_grad():
    sample_batch = next(iter(val_loader))
    sample_input_ids = sample_batch["input_ids"].to(device)
    sample_attention_mask = sample_batch["attention_mask"].to(device)

    for _ in range(10):
        _ = model(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask
        )
        hpu.synchronize()

# Actual benchmark
with torch.no_grad():
    for _ in tqdm(range(num_inference_runs), desc="Inference benchmark"):
        start = time.time()
        _ = model(
            input_ids=sample_input_ids,
            attention_mask=sample_attention_mask
        )
        hpu.synchronize()
        inference_times.append((time.time() - start) * 1000)

avg_inference_latency = np.mean(inference_times)
p50_latency = np.percentile(inference_times, 50)
p95_latency = np.percentile(inference_times, 95)
p99_latency = np.percentile(inference_times, 99)

benchmark_results["inference_latency"] = {
    "avg_ms": avg_inference_latency,
    "p50_ms": p50_latency,
    "p95_ms": p95_latency,
    "p99_ms": p99_latency
}

print(f"\nInference Latency (batch_size={CONFIG['batch_size']}):")
print(f"  - Average: {avg_inference_latency:.2f} ms")
print(f"  - P50: {p50_latency:.2f} ms")
print(f"  - P95: {p95_latency:.2f} ms")
print(f"  - P99: {p99_latency:.2f} ms")

# =============================================================
# 12. Final Results Summary
# =============================================================
print("\n" + "="*60)
print("BENCHMARK RESULTS - Intel Gaudi HPU (EAGER MODE)")
print("="*60)
print(f"Model: {CONFIG['model_name']}")
print(f"Batch Size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"\nPerformance Metrics:")
print(f"  - Total Training Time: {total_training_time:.2f} seconds")
print(f"  - Avg Epoch Time: {np.mean(benchmark_results['epoch_times']):.2f} seconds")
print(f"  - Avg Throughput: {np.mean(benchmark_results['throughput_samples_per_sec']):.2f} samples/sec")
print(f"  - Avg Batch Latency: {np.mean(benchmark_results['latency_per_batch_ms']):.2f} ms")
print(f"\nAccuracy:")
print(f"  - Final Validation Accuracy: {benchmark_results['final_accuracy'] * 100:.2f}%")
print(f"\nInference Latency:")
print(f"  - Average: {avg_inference_latency:.2f} ms")
print(f"  - P95: {p95_latency:.2f} ms")
print("="*60)

# =============================================================
# 13. Save Results to JSON
# =============================================================
results_file = "benchmark_results_gaudi_eager.json"
with open(results_file, "w") as f:
    json.dump(benchmark_results, f, indent=2)
print(f"\nResults saved to {results_file}")

# Save model
model.save_pretrained("bert_sst2_gaudi_eager_model")
tokenizer.save_pretrained("bert_sst2_gaudi_eager_model")
print("Model saved to bert_sst2_gaudi_eager_model/")
print("\nBenchmark Complete!")
