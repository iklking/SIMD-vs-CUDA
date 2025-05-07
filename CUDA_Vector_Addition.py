import cupy as cp
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

vector_results = []

# Perform vector addition
def run_vector_addition(N):
  # Create random vectors
  A_cpu = np.random.rand(N).astype(np.float32)
  B_cpu = np.random.rand(N).astype(np.float32)
  A_gpu = cp.asarray(A_cpu)
  B_gpu = cp.asarray(B_cpu)
  
  # CPU timing
  start_cpu = time.perf_counter()
  C_cpu = np.add(A_cpu, B_cpu)
  end_cpu = time.perf_counter()
  cpu_time = (end_cpu - start_cpu) * 1000
  
  # GPU timing
  cp.cuda.Device(0).synchronize()
  start_gpu = time.perf_counter()
  C_gpu = cp.add(A_gpu, B_gpu)
  cp.cuda.Device(0).synchronize()
  end_gpu = time.perf_counter()
  gpu_time = (end_gpu - start_gpu) * 1000
  
  # Verify correctness
  C_gpu_cpu = cp.asnumpy(C_gpu)
  error = np.max(np.abs(C_cpu - C_gpu_cpu))
  
   # Print results
  print(f"N = {N} | CPU Time: {cpu_time:.6f} ms | CUDA Time: {gpu_time:.6f} ms | Max Error: {error:.6f}")
  
  vector_results.append({
    "N": N,
    "CPU Time (ms)": cpu_time,
    "CUDA Time (ms)": gpu_time,
    "Max Error": error
    })
  
# Run experiments
for exp in range(10, 21, 2):
  N = 2 ** exp
  run_vector_addition(N)
