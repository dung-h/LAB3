# LAB 3: Estimating π using Monte Carlo Method

## Team Members
- Lê Nguyễn Kim Khôi (2311671)
- Hồ Anh Dũng (2310543)
- Nguyễn Thái Học (2311100)
- Nguyễn Thiện Minh (2312097)
- Huỳnh Đức Nhân (2312420)

## Overview
This project implements and compares different approaches to estimating the value of π using the Monte Carlo method. The Monte Carlo method works by generating random points within a square and counting how many fall within an inscribed circle. The ratio of points inside the circle to the total points approximates π/4.

## Approaches Implemented

### Approach 1: Single-Threaded Implementation
- Implemented in `main.c` as the `approach1()` function
- Generates random points sequentially
- Uses custom 64 bit random number generator for high-quality randomness

### Approach 2: Multi-Threaded with Local Counters
- Implemented in `main.c` as `multi_threaded_approach()` with `use_shared_counter = 0`
- Divides the workload among N threads
- Each thread maintains its own local counter
- Results are combined at the end to calculate π

### Approach 3: Multi-Threaded with Shared Counter
- Implemented in `main.c` as `multi_threaded_approach()` with `use_shared_counter = 1`
- Threads share a global counter protected by a mutex
- Each point found inside the circle requires a mutex lock/unlock operation

### Approach 4: Enhanced Shared Counter (Batched)
- Implemented in `solution for approach3.c`
- Reduces mutex contention by using a batched counter approach
- Threads update the shared counter in batches instead of for each point
- Significantly improves performance over Approach 3 for high thread counts

## Project Structure
- `main.c` - Contains implementations of Approaches 1, 2, and 3
- `solution for approach3.c` - Contains the enhanced solution for Approach 3 (batched)
- `pi.csv` - Results from all approaches (timing, π estimates, and error)
- `plot_generator.py` - Python script for generating performance visualizations
- `plots/` - Directory containing generated charts and visualizations
- `results for solution of approach 3.csv` - Results specifically for the batched approach

## Compilation and Execution
```bash
# Compile the main implementation
gcc main.c -o main -lm -lpthread

# Run the main implementation
./main > pi.csv

# Compile the enhanced Approach 3
gcc "solution for approach3.c" -o approach3_solution -lm -lpthread

# Run the enhanced Approach 3
./approach3_solution

# Generate visualization plots
python plot_generator.py
```

## Performance Analysis
The project analyzes performance across different approaches by varying:
- Number of points (1,000,000 to 100,000,000)
- Number of threads (1 to 1,000)

Key metrics measured:
- Execution time
- Accuracy of π approximation
- Speedup relative to the single-threaded approach

## Key Findings
1. Approach 2 (local counters) scales well with increased thread count
2. Approach 3 (shared counter with mutex) suffers from mutex contention
3. Approach 4 (batched shared counter) significantly reduces mutex overhead
4. Thread scaling efficiency diminishes beyond a certain number of threads
5. The accuracy of π approximation improves with more sample points

## Visualization
The `plot_generator.py` script generates various plots to visualize:
- Execution time vs. number of threads
- Speedup vs. number of threads
- Error vs. number of threads
- Comparison between approaches
