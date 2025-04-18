import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- Configuration ---
CSV_FILE = 'pi.csv'  # Input CSV filename
OUTPUT_DIR = 'plots'  # Directory to save generated plots
TRUE_PI = 3.14159265359  # Reference value of Pi
# Select nPoints for the overall comparison plot
NPOINTS_FOR_OVERALL_COMPARISON = 100000000  # 100 million points
# Points to completely exclude from plots
EXCLUDE_POINTS = [1000000000]  # Exclude 1 billion points if present

# --- Description of approaches for better labels ---
APPROACH_NAMES = {
    1: "Single-threaded",
    2: "Multi-threaded (Local Counters)",
    3: "Multi-threaded (Shared Counter + Mutex)",
    4: "Multi-threaded (Batched Shared Counter)"
}

# --- Helper Functions ---
def save_plot(filename, title):
    """Adds title, legend, grid, saves the current plot, and closes it."""
    try:
        plt.title(title, fontsize=14)
        # Only add legend if there are labeled elements
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, title="Legend", title_fontsize='11', fontsize='9')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')  # Save with good resolution
        print(f"   -> Saved: {filepath}")
    except Exception as e:
        print(f"   -> ERROR saving plot '{filename}': {e}", file=sys.stderr)
    finally:
        plt.close()  # Close the plot figure to free memory

# --- Main Script Execution ---

# 1. Create Output Directory
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory '{OUTPUT_DIR}'")
    except OSError as e:
        print(f"ERROR: Could not create directory '{OUTPUT_DIR}'. Error: {e}", file=sys.stderr)
        sys.exit(1)  # Exit if cannot create directory

# 2. Read and Process Data
print(f"\n--- Reading and processing data from '{CSV_FILE}' ---")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"CSV file read successfully. Row count: {len(df)}")
except FileNotFoundError:
    print(f"ERROR: File '{CSV_FILE}' not found. Please check the path.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR reading CSV file: {e}", file=sys.stderr)
    sys.exit(1)

# Check for required columns
required_cols = ['iteration', 'approach', 'nPoints', 'nThreads', 'time', 'pi', 'error']
if not all(col in df.columns for col in required_cols):
    missing_cols = [col for col in required_cols if col not in df.columns]
    print(f"ERROR: CSV file missing required columns: {missing_cols}", file=sys.stderr)
    sys.exit(1)

# Filter out excluded points
print(f"Filtering out excluded point sizes: {EXCLUDE_POINTS}")
original_rows = len(df)
df = df[~df['nPoints'].isin(EXCLUDE_POINTS)]
filtered_rows = len(df)
print(f"Filtered out {original_rows - filtered_rows} data rows.")

# Calculate average values for each configuration (ignoring iteration)
print("Calculating average values (time, error)...")
try:
    avg_df = df.groupby(['approach', 'nPoints', 'nThreads'], as_index=False)[['time', 'pi', 'error']].mean()
    print(f"Averaging successful. Unique configurations: {len(avg_df)}")
except TypeError as e:
    print(f"ERROR: Columns 'time', 'pi', or 'error' may contain non-numeric values. Pandas error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR during average calculation: {e}", file=sys.stderr)
    sys.exit(1)

# Add approach name column
avg_df['approach_name'] = avg_df['approach'].map(APPROACH_NAMES)

# 3. Calculate Speedup
print("Calculating Speedup metrics...")
# Get baseline time for Approach 1 (Single-Thread)
time_approach1 = avg_df[avg_df['approach'] == 1][['nPoints', 'time']].rename(columns={'time': 'time_1'})
if time_approach1.empty:
    print("WARNING: No data found for Approach 1. Speedup vs PP1 will not be calculated.")

# Get baseline time for Approach 2 (Multi-Thread, Local Counter)
time_approach2 = avg_df[avg_df['approach'] == 2][['nPoints', 'nThreads', 'time']].rename(columns={'time': 'time_2'})
if time_approach2.empty:
    print("WARNING: No data found for Approach 2. Speedup PP PP3 vs PP2 will not be calculated.")

# Get baseline time for Approach 3 (Multi-Thread, Shared Counter)
time_approach3 = avg_df[avg_df['approach'] == 3][['nPoints', 'nThreads', 'time']].rename(columns={'time': 'time_3'})
if time_approach3.empty:
    print("WARNING: No data found for Approach 3. Speedup PP4 vs PP3 will not be calculated.")

# Merge baseline times into the main average dataframe
merged_df = pd.merge(avg_df, time_approach1, on='nPoints', how='left')
merged_df = pd.merge(merged_df, time_approach2, on=['nPoints', 'nThreads'], how='left')
merged_df = pd.merge(merged_df, time_approach3, on=['nPoints', 'nThreads'], how='left')

# Calculate Speedup safely (avoid division by zero or NaN)
epsilon = 1e-9  # Small number to avoid division by zero

# Speedup vs Approach 1
merged_df['speedup_vs_1'] = np.where(
    (merged_df['time'] > epsilon) & pd.notna(merged_df['time_1']), 
    merged_df['time_1'] / merged_df['time'],
    np.nan 
)

# Speedup of Approach 3 vs Approach 2
merged_df['speedup_3_vs_2'] = np.where(
    (merged_df['approach'] == 3) & (merged_df['time'] > epsilon) & pd.notna(merged_df['time_2']), 
    merged_df['time_2'] / merged_df['time'],
    np.nan 
)

# Speedup of Approach 4 vs Approach 3
merged_df['speedup_4_vs_3'] = np.where(
    (merged_df['approach'] == 4) & (merged_df['time'] > epsilon) & pd.notna(merged_df['time_3']), 
    merged_df['time_3'] / merged_df['time'],
    np.nan 
)
print("Speedup calculation complete.")

# --- 4. Plotting ---
print("\n--- Starting plots generation ---")

# Get unique values for loops and axes
unique_nPoints = sorted(merged_df['nPoints'].unique())

# Threads for multi-threaded plots
multi_thread_df = merged_df[merged_df['approach'] > 1]
unique_nThreads = sorted(multi_thread_df['nThreads'].unique()) if not multi_thread_df.empty else []

# Define markers and linestyles for consistency
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
linestyles = ['-', '--', ':', '-.']

# === Plotting for Approach 1 ===
print("Plotting: Approach 1 charts...")
pp1_data = merged_df[merged_df['approach'] == 1].sort_values('nPoints')
if not pp1_data.empty:
    # 1.1 Time vs nPoints (Approach 1)
    plt.figure(figsize=(8, 5))
    plt.plot(pp1_data['nPoints'], pp1_data['time'], marker='o', linestyle='-', 
             label=f'{APPROACH_NAMES[1]} - Avg Time')
    plt.xlabel("Number of Points (nPoints)")
    plt.ylabel("Average Execution Time (seconds)")
    plt.xscale('log')  # Keep log scale for x-axis (nPoints)
    save_plot('plot_1_1_pp1_time_vs_npoints.png', f'{APPROACH_NAMES[1]}: Execution Time by Point Count')

    # 1.2 Error vs nPoints (Approach 1)
    plt.figure(figsize=(8, 5))
    plt.plot(pp1_data['nPoints'], pp1_data['error'], marker='o', linestyle='-', color='red', 
             label=f'{APPROACH_NAMES[1]} - Avg Error')
    plt.xlabel("Number of Points (nPoints)")
    plt.ylabel("Average Absolute Error")
    plt.xscale('log')
    plt.yscale('log')  # Keep log scale for error
    save_plot('plot_1_2_pp1_error_vs_npoints.png', f'{APPROACH_NAMES[1]}: Error by Point Count')
else:
    print("   -> Skipping Approach 1 plots due to missing data.")

# === Plotting for Approach 2 ===
print("Plotting: Approach 2 charts...")
data_pp2 = merged_df[merged_df['approach'] == 2]
if not data_pp2.empty and unique_nThreads:
    # 2.1 Time vs nThreads (Approach 2)
    plt.figure(figsize=(10, 6))
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp2[data_pp2['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty:
            plt.plot(subset['nThreads'], subset['time'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Execution Time (seconds)")
    plt.xscale('log')  # Use log scale for thread counts
    plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_2_1_pp2_time_vs_threads.png', f'{APPROACH_NAMES[2]}: Time vs Thread Count')

    # 2.2 Speedup (vs PP1) vs nThreads (Approach 2)
    plt.figure(figsize=(10, 6))
    has_speedup_data_pp2 = False
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp2[data_pp2['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty and not subset['speedup_vs_1'].isnull().all():
            plt.plot(subset['nThreads'], subset['speedup_vs_1'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
            has_speedup_data_pp2 = True
    if has_speedup_data_pp2:
        # Ideal speedup line
        plt.plot(unique_nThreads, unique_nThreads, 'k--', label='Ideal Speedup (linear)')
        plt.xlabel("Number of Threads")
        plt.ylabel(f"Speedup vs {APPROACH_NAMES[1]}")
        plt.xscale('log')  # Use log scale
        plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
        save_plot('plot_2_2_pp2_speedup_vs_1.png', f'{APPROACH_NAMES[2]}: Speedup vs Approach 1')
    else:
        print("   -> Skipping Speedup PP2 vs PP1 plot due to missing baseline data.")
        plt.close()

    # 2.3 Error vs nThreads (Approach 2)
    plt.figure(figsize=(10, 6))
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp2[data_pp2['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty:
            plt.plot(subset['nThreads'], subset['error'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}',
                     color='red') 
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Absolute Error")
    plt.xscale('log')  # Use log scale for thread counts
    plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
    plt.yscale('log')
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_2_3_pp2_error_vs_threads.png', f'{APPROACH_NAMES[2]}: Error by Thread Count')
else:
    print("   -> Skipping Approach 2 plots due to missing data.")

# === Plotting for Approach 3 ===
print("Plotting: Approach 3 charts...")
data_pp3 = merged_df[merged_df['approach'] == 3]
if not data_pp3.empty and unique_nThreads:
    # 3.1 Time vs nThreads (Approach 3)
    plt.figure(figsize=(10, 6))
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp3[data_pp3['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty:
            plt.plot(subset['nThreads'], subset['time'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Execution Time (seconds)")
    plt.xscale('log')  # Log scale for thread counts
    plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_3_1_pp3_time_vs_threads.png', f'{APPROACH_NAMES[3]}: Time vs Thread Count')

    # 3.2 Speedup vs Approach 2 (for each thread count)
    plt.figure(figsize=(10, 6))
    has_speedup_data = False
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp3[data_pp3['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty and not subset['speedup_3_vs_2'].isnull().all():
            plt.plot(subset['nThreads'], subset['speedup_3_vs_2'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
            has_speedup_data = True
    if has_speedup_data:
        plt.xlabel("Number of Threads")
        plt.ylabel(f"Speedup vs {APPROACH_NAMES[2]}")
        plt.axhline(y=1.0, color='k', linestyle='--', label='Equal Performance')
        plt.xscale('log')  # Log scale for thread counts
        plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
        save_plot('plot_3_2_pp3_speedup_vs_2.png', f'{APPROACH_NAMES[3]}: Speedup vs Approach 2')
    else:
        print("   -> Skipping Speedup PP3 vs PP2 plot due to missing data.")
        plt.close()

    # 3.3 Speedup vs Approach 1
    plt.figure(figsize=(10, 6))
    has_speedup_data_pp3 = False
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp3[data_pp3['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty and not subset['speedup_vs_1'].isnull().all():
            plt.plot(subset['nThreads'], subset['speedup_vs_1'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
            has_speedup_data_pp3 = True
    if has_speedup_data_pp3:
        # Ideal speedup line
        plt.plot(unique_nThreads, unique_nThreads, 'k--', label='Ideal Speedup (linear)')
        plt.xlabel("Number of Threads")
        plt.ylabel(f"Speedup vs {APPROACH_NAMES[1]}")
        plt.xscale('log')  # Log scale for thread counts
        plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
        save_plot('plot_3_3_pp3_speedup_vs_1.png', f'{APPROACH_NAMES[3]}: Speedup vs Approach 1')
    else:
        print("   -> Skipping Speedup PP3 vs PP1 plot due to missing baseline data.")
        plt.close()

    # 3.4 Error vs nThreads (Approach 3)
    plt.figure(figsize=(10, 6))
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp3[data_pp3['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty:
            plt.plot(subset['nThreads'], subset['error'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}',
                     color='red')
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Absolute Error")
    plt.xscale('log')  # Log scale for thread counts
    plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
    plt.yscale('log')
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_3_4_pp3_error_vs_threads.png', f'{APPROACH_NAMES[3]}: Error by Thread Count')
else:
    print("   -> Skipping Approach 3 plots due to missing data.")

# === Plotting for Approach 4 ===
print("Plotting: Approach 4 charts...")
data_pp4 = merged_df[merged_df['approach'] == 4]
if not data_pp4.empty and unique_nThreads:
    # 4.1 Time vs nThreads (Approach 4)
    plt.figure(figsize=(10, 6))
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp4[data_pp4['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty:
            plt.plot(subset['nThreads'], subset['time'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Execution Time (seconds)")
    plt.xscale('log')  # Log scale for thread counts
    plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_4_1_pp4_time_vs_threads.png', f'{APPROACH_NAMES[4]}: Time vs Thread Count')

    # 4.2 Speedup vs Approach 3 (for each thread count)
    plt.figure(figsize=(10, 6))
    has_speedup_data = False
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp4[data_pp4['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty and not subset['speedup_4_vs_3'].isnull().all():
            plt.plot(subset['nThreads'], subset['speedup_4_vs_3'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
            has_speedup_data = True
    if has_speedup_data:
        plt.xlabel("Number of Threads")
        plt.ylabel(f"Speedup vs {APPROACH_NAMES[3]}")
        plt.axhline(y=1.0, color='k', linestyle='--', label='Equal Performance')
        plt.xscale('log')  # Log scale for thread counts
        plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
        save_plot('plot_4_2_pp4_speedup_vs_3.png', f'{APPROACH_NAMES[4]}: Speedup vs Approach 3')
    else:
        print("   -> Skipping Speedup PP4 vs PP3 plot due to missing data.")
        plt.close()

    # 4.3 Speedup vs Approach 1
    plt.figure(figsize=(10, 6))
    has_speedup_data_pp4 = False
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp4[data_pp4['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty and not subset['speedup_vs_1'].isnull().all():
            plt.plot(subset['nThreads'], subset['speedup_vs_1'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}')
            has_speedup_data_pp4 = True
    if has_speedup_data_pp4:
        # Ideal speedup line
        plt.plot(unique_nThreads, unique_nThreads, 'k--', label='Ideal Speedup (linear)')
        plt.xlabel("Number of Threads")
        plt.ylabel(f"Speedup vs {APPROACH_NAMES[1]}")
        plt.xscale('log')  # Log scale for thread counts
        plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
        save_plot('plot_4_3_pp4_speedup_vs_1.png', f'{APPROACH_NAMES[4]}: Speedup vs Approach 1')
    else:
        print("   -> Skipping Speedup PP4 vs PP1 plot due to missing baseline data.")
        plt.close()

    # 4.4 Error vs nThreads (Approach 4)
    plt.figure(figsize=(10, 6))
    for i, n_points in enumerate(unique_nPoints):
        subset = data_pp4[data_pp4['nPoints'] == n_points].sort_values('nThreads')
        if not subset.empty:
            plt.plot(subset['nThreads'], subset['error'],
                     marker=markers[i % len(markers)],
                     linestyle=linestyles[i % len(linestyles)],
                     label=f'N={n_points:,}',
                     color='red')
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Absolute Error")
    plt.xscale('log')  # Log scale for thread counts
    plt.xticks(unique_nThreads, [str(x) for x in unique_nThreads])
    plt.yscale('log')
    plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_4_4_pp4_error_vs_threads.png', f'{APPROACH_NAMES[4]}: Error by Thread Count')
else:
    print("   -> Skipping Approach 4 plots due to missing data.")

# === Overall Comparison Plot ===
print(f"Plotting: Overall comparison (for nPoints = {NPOINTS_FOR_OVERALL_COMPARISON:,})...")
comp_data = merged_df[merged_df['nPoints'] == NPOINTS_FOR_OVERALL_COMPARISON]

if not comp_data.empty:
    plt.figure(figsize=(12, 7))

    # Data for each approach at the chosen nPoints
    pp1_comp = comp_data[comp_data['approach'] == 1]
    pp2_comp = comp_data[comp_data['approach'] == 2].sort_values('nThreads')
    pp3_comp = comp_data[comp_data['approach'] == 3].sort_values('nThreads')
    pp4_comp = comp_data[comp_data['approach'] == 4].sort_values('nThreads')

    # Determine thread axis from multi-threaded data
    threads_axis = sorted(pp2_comp['nThreads'].unique()) if not pp2_comp.empty else (
        sorted(pp3_comp['nThreads'].unique()) if not pp3_comp.empty else (
            sorted(pp4_comp['nThreads'].unique()) if not pp4_comp.empty else [1]))

    # Plot Approach 1 time (constant line across threads axis)
    if not pp1_comp.empty and threads_axis:
        time_pp1_const = pp1_comp['time'].iloc[0]
        plt.plot(threads_axis, [time_pp1_const] * len(threads_axis),
                 marker='^', linestyle='--', label=f'{APPROACH_NAMES[1]} ({time_pp1_const:.2f}s)')
    elif not pp1_comp.empty:
        plt.plot([1], pp1_comp['time'], marker='^', linestyle='--', 
                 label=f'{APPROACH_NAMES[1]} ({pp1_comp["time"].iloc[0]:.2f}s)')
        threads_axis = [1]

    # Plot Approach 2 time vs threads
    if not pp2_comp.empty:
        plt.plot(pp2_comp['nThreads'], pp2_comp['time'], marker='o', linestyle='-', 
                 label=APPROACH_NAMES[2])

    # Plot Approach 3 time vs threads
    if not pp3_comp.empty:
        plt.plot(pp3_comp['nThreads'], pp3_comp['time'], marker='s', linestyle='-', 
                 label=APPROACH_NAMES[3])

    # Plot Approach 4 time vs threads
    if not pp4_comp.empty:
        plt.plot(pp4_comp['nThreads'], pp4_comp['time'], marker='D', linestyle='-', 
                 label=APPROACH_NAMES[4])

    plt.xlabel("Number of Threads")
    plt.ylabel("Average Execution Time (seconds)")
    if threads_axis:
        plt.xscale('log')  # Use log scale
        plt.xticks(threads_axis, [str(x) for x in threads_axis])
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    save_plot('plot_5_1_overall_time_comparison.png', 
              f'Execution Time Comparison (nPoints = {NPOINTS_FOR_OVERALL_COMPARISON:,})')
else:
    print(f"   -> Skipping overall comparison plot due to missing data for nPoints = {NPOINTS_FOR_OVERALL_COMPARISON:,}.")

print(f"\n--- Completed! ---")
print(f"All plots have been saved to '{OUTPUT_DIR}' directory.")