#!/bin/bash

export DATASET=/home/exouser/dataset

# Clear the output file or create it fresh
> sim_result_more_new.txt

# Define arrays for the parameter combinations
num_nodes=(1 2 3 4 5)
ngpus_per_node=(2 4)
prefill_targets=(150 200 250)
decode_targets=(50 75 100 125 150)

# Count total combinations for progress tracking
total_combinations=$((${#num_nodes[@]} * ${#ngpus_per_node[@]} * ${#prefill_targets[@]} * ${#decode_targets[@]}))
current_combination=0

echo "Starting simulations for ${total_combinations} configurations..."
echo "Results will be saved to sim_result_more_new.txt"
echo ""

# Loop through all combinations
for num_node in "${num_nodes[@]}"; do
    for ngpu_per_node in "${ngpus_per_node[@]}"; do
        for prefill_target in "${prefill_targets[@]}"; do
            for decode_target in "${decode_targets[@]}"; do
                current_combination=$((current_combination + 1))
                
                echo "[${current_combination}/${total_combinations}] Running: num_node=${num_node}, ngpu_per_node=${ngpu_per_node}, prefill_target=${prefill_target}ms, decode_target=${decode_target}ms"
                
                python -m simdistserve.simulate_print_all \
                    --num-node ${num_node} \
                    --ngpu-per-node ${ngpu_per_node} \
                    --model-type "facebook/opt-13b" \
                    --workload sharegpt \
                    --backend distserve \
                    --prefill-target ${prefill_target} \
                    --decode-target ${decode_target} \
                    --prefill-percentage 90 \
                    --decode-percentage 90 \
                    --max-per-gpu-rate 5 \
                    --esp 0.25 \
                    --N 300 >> sim_result_more_new.txt 2>&1
                
                echo "  ✓ Completed"
            done
        done
    done
done

echo ""
echo "================================================"
echo "All simulations completed!"
echo "Total configurations tested: ${total_combinations}"
echo "Results saved to sim_result_more_new.txt"
echo "================================================"