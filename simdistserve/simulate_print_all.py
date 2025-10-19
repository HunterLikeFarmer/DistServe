import argparse

from simdistserve.benchmarks.parallel_bisect import simulate_bisect_search
from simdistserve.constants import ModelTypes


def parse_args():
    parser = argparse.ArgumentParser("Simulate DistServe or vLLM to find the optimal configuration.")
    parser.add_argument("--ngpu-per-node", type=int, default=4)
    parser.add_argument("--num-node", type=int, default=1)
    parser.add_argument("--is-high-affinity", action="store_true")
    parser.add_argument("--backend", type=str, default="distserve",
                        help="Choose from: distserve, vllm")
    parser.add_argument("--workload", type=str, default="sharegpt",
                        help="Choose from: sharegpt, humaneval, longbench")
    parser.add_argument("--prefill-target", type=int, default=200,
                        help="Prefill TTFT attainment target in ms (default 200ms)")
    parser.add_argument("--decode-target", type=int, default=100,
                        help="Decode TPOT attainment target in ms (default 100ms)")
    parser.add_argument("--prefill-percentage", type=int, default=90,
                        help="Percentage of prefill target (default P90)")
    parser.add_argument("--decode-percentage", type=int, default=90,
                        help="Percentage of prefill target (default P90)")
    parser.add_argument("--max-per-gpu-rate", type=int, default=5,
                        help="Max per GPU rate to search (default 5)")
    parser.add_argument("--esp", type=float, default=0.25,
                        help="Stopping criteria: `high - low < esp` (default esp = 0.25)")
    parser.add_argument("--N", type=int, default=300,
                        help="Number of samples to simulate (default 1000)")
    parser.add_argument("--model-type", type=str, default="opt_13b",
                        help="Model type to simulate (opt_13b, opt_66b, opt_175b)")

    args = parser.parse_args()
    args.model_type = ModelTypes.model_str_to_object(args.model_type)
    return args


def print_all_configs(config_to_best_per_gpu_rate, backend):
    """
    Print all configurations in a simple parseable format.
    Output format: tab-separated values, one config per line.
    """
    if backend == 'distserve':
        # Print header
        print("pp_cross\ttp_prefill\tpp_prefill\ttp_decode\tpp_decode\ttotal_gpus\tper_gpu_rate\ttotal_throughput")
        
        # Print each config
        for config, per_gpu_rate in config_to_best_per_gpu_rate.items():
            pp_cross, tp_prefill, pp_prefill, tp_decode, pp_decode = config
            num_gpu = pp_cross * (tp_prefill * pp_prefill + tp_decode * pp_decode)
            total_throughput = per_gpu_rate * num_gpu
            print(f"{pp_cross}\t{tp_prefill}\t{pp_prefill}\t{tp_decode}\t{pp_decode}\t{num_gpu}\t{per_gpu_rate:.2f}\t{total_throughput:.2f}")
    
    elif backend == 'vllm':
        # Print header
        print("tp\tpp\ttotal_gpus\tper_gpu_rate\ttotal_throughput")
        
        # Print each config
        for config, per_gpu_rate in config_to_best_per_gpu_rate.items():
            tp, pp = config
            num_gpu = tp * pp
            total_throughput = per_gpu_rate * num_gpu
            print(f"{tp}\t{pp}\t{num_gpu}\t{per_gpu_rate:.2f}\t{total_throughput:.2f}")


def check_dataset_env_var():
    import os
    if "DATASET" in os.environ:
        return
    raise KeyError(
        "Please set the environment variable `DATASET` to the path of the workload datasets. "
        "For user who started the environment with `DistServe-AE-GPU` docker image, "
        "simply do:\nexport DATASET=`/app/dataset`\n"
        "See the `repro-dataset.md` to prepare for workload dataset if you are using your custom environment."
    )


if __name__ == '__main__':
    args = parse_args()
    print(args)

    result = simulate_bisect_search(
        args.num_node,
        args.ngpu_per_node,
        model_type=args.model_type,
        is_dist_high=args.is_high_affinity,
        backend=args.backend,
        attainment=(args.prefill_target, args.decode_target, args.prefill_percentage, args.decode_percentage),
        max_per_gpu_rate=args.max_per_gpu_rate,
        esp=args.esp,
        N=args.N,
    )

    # Print all configurations in parseable format
    print_all_configs(result, args.backend)
