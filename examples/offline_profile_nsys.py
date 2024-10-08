import inspect
import json
import sys
from argparse import RawTextHelpFormatter
from dataclasses import asdict, dataclass
from typing import Optional

import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.profiler import layerwise_profile
from vllm.utils import FlexibleArgumentParser

BATCH_SIZE_DEFAULT = 1
PROMPT_LEN_DEFAULT = 256
OUTPUT_LEN_DEFAULT = 2


@dataclass
class ProfileContext:
    model: str
    tokenizer: str
    model_revision: str
    quantization: str
    max_model_len: int
    max_num_batched_tokens: int
    prompt_len: int
    output_len: int
    batch_size: int
    dtype: str
    tensor_parallel_size: int
    allow_cuda_graphs: bool


def get_dtype(dtype: str):
    if dtype == "torch.float":
        return torch.float
    else:
        return dtype


def run_profile(context: ProfileContext, csv_output: Optional[str],
                json_output: Optional[str]):
    print("Run profile with:")
    for key, value in asdict(context).items():
        print(f"  {key} = {value}")

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=args.output_len,
                                     ignore_eos=True)

    # Create LLM
    llm = LLM(model=context.model,
              tokenizer=context.tokenizer
              if context.tokenizer is not None else context.model,
              revision=context.model_revision,
              enforce_eager=not context.allow_cuda_graphs,
              tensor_parallel_size=context.tensor_parallel_size,
              gpu_memory_utilization=0.9,
              max_model_len=context.max_model_len,
              quantization=context.quantization,
              dtype=get_dtype(context.dtype),
              max_num_batched_tokens=context.max_num_batched_tokens)

    batch_size = context.batch_size
    prompt_len = context.prompt_len
    output_len = context.output_len

    scheduler_config = llm.llm_engine.scheduler_config
    max_model_len = llm.llm_engine.model_config.max_model_len
    max_num_batched_tokens = scheduler_config.max_num_batched_tokens
    max_num_seqs = scheduler_config.max_num_seqs

    if batch_size * prompt_len > max_num_batched_tokens:
        print(f"ERROR: chosen batch_size * prompt_len "
              f"({batch_size} * {prompt_len} = {batch_size * prompt_len}) is  "
              f"larger than max_num_batched_tokens ({max_num_batched_tokens}) "
              f"and therefore cannot be run in a single profile step, please "
              f"choose a smaller batch size or prompt length, or increase "
              f"--max-num-batched-tokens")
        sys.exit(-1)
    if batch_size >= max_num_seqs:
        print(
            f"ERROR: chosen batch_size ({batch_size}) is larger than "
            f"max_num_seqs ({max_num_seqs}) and therefore cannot be run in a "
            f"single profile step, please choose a smaller batch size")
        sys.exit(-1)
    print("llm.llm_engine.model_config.max_model_len: ",
          llm.llm_engine.model_config.max_model_len)
    if prompt_len + output_len > llm.llm_engine.model_config.max_model_len:
        print(
            f"ERROR: chosen prompt_len + output_len ({prompt_len} + "
            f"{output_len} = {prompt_len + output_len}) is larger than the "
            f"model's max_model_len ({max_model_len}), please choose a smaller "
            f"prompt_len or output_len, or increase --max-model-len")
        sys.exit(-1)

    def add_requests():
        for i in range(batch_size):
            prompt_token_ids = torch.randint(
                llm.llm_engine.model_config.get_vocab_size(),
                size=(prompt_len, )).tolist()

            llm.llm_engine.add_request(
                request_id=f"seq{i}",
                inputs={'prompt_token_ids': prompt_token_ids},
                params=sampling_params)
    def abort_requests():
        for i in range(batch_size):
            llm.llm_engine.abort_request(f"seq{i}")
    
    # Warm up run
    print("Warm up run ...")
    add_requests()
    llm.llm_engine.step() # Prefill
    llm.llm_engine.step() # Decode
    abort_requests()
    
    print("Profile run ...")
    
    # Profile run
    add_requests()
    torch.cuda.cudart().cudaProfilerStart()

    torch.cuda.nvtx.range_push("prefill")
    llm.llm_engine.step()  # First step is prefill
    torch.cuda.nvtx.range_pop()
    
    for x in range(args.output_len - 1):
        torch.cuda.nvtx.range_push("decode_{}".format(x))
        llm.llm_engine.step()
        torch.cuda.nvtx.range_pop()
        
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="""
Script to be used with nsys to profile a model

    example:
    ```
    nsys profile -w true -t cuda,nvtx,cudnn,cublas \\
        --capture-range=cudaProfilerApi --capture-range-end=stop -x true \\
        -o report_out  python examples/offline_profile_nsys.py \\
        --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 --batch-size 4 \\
        --prompt-len 512 --json Llama31-8b-FP8-vec-large-grid-silu-mul \\
        --max-num-batched-tokens 8196
    ```
""",
                                    formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='The name or path of a HuggingFace Transformers model.')
    parser.add_argument("--tokenizer",
                        type=str,
                        default=None,
                        help="path to the tokenizer")

    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export the results as multiple csv file. This should be the root "
        "filename, will create <filename>_prefill_model_table.csv, "
        "<filename>_prefill_summary_table.csv, "
        "<filename>_decode_model_table.csv, and "
        "<filename>_decode_summary_table.csv")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Export the results as a json file. This should be the filename")
    parser.add_argument("--quantization",
                        "-q",
                        type=str,
                        choices=[*QUANTIZATION_METHODS, None],
                        default=None)
    parser.add_argument("--dtype",
                        type=str,
                        default='auto',
                        help="model dtype")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum length of a sequence (including prompt and output)")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to be processed in a single iteration. "
        " Should be greater than batch-size * prompt-len so the prefill can "
        " run in a single iteration.")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=PROMPT_LEN_DEFAULT,
        help=f"Length of the random prompt to use when profiling, all batched "
        f"requests use the same prompt_len, default={PROMPT_LEN_DEFAULT}")
    parser.add_argument("--batch-size",
                        type=int,
                        default=BATCH_SIZE_DEFAULT,
                        help=f"Number of requests to run as a single batch, "
                        f"default={BATCH_SIZE_DEFAULT}")
    parser.add_argument("--tensor-parallel-size",
                        "-tp",
                        type=int,
                        default=1,
                        help="Number of GPUs to use i.e. tensor parallelism, "
                        "default=1")
    parser.add_argument(
        "--allow-cuda-graphs",
        action='store_true',
        help="Enables cuda graphs to be used, well remove a lot of the module "
        "level info in the profiler results since almost everything runs in "
        "the graph where we do not have access to an informative stack trace")
    parser.add_argument(
        "--output-len",
        type=int,
        default=OUTPUT_LEN_DEFAULT,
        help="Number of llm steps to run (includes prefill and decode) "
        "- default={OUTPUT_LEN_DEFAULT}")

    args = parser.parse_args()

    context = ProfileContext(
        **{
            k: v
            for k, v in vars(args).items()
            if k in inspect.signature(ProfileContext).parameters
        })
    run_profile(context, csv_output=args.csv, json_output=args.json)
