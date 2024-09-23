import os

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported
from vllm import LLM, SamplingParams
from vllm.compilation.backends import vllm_backend
from vllm.plugins import set_torch_compile_backend
from vllm.utils import cuda_device_count_stateless, is_hip

from ..utils import fork_new_process_for_each_test

TEST_MODELS = [
    ("facebook/opt-125m", {}),
    ("nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change", {
        "dtype": torch.float16,
        "quantization": "compressed-tensors"
    }),
    ("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", {
        "dtype": torch.float16,
        "quantization": "fp8"
    }),
    ("nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dyn-Per-Token-2048-Samples", {
        "quantization": "compressed-tensors"
    }),
    ("meta-llama/Meta-Llama-3-8B", {}),
]

# TODO: enable in pytorch 2.5
if False and is_quant_method_supported("aqlm"):
    TEST_MODELS.append(("ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf", {
        "quantization": "aqlm"
    }))

# TODO: enable in pytorch 2.5
if False and is_quant_method_supported("gguf"):
    TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", {
        "quantization": "gguf"
    }))

if is_quant_method_supported("gptq"):
    TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ", {
        "quantization": "gptq"
    }))

if is_quant_method_supported("gptq_marlin"):
    TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", {
        "quantization": "gptq_marlin"
    }))

if is_quant_method_supported("gptq_marlin_24"):
    TEST_MODELS.append(("alexm-nm/tinyllama-24-marlin24-4bit-g128", {
        "quantization": "gptq_marlin_24"
    }))

if is_quant_method_supported("marlin"):
    TEST_MODELS.append(("robertgshaw2/TinyLlama-1.1B-Chat-v1.0-g128-marlin", {
        "quantization": "marlin"
    }))

if not is_hip() and is_quant_method_supported("awq"):
    TEST_MODELS.append(("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", {
        "quantization": "AWQ"
    }))


@pytest.mark.parametrize("model_info", TEST_MODELS)
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("backend", ["eager", "inductor", vllm_backend])
@fork_new_process_for_each_test
def test_full_graph(model_info, tp_size, backend):

    # Skip the test if there are not enough CUDA devices.
    if cuda_device_count_stateless() < tp_size:
        pytest.skip("Not enough CUDA devices for the test.")

    # make sure these models can be captured in full graph mode
    if "VLLM_TEST_DYNAMO_GRAPH_CAPTURE" not in os.environ:
        os.environ["VLLM_TEST_DYNAMO_GRAPH_CAPTURE"] = "1"
        os.environ["VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE"] = "1"

    model = model_info[0]
    model_kwargs = model_info[1]

    # Inductor doesn't support fp8/gptq_marlin_24 yet.
    quantization = model_kwargs.get("quantization")
    if (quantization == "fp8"
            or quantization == "gptq_marlin_24") and backend != "eager":
        return

    set_torch_compile_backend(backend)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=model,
              enforce_eager=True,
              tensor_parallel_size=tp_size,
              disable_custom_all_reduce=True,
              **model_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    llm = None
    outputs = None
    sampling_params = None

    # Cleanup dynamo state so new models don't cause lots of
    # recompilation.
    torch._dynamo.reset()
