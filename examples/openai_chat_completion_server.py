"""Example of launching the vLLM openai-compatible server within Python (as opposed to via CLI.)

This example is useful if you want to debug the server.

Equivalent to the following CLI command:

`VLLM_USE_V1=1 vllm serve microsoft/Phi-3.5-vision-instruct --trust-remote-code --max-model-len 4096`
"""
import sys
import os
os.environ["VLLM_USE_V1"] = "1"
from vllm.scripts import main

if __name__ == '__main__':
    sys.argv=['vllm','serve','microsoft/Phi-3.5-vision-instruct','--trust-remote-code','--max-model-len','4096']
    main()