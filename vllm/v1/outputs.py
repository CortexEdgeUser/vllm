import enum
from typing import Dict, List, Optional

import msgspec
import torch


class SamplerOutput(msgspec.Struct,
                    array_like=True,
                    omit_defaults=True,
                    gc=False):

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: Optional[torch.Tensor]

    # TODO: Support prompt logprobs.
    prompt_logprob_token_ids: Optional[torch.Tensor]
    prompt_logprobs: Optional[torch.Tensor]


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use List instead.
class ModelRunnerOutput(msgspec.Struct,
                        array_like=True,
                        omit_defaults=True,
                        gc=False):

    # [num_reqs]
    req_ids: List[str]
    # req_id -> index
    req_id_to_index: Dict[str, int]

    # [num_reqs]
    sampled_token_ids_cpu: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids_cpu: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs_cpu: Optional[torch.Tensor]

class WorkerOutputType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """
    NUM_BLOCKS = b'\x00'
    MODEL_RUNNER_OUTPUT = b'\x01'
