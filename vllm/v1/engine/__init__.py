from dataclasses import dataclass
from typing import List, Optional, Union

import msgspec

from vllm.lora.request import LoRARequest
from vllm.sampling_params import RequestOutputKind, SamplingParams

POLLING_TIMEOUT_MS = 5000


@dataclass
class DetokenizerRequest:

    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    stop: List[str]
    include_stop_str_in_output: bool


class EngineCoreRequest(msgspec.Struct):

    # NOTE: prompt and prompt_token_ids should be DecoderOnlyInput,
    # but this object is currently not playing well with msgspec
    # due to circular imports and typing we have in data.py

    request_id: str
    #NOTE(Nick): I don't think we need to pass prompt here since it should
    # always be tokenized?
    prompt: Optional[str]
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    eos_token_id: Optional[int]
    arrival_time: float
    lora_request: Optional[LoRARequest]


@dataclass
class EngineCoreOutput:

    request_id: str
    new_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


class EngineCoreOutputs(msgspec.Struct):

    #NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout and using an int enum for finish/stop reason

    # [num_reqs]
    outputs: List[EngineCoreOutput]
