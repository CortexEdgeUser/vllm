from typing import List, Optional, Union

import torch

import vllm.envs as envs
from vllm.attention import AttentionMetadata
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.sequence import IntermediateTensors


def support_compile_llama_style(cls: type):
    """
    A decorator to add support for compiling the forward method of a class.
    If a module's **forward signature** is compatible with llama, this 
    decorator can be used to enable the compilation of the forward method.
    """

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    def __init__(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self._use_torch_compile = envs.VLLM_TEST_TORCH_COMPILE_LEVEL > 0
        if self._use_torch_compile:
            TorchCompileWrapperWithCustomDispatcher.__init__(self)

    cls.__init__ = __init__

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if not self._use_torch_compile:
            return self.forward(input_ids, positions, kv_caches, attn_metadata,
                                intermediate_tensors)

        # the first compilation needs to have dynamic shapes marked
        if len(self.compiled_codes) < 1:
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(positions, 0)
            if intermediate_tensors is not None:
                for tensors in intermediate_tensors.tensors.values():
                    torch._dynamo.mark_dynamic(tensors, 0)
            return self.compiled_callable(input_ids, positions, kv_caches,
                                          attn_metadata, intermediate_tensors)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if not self.use_custom_dispatcher:
            return self.compiled_callable(input_ids, positions, kv_caches,
                                          attn_metadata, intermediate_tensors)

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(input_ids, positions, kv_caches,
                                        attn_metadata, intermediate_tensors)
            return model_output

    cls.__call__ = __call__
    return cls
