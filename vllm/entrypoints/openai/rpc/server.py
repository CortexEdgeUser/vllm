import asyncio
import signal
from typing import Any, Coroutine

import cloudpickle
import zmq
import zmq.asyncio
from typing_extensions import Never

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.rpc import (VLLM_RPC_ZMQ_MAX_SOCKETS,
                                         VLLM_RPC_HEALTHY_STR,
                                         VLLM_RPC_SUCCESS_STR, RPCAbortRequest,
                                         RPCGenerateRequest, RPCUtilityRequest)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


class AsyncEngineRPCServer:

    def __init__(self, async_engine_args: AsyncEngineArgs,
                 usage_context: UsageContext, rpc_path: str):
        # Initialize engine first.
        self.engine = AsyncLLMEngine.from_engine_args(async_engine_args,
                                                      usage_context)

        # Initialize context.
        self.context = zmq.asyncio.Context()
        self.context.set(zmq.MAX_SOCKETS, VLLM_RPC_ZMQ_MAX_SOCKETS)

        # Init socket for readiness state.
        self.socket = self.context.socket(zmq.constants.ROUTER)
        self.socket.bind(rpc_path)

    def cleanup(self):
        """Cleanup all resources."""
        self.socket.close()
        self.context.destroy()

    async def get_config(self, identity, part2, request):
        try:
            if request == RPCUtilityRequest.GET_MODEL_CONFIG:
                config = await self.engine.get_model_config()
            elif request == RPCUtilityRequest.GET_DECODING_CONFIG:
                config = await self.engine.get_decoding_config()
            elif request == RPCUtilityRequest.GET_LORA_CONFIG:
                config = await self.engine.get_lora_config()
            elif request == RPCUtilityRequest.GET_SCHEDULER_CONFIG:
                config = await self.engine.get_scheduler_config()
            elif request == RPCUtilityRequest.GET_PARALLEL_CONFIG:
                config = await self.engine.get_parallel_config()
            else:
                raise ValueError("Unknown Config Request: %s", request)

            await self.socket.send_multipart([
                identity, part2, cloudpickle.dumps(config)
            ])

        except Exception as e:
            ### Notify client of all failures
            await self.socket.send_multipart([
                identity, part2, cloudpickle.dumps(e)
            ])

    async def is_tracing_enabled(self, identity, part2):
        """Send the is_tracing_enabled flag"""
        tracing_flag = await self.engine.is_tracing_enabled()

        await self.socket.send_multipart([
            identity, part2, cloudpickle.dumps(tracing_flag)
        ])

    async def do_log_stats(self, identity, part2):
        """Log stats and confirm success."""
        await self.engine.do_log_stats()

        await self.socket.send_multipart([
            identity, part2, cloudpickle.dumps(VLLM_RPC_SUCCESS_STR),
        ])

    async def is_server_ready(self, identity, part2):
        """Notify the client that we are ready."""
        await self.socket.send_multipart([
            identity, part2, cloudpickle.dumps(VLLM_RPC_SUCCESS_STR),
        ])

    async def abort(self, identity, part2, request: RPCAbortRequest):
        """Abort request and notify the client of success."""
        try:
            # Abort the request in the llm engine.
            await self.engine.abort(request.request_id)
        except Exception:
            logger.warning("Failed to abort request %s", request.request_id)
        finally:
            # Send confirmation to the client.
            await self.socket.send_multipart([
                identity, part2, cloudpickle.dumps(VLLM_RPC_SUCCESS_STR),
            ])

    async def generate(self, identity, part2, generate_request: RPCGenerateRequest):
        try:
            results_generator = self.engine.generate(
                generate_request.inputs,
                sampling_params=generate_request.sampling_params,
                request_id=generate_request.request_id,
                lora_request=generate_request.lora_request,
                trace_headers=generate_request.trace_headers,
                prompt_adapter_request=generate_request.prompt_adapter_request)

            async for request_output in results_generator:
                await self.socket.send_multipart([
                    identity, part2, cloudpickle.dumps(request_output)
                ])

        except Exception as e:
            await self.socket.send_multipart([
                identity, part2, cloudpickle.dumps(e)
            ])

    async def check_health(self, identity, part2):
        try:
            await self.engine.check_health()
            await self.socket.send_multipart([
                identity, part2, cloudpickle.dumps(VLLM_RPC_HEALTHY_STR)
            ])

        except Exception as e:
            await self.socket.send_multipart([
                identity, part2, cloudpickle.dumps(e)
            ])

    def _make_handler_coro(self, identity, part2,
                           message) -> Coroutine[Any, Any, Never]:
        """Route the zmq message to the handler coroutine."""

        request = cloudpickle.loads(message)

        if isinstance(request, RPCGenerateRequest):
            return self.generate(identity, part2, request)

        elif isinstance(request, RPCAbortRequest):
            return self.abort(identity, part2, request)

        elif isinstance(request, RPCUtilityRequest):
            if request in [RPCUtilityRequest.GET_MODEL_CONFIG,
                           RPCUtilityRequest.GET_PARALLEL_CONFIG,
                           RPCUtilityRequest.GET_DECODING_CONFIG,
                           RPCUtilityRequest.GET_SCHEDULER_CONFIG,
                           RPCUtilityRequest.GET_LORA_CONFIG]:
                return self.get_config(identity, part2, request)
            elif request == RPCUtilityRequest.DO_LOG_STATS:
                return self.do_log_stats(identity, part2)
            elif request == RPCUtilityRequest.IS_SERVER_READY:
                return self.is_server_ready(identity, part2)
            elif request == RPCUtilityRequest.CHECK_HEALTH:
                return self.check_health(identity, part2)
            elif request == RPCUtilityRequest.IS_TRACING_ENABLED:
                return self.is_tracing_enabled(identity, part2)
            else:
                raise ValueError(f"Unknown RPCUtilityRequest type: {request}")

        else:
            raise ValueError(f"Unknown RPCRequest type: {request}")

    async def run_server_loop(self):
        """Inner RPC Server Loop"""

        running_tasks = set()
        while True:
            # Wait for a request.
            identity, part2, message = await self.socket.recv_multipart()

            # Process the request async.
            task = asyncio.create_task(
                self._make_handler_coro(identity, part2, message))

            # We need to keep around a strong reference to the task,
            # to avoid the task disappearing mid-execution as running tasks
            # can be GC'ed. Below is a common "fire-and-forget" tasks
            # https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task
            running_tasks.add(task)
            task.add_done_callback(running_tasks.discard)


async def run_server(server: AsyncEngineRPCServer):
    # Put the server task into the asyncio loop.
    loop = asyncio.get_running_loop()
    server_task = loop.create_task(server.run_server_loop())

    # Interruption handling.
    def signal_handler() -> None:
        # Kill the server on interrupt / terminate
        server_task.cancel()

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    try:
        await server_task
    except asyncio.CancelledError:
        logger.info("vLLM ZMQ RPC Server was interrupted.")
    finally:
        # Clean up all resources.
        server.cleanup()


def run_rpc_server(async_engine_args: AsyncEngineArgs,
                   usage_context: UsageContext, rpc_path: str):
    server = AsyncEngineRPCServer(async_engine_args, usage_context, rpc_path)
    asyncio.run(run_server(server))
