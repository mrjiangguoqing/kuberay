import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
#from vllm.entrypoints.openai.serving_models import LoRAModulePath
from vllm.utils import FlexibleArgumentParser
#from vllm.entrypoints.openai.serving_models import OpenAIServingModels 

from dataclasses import dataclass



@dataclass
class BaseModelPath:
    name: str
    model_path: str

BASE_MODEL_PATHS = [BaseModelPath(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")]

logger = logging.getLogger("ray.serve")

app = FastAPI()


os.environ['TP_SOCKET_IFNAME'] = 'eth0'
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['NCCL_DEBUG'] = 'TRACE'
os.environ['VLLM_TRACE_FUNCTION'] = '1'    
os.environ['NCCL_IB_DISABLEN'] = '1'    


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        #model: str,
        # lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.setLevel(logging.DEBUG)
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        # self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            #models = OpenAIServingModels(
            #    engine_client=self.engine,
            #    model_config=model_config,
            #    base_model_paths=[BaseModelPath(name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", model_path="/data/model_data/models--DeepSeek-R1-Distill-Qwen-1.5B")],
            #)

            # Determine the name of the served model for the OpenAI client.
            # if self.engine_args.served_model_name is not None:
                # served_model_names = self.engine_args.served_model_name
            # else:
                # served_model_names = [self.engine_args.model]

            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                BASE_MODEL_PATHS,
                #models,
                # served_model_names=served_model_names,
                #chat_template_content_format="auto",
                # enable_reasoning: bool = False,
                response_role=self.response_role,
                # lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                #prompt_adapters=None,
                #request_logger=None,
                lora_modules=None,          # 显式传递 None
                prompt_adapters=None,       # 显式传递 None
                request_logger=None,  
            )
        
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())



def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    #placement_group_bundles=bundles, placement_group_strategy="PACK"

    #bundles=[{"CPU":8,"GPU": 1},{"CPU":7,"GPU": 1},{"CPU":0,"GPU": 1},{"CPU":0,"GPU": 1}]
    bundles=[{"CPU":7},{"GPU": 1},{"GPU": 1},{"GPU": 1},{"GPU": 1}]
    #bundles=[{"CPU":7,"GPU": 1}]
    
    return VLLMDeployment.options(placement_group_bundles=bundles, placement_group_strategy="STRICT_SPREAD").bind(
        engine_args,
        parsed_args.response_role,
        #parsed_args.lora_modules,
        parsed_args.chat_template,
        #parsed_args.model
    )

#"model": os.environ['MODEL_ID'],"cpu-offload-gb": os.environ['cpu-offload-gb']
model = build_app(
    {"model": os.environ['MODEL_ID'],"tensor-parallel-size": os.environ['TENSOR_PARALLELISM'], "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM'],"swap-space": os.environ['SWAP_SPACE'],"distributed_executor_backend": os.environ['distributed_executor_backend'],"dtype": os.environ['dtype'],"cpu-offload-gb": os.environ['cpu-offload-gb'],"download-dir": os.environ['download-dir'],"max_num_batched_tokens": os.environ['max_num_batched_tokens'],"max_num_seqs": os.environ['max_num_seqs'],"kv-cache-dtype": os.environ['kv-cache-dtype'],"enable_chunked_prefill": os.environ['enable_chunked_prefill']})
