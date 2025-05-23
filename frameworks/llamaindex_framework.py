import os
from pyexpat import model
from typing import Any
from openai import OpenAI
from loguru import logger

from llama_index.program.openai import OpenAIPydanticProgram

from frameworks.base import BaseFramework, experiment


class LlamaIndexFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if self.llm_provider == "openai":
            self.client = OpenAI()
            logger.debug("OpenAI 클라이언트가 초기화되었습니다.")
        elif self.llm_provider == "ollama" or self.llm_provider == "vllm":
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="empty",
            )
            logger.debug("Ollama/Vllm 클라이언트가 초기화되었습니다.")
        elif self.llm_provider == "google":
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            logger.debug("Google 클라이언트가 초기화되었습니다.")
            
        # TODO: Swap the Program based on self.llm_model
        self.llamaindex_client = OpenAIPydanticProgram.from_defaults(
            output_cls=self.response_model,
            prompt_template_str=self.prompt,
            llm_model=self.client,
        )

    def run(
        self, retries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(retries=retries, expected_response=expected_response)
        def run_experiment(inputs):
            response = self.llamaindex_client(**inputs, description="Data model of items present in the text")
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
