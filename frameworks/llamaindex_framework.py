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
        
        if self.llm_model_host == "openai":
            self.client = OpenAI()
            logger.debug("OpenAI 클라이언트가 초기화되었습니다.")
        elif self.llm_model_host == "ollama":
            self.client = OpenAI(
                base_url=os.environ['OLLAMA_HOST'],
                api_key="ollama",
            )
            logger.debug("Ollama 클라이언트가 초기화되었습니다.")
            
        # TODO: Swap the Program based on self.llm_model
        self.llamaindex_client = OpenAIPydanticProgram.from_defaults(
            output_cls=self.response_model,
            prompt_template_str=self.prompt,
            llm_model=self.client,
        )

    def run(
        self, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            response = self.llamaindex_client(**inputs, description="Data model of items present in the text")
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
