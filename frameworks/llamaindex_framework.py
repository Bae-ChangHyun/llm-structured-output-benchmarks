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
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY is not set in the environment variables!")
                raise ValueError("Export OEPN_API_KEY in your environment variables.")
                
            self.openai_client = OpenAI()
        elif self.llm_model_host == "ollama":
            self.openai_client = OpenAI(
                base_url=os.environ['OLLAMA_HOST'],
                api_key="ollama",
            )
            
        # TODO: Swap the Program based on self.llm_model
        self.llamaindex_client = OpenAIPydanticProgram.from_defaults(
            output_cls=self.response_model,
            prompt_template_str=self.prompt,
            llm_model=self.openai_client,
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
