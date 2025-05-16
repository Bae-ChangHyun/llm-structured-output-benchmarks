import os
from typing import Any, Dict
import json

from ollama import chat
from loguru import logger

from frameworks.base import BaseFramework, experiment


class VanillaOllamaFramework(BaseFramework):
    """_summary_
    https://ollama.com/blog/structured-outputs
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def run(
        self, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            response = chat(
                model=self.llm_model,
                format=self.response_model.model_json_schema(),
                messages=[
                    {"role": "user", "content": self.prompt.format(**inputs)}
                ],
                options={'temperature': 0},
            )
            content = json.loads(response.message.content)
            
            return content

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        
        logger.info(f"실행 결과 - 성공률: {percent_successful}, 응답 수: {len(predictions) if predictions else 0}")
        return predictions, percent_successful, metrics, latencies