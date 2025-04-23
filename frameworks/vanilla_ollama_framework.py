import os
from typing import Any

from openai import OpenAI
from loguru import logger

from frameworks.base import BaseFramework, experiment


class VanillaOllamaFramework(BaseFramework):
    """_summary_
    https://ollama.com/blog/structured-outputs
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.openai_client = OpenAI(base_url=os.environ['OLLAMA_HOST'], api_key="1")
        
        logger.info("Ollama 클라이언트가 초기화되었습니다.")

    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            response = self.openai_client.beta.chat.completions.parse(
                model=self.llm_model,
                response_format=self.response_model,
                messages=[
                    {"role": "user", "content": self.prompt.format(**inputs)}
                ],
            )
            return response.choices[0].message.parsed

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        logger.info(f"실행 결과 - 성공률: {percent_successful}, 응답 수: {len(predictions) if predictions else 0}")
        return predictions, percent_successful, metrics, latencies