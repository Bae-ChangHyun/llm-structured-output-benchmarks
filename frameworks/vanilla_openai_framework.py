from typing import Any
import os

from openai import OpenAI
from loguru import logger

from frameworks.base import BaseFramework, experiment


class VanillaOpenAIFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set in the environment variables!")
            raise ValueError("Export OEPN_API_KEY in your environment variables.")
            
        self.openai_client = OpenAI()
        
        logger.info("OpenAI 클라이언트가 초기화되었습니다.")

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
