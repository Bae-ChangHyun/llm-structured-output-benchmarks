from typing import Any
import os

from openai import OpenAI
from loguru import logger

from frameworks.base import BaseFramework, experiment


class VanillaOpenAIFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if self.llm_model_host == "openai":
            self.client = OpenAI()
            logger.debug("OpenAI 클라이언트가 초기화되었습니다.")
        elif self.llm_model_host == "ollama":
            self.client = OpenAI(
                base_url=self.host,
                api_key="ollama",
            )
            logger.debug("Ollama 클라이언트가 초기화되었습니다.")
        elif self.llm_model_host == "google":
            self.client = OpenAI(
                base_url=self.host,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
            logger.debug("Google 클라이언트가 초기화되었습니다.")

    def run(
        self, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            response = self.client.beta.chat.completions.parse(
                model=self.llm_model,
                response_format=self.response_model,
                messages=[
                    {"role": "user", "content": self.prompt.format(**inputs)}
                ],
            )
            logger.debug(f"예측 결과: {response.choices[0].message.parsed}")
            return response.choices[0].message.parsed

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        logger.info(f"실행 결과 - 성공률: {percent_successful}, 응답 수: {len(predictions) if predictions else 0}")
        return predictions, percent_successful, metrics, latencies
