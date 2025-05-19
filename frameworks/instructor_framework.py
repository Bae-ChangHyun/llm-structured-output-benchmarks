import os
from typing import Any
from loguru import logger

import instructor
from openai import OpenAI
import google.generativeai as genai

from frameworks.base import BaseFramework, experiment


class InstructorFramework(BaseFramework):
    # https://python.useinstructor.com
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.llm_model_host == "openai":
            self.instructor_client = instructor.from_openai(
                client = OpenAI())
            logger.debug("OpenAI 클라이언트가 초기화되었습니다.")
        elif self.llm_model_host == "ollama":
            self.instructor_client = instructor.from_openai(
                cleint = OpenAI(
                base_url = os.environ['OLLAMA_HOST'],
                api_key = "ollama",
                ),
                mode=instructor.Mode.JSON)
            logger.debug("Ollama 클라이언트가 초기화되었습니다.")
        elif self.llm_model_host == "google":
            self.instructor_client = instructor.from_gemini(
                client=genai.GenerativeModel(model_name=self.llm_model),
                mode=instructor.Mode.GEMINI_JSON
            )
            logger.debug("Google 클라이언트가 초기화되었습니다.")

    def run(
        self, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response)
        def run_experiment(inputs):
            if self.llm_model_host == "google":
                response = self.instructor_client.chat.completions.create(
                    response_model=self.response_model,
                    max_retries=self.retries,
                    messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
                )
            else:
                response = self.instructor_client.chat.completions.create(
                    model=self.llm_model,
                    response_model=self.response_model,
                    max_retries=self.retries,
                    messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
                )
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
