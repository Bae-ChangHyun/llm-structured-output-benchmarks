# filepath: /home/bch/Project/main_project/scout_partners/llm-ner-benchmarks/frameworks/guradrails_framework.py
import litellm
import openai
from typing import Any

from guardrails import Guard
from openai import OpenAI
from loguru import logger

from frameworks.base import BaseFramework, experiment


class GuardrailsFramework(BaseFramework):
    """
    GuardrailsFramework는 Guardrails 라이브러리를 사용하여 NER 작업을 수행합니다.
    Guardrails는 LLM의 출력을 구조화하고 검증하는 데 사용됩니다.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.guard = Guard.for_pydantic(output_class=self.response_model)

    def run(
        self, max_tries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(max_tries=max_tries, expected_response=expected_response)
        def run_experiment(inputs):
            
            if self.llm_provider == "ollama":
                response = self.guard(
                    litellm.completion,
                    model=f"ollama/{self.llm_model}",
                    api_base = "http://localhost:11434",
                    messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
                    )
            elif self.llm_provider == "openai":
                response = self.guard(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
                    tools=self.guard.json_function_calling_tool([]),
                    tool_choice="required",
                )
            elif self.llm_provider == "google":
                response = self.guard(
                    model=f"gemini/{self.llm_model}",
                    messages=[{"role": "user", "content": self.prompt.format(**inputs)}],
                )
                
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        logger.info(f"실행 결과 - 성공률: {percent_successful}, 응답 수: {len(predictions) if predictions else 0}")
        return predictions, percent_successful, metrics, latencies