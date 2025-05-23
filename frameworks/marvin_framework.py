from typing import Any

import marvin

from frameworks.base import BaseFramework, experiment


class MarvinFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        marvin.settings.openai.chat.completions.model = self.llm_model

    def run(
        self, max_tries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(max_tries=max_tries, expected_response=expected_response)
        def run_experiment(inputs):
            response = marvin.cast(self.prompt.format(**inputs), self.response_model)
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies