import json
from typing import Any

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import pipeline

from frameworks.base import BaseFramework, experiment


class LMFormatEnforcerFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.parser = JsonSchemaParser(self.response_model.schema())
        max_length = kwargs.get("max_length", 4096)

        if self.llm_provider == "transformers":
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                device_map=self.device,
                max_length=max_length,
            )
            self.prefix_function = build_transformers_prefix_allowed_tokens_fn(
                self.hf_pipeline.tokenizer, self.parser
            )
        else:
            raise ValueError(f"Model host: {self.llm_provider} not supported")

    def run(
        self, max_tries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(max_tries=max_tries, expected_response=expected_response)
        def run_experiment(inputs):
            prompt = self.prompt.format(
                json_schema=self.response_model.schema(), **inputs
            )
            response = self.hf_pipeline(
                prompt, prefix_allowed_tokens_fn=self.prefix_function
            )
            response = response[0]["generated_text"][len(prompt) :].strip()
            response = self.response_model(**json.loads(response))
            return response

        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        return predictions, percent_successful, metrics, latencies
