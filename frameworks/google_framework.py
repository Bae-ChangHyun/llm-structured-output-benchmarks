from typing import Any
import json
import os
import traceback

import google.generativeai as genai
from loguru import logger
from pydantic import BaseModel

from frameworks.base import BaseFramework, experiment


class GoogleFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        genai.configure()
        self.client = genai.GenerativeModel(self.llm_model)
    
    def run(
        self, retries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(retries=retries, expected_response=expected_response)
        def run_experiment(inputs):
            #https://ai.google.dev/gemini-api/docs/structured-output?hl=ko&lang=python
            #https://ai.google.dev/gemini-api/docs/structured-output?hl=ko&lang=rest

            response = self.client.generate_content(
                self.prompt.format(**inputs),
                generation_config={
                    "response_schema": self.response_model,  
                    "response_mime_type":"application/json"
                },
            )
            return json.loads(response.candidates[0].content.parts[0].text)
           
        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        logger.info(f"실행 결과 - 성공률: {percent_successful}, 응답 수: {len(predictions) if predictions else 0}")
        return predictions, percent_successful, metrics, latencies
