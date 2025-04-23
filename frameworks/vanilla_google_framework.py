from typing import Any
import json
import os
import traceback

import google.generativeai as genai
from loguru import logger
from pydantic import BaseModel

from frameworks.base import BaseFramework, experiment


class VanillaGoogleFramework(BaseFramework):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY is not set in the environment variables!")
            raise ValueError("Export GOOGLE_API_KEY in your environment variables.")
    
        genai.configure()
    
    def _convert_to_gemini_schema(self, schema):
        """일반 JSON 스키마를 Gemini API에 맞는 형식으로 변환"""
        # Gemini API에서 지원하지 않는 필드 제거
        unsupported_fields = ['title', '$schema', 'description', 'examples', 'allOf', 'anyOf', 
                             'oneOf', 'not', 'if', 'then', 'else', 'default', 'format', 
                             'enum', 'const', 'pattern', 'minLength', 'maxLength', 'minimum', 
                             'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
                             'required', 'dependencies', 'additionalProperties', 'propertyNames',
                             'contains', 'minProperties', 'maxProperties', 'minItems', 'maxItems',
                             'uniqueItems', 'additionalItems']
        
        # 스키마가 딕셔너리인 경우 재귀적으로 처리
        if isinstance(schema, dict):
            result = {}
            
            if 'anyOf' in schema and isinstance(schema['anyOf'], list) and len(schema['anyOf']) > 0:
                first_option = schema['anyOf'][0]
                if isinstance(first_option, dict) and first_option.get('type') == 'array':
                    return self._convert_to_gemini_schema(first_option)
                return self._convert_to_gemini_schema(first_option)
            
            # 속성 처리
            if 'properties' in schema:
                result['properties'] = {}
                for prop_name, prop_schema in schema['properties'].items():
                    result['properties'][prop_name] = self._convert_to_gemini_schema(prop_schema)
            
            # 배열 항목 처리
            if 'items' in schema:
                result['items'] = self._convert_to_gemini_schema(schema['items'])
            
            # 기타 지원되는 키 복사 (unsupported_fields에 없는 키)
            for key, value in schema.items():
                if key not in unsupported_fields and key not in result:
                    result[key] = value
                    
            return result
            
        # IF list, convert each item
        elif isinstance(schema, list):
            return [self._convert_to_gemini_schema(item) for item in schema]
            
        return schema
        
    def run(
        self, task: str, n_runs: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(n_runs=n_runs, expected_response=expected_response, task=task)
        def run_experiment(inputs):
            model = genai.GenerativeModel(self.llm_model)
            
            # response_model을 JSON 스키마로 변환
            schema = self.response_model.model_json_schema()
            gemini_schema = self._convert_to_gemini_schema(schema)
            
            ## For debugging
            #schema_str = json.dumps(schema)
            #gemini_schema_str = json.dumps(gemini_schema)
            #logger.info(f"Original Schema: {schema_str}...")
            #logger.info(f"Converted Schema for Gemini: {gemini_schema_str[:100]}...")
              
            response = model.generate_content(
                self.prompt.format(**inputs),
                generation_config={
                    "response_schema": gemini_schema,  # Using the converted schema for Gemini(orginal if for OpenAI)
                    "response_mime_type":"application/json"
                },
            )
            return json.loads(response.candidates[0].content.parts[0].text)
           
        predictions, percent_successful, metrics, latencies = run_experiment(inputs)
        logger.info(f"실행 결과 - 성공률: {percent_successful}, 응답 수: {len(predictions) if predictions else 0}")
        return predictions, percent_successful, metrics, latencies
 