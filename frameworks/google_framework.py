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
    
    def _convert_to_gemini_schema(self, schema):
        """일반 JSON 스키마를 Gemini API에 맞는 형식으로 변환"""
        # Gemini API에서 지원하지 않는 필드 제거
        unsupported_fields = ['title', '$schema', 'examples', 'allOf', 'anyOf', 
                             'oneOf', 'not', 'if', 'then', 'else', 'default', 'format', 
                             'enum', 'const', 'pattern', 'minLength', 'maxLength', 'minimum', 
                             'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf',
                             'required', 'dependencies', 'additionalProperties', 'propertyNames',
                             'contains', 'minProperties', 'maxProperties', 'minItems', 'maxItems',
                             'uniqueItems', 'additionalItems',  '$defs', '$ref']
        
        # 스키마가 딕셔너리인 경우 재귀적으로 처리
        if isinstance(schema, dict):
            result = {}
            
            # description 필드가 있으면 먼저 유지 (anyOf 처리 전에 저장)
            if 'description' in schema:
                result['description'] = schema['description']
            
            if 'anyOf' in schema and isinstance(schema['anyOf'], list) and len(schema['anyOf']) > 0:
                first_option = schema['anyOf'][0]
                converted_option = self._convert_to_gemini_schema(first_option)
                
                # 변환된 옵션이 딕셔너리면 원본 스키마의 description을 전달
                if isinstance(converted_option, dict) and 'description' in schema:
                    converted_option['description'] = schema['description']
                
                # description이 이미 결과에 있으면 그대로 유지하면서 나머지 필드 병합
                for key, value in converted_option.items():
                    if key != 'description' or 'description' not in result:
                        result[key] = value
                        
                return result
            
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
        self, max_tries: int, expected_response: Any = None, inputs: dict = {}
    ) -> tuple[list[Any], float, dict, list[list[float]]]:
        @experiment(max_tries=max_tries, expected_response=expected_response)
        def run_experiment(inputs):
            #https://ai.google.dev/gemini-api/docs/structured-output?hl=ko&lang=python
            #https://ai.google.dev/gemini-api/docs/structured-output?hl=ko&lang=rest
            model = genai.GenerativeModel(self.llm_model)
            
            # response_model을 JSON 스키마로 변환
            schema = self.response_model.model_json_schema()
            gemini_schema = self._convert_to_gemini_schema(schema)
            
            # For debugging
            # schema_str = json.dumps(schema)
            # gemini_schema_str = json.dumps(gemini_schema)
            # with open("schema.json", "w") as f:
            #     f.write(schema_str)
            # with open("gemini_schema.json", "w") as f:
            #     f.write(gemini_schema_str)
            
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
