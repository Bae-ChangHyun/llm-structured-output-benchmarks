import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Optional

import pandas as pd
import json
from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm
import traceback

from data_sources.data_models import ner_model
from config.config_checker import compatibility_checker, FrameworkCompatibilityError

def response_parsing(response: Any) -> Any:
    if isinstance(response, list):
        response = {
            member.value if isinstance(member, Enum) else member for member in response
        }
    elif is_dataclass(response):
        response = asdict(response)
    elif isinstance(response, BaseModel):
        response = response.model_dump(exclude_none=True)
    return response


def calculate_metrics(
    y_true: dict[str, list[str]], y_pred: dict[str, list[str]]
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """Calculate the total True positives, False positives and False negatives for each entity in the NER task.

    Args:
        y_true (dict[str, list[str]]): The actual labels in the format {"entity1": ["value1", "value2"], "entity2": ["value3"]}
        y_pred (dict[str, list[str]]): The predicted labels in the format {"entity1": ["value1", "value2"], "entity2": ["value3"]}

    Returns:
        tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, dict[str, float]]]: True positives, False positives and False negatives for each entity.
    """
    tp, fp, fn = {}, {}, {}
    for entity in y_true:
        tp[entity] = 0
        fp[entity] = 0
        fn[entity] = 0

        true_values = set(y_true.get(entity, []))
        pred_values = set(y_pred.get(entity, []))

        tp[entity] += len(true_values & pred_values)
        fp[entity] += len(pred_values - true_values)
        fn[entity] += len(true_values - pred_values)

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def experiment(
    n_runs: int = 10,
    expected_response: Any = None,
) -> Callable[..., tuple[list[Any], int, Optional[dict], list[list[float]]]]:
    """Decorator to run an LLM call function multiple times and return the responses

    Args:
        n_runs (int): Number of times to run the function
        expected_response (Any): The expected response. If provided, the decorator will calculate accurary too.

    Returns:
        Callable[..., Tuple[List[Any], int, Optional[dict], list[list[float]]]]: A function that returns a list of outputs from the function runs, percent of successful runs, metrics if expected_response is provided else None and list of latencies for each call.
    """

    def experiment_decorator(func):
        def wrapper(*args, **kwargs):
            # self 객체 (BaseFramework 인스턴스) 가져오기
            self = args[0]
            # config에서 api_delay_seconds 값을 가져오기 (없으면 0)
            api_delay_seconds = getattr(self, "api_delay_seconds", 0)

            responses, latencies = [], []
            for i in tqdm(range(n_runs), leave=False):
                try:
                    start_time = time.time()
                    logger.debug(f"실험 실행 {i+1}/{n_runs} 시작")
                    response = func(*args, **kwargs)
                    end_time = time.time()
                    
                    logger.debug(f"Response: {str(response)[:200]}...")
                    response = response_parsing(response)

                    if "classes" in response:
                        response = response_parsing(response["classes"])

                    responses.append(response)
                    latencies.append(end_time - start_time)
                    logger.debug(f"실험 실행 {i+1}/{n_runs} Success (Time: {end_time - start_time:.2f}초)")
                    if api_delay_seconds > 0:
                        time.sleep(api_delay_seconds)
                except Exception as e:
                    logger.error(f"실험 실행 {i+1}/{n_runs} Failure: {str(e)}")

                    logger.error(traceback.format_exc())

            num_successful = len(responses)
            percent_successful = num_successful / n_runs
            logger.info(f"총 {n_runs}회 시도 중 {num_successful}회 성공 (성공률: {percent_successful:.2%})")

            framework_metrics = []
            for response in responses:
                framework_metrics.append(calculate_metrics(expected_response, response))

            return (
                responses,
                percent_successful,
                framework_metrics if expected_response else None,
                latencies,
            )

        return wrapper

    return experiment_decorator


class BaseFramework(ABC):
    prompt: str
    llm_model: str
    llm_provider: str
    base_url: str
    retries: int
    source_data_pickle_path: str
    sample_rows: int
    response_model: Any
    device: str
    api_delay_seconds: float  # API 요청 사이의 지연 시간(초)
    description_path: str

    def __init__(self, *args, **kwargs) -> None:
        self.prompt = kwargs.get("prompt", "")
        self.llm_model = kwargs.get("llm_model", "gpt-3.5-turbo")
        self.llm_provider = kwargs.get("llm_provider", "openai")
        self.base_url = kwargs.get("base_url", os.environ.get("OLLAMA_HOST", ""))
        self.retries = kwargs.get("retries", 0)
        self.device = kwargs.get("device", "cpu")
        self.api_delay_seconds = kwargs.get("api_delay_seconds", 0)  # API 지연 시간 설정
        self.description_path = kwargs.get("description_path", "")

        # Check framework compatibility with model host
        framework_name = self.__class__.__name__
        compatibility_checker.check_compatibility(framework_name, self.llm_provider)
        
        source_data_pickle_path = kwargs.get("source_data_pickle_path", "")

        # Load the data
        if source_data_pickle_path:
            self.source_data = pd.read_pickle(source_data_pickle_path)

            sample_rows = kwargs.get("sample_rows", 0)
            if sample_rows:
                self.source_data = self.source_data.sample(sample_rows)
                self.source_data = self.source_data.reset_index(drop=True)
            logger.info(f"Loaded source data from {source_data_pickle_path}")
        else:
            self.source_data = None
        


        self.entities = list({key for d in self.source_data["labels"] for key in d.keys()})
        
        if self.description_path != "":
            with open(self.description_path, "r", encoding="utf-8") as file:
                self.descriptions = json.load(file)
        self.response_model = ner_model(self.entities, self.descriptions)

    @abstractmethod
    def run(self, n_runs: int, expected_response: Any, *args, **kwargs): ...
