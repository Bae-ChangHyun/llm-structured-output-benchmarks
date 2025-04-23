import os
import pickle
import time

import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from frameworks import factory, metrics
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()


@app.command()
def run_benchmark(
    config_path: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration YAML file with framework settings",
    )
):
    """
    벤치마크 실행: 지정된 config 파일에 정의된 모든 프레임워크와 모델에 대해 벤치마크를 실행합니다.
    
    예시:
    - 기본 설정으로 실행: python -m main run-benchmark
    - 다른 설정 파일로 실행: python -m main run-benchmark --config my_config.yaml
    - 짧은 옵션으로 실행: python -m main run-benchmark -c experiment_config.yaml
    """
    if not os.path.exists(config_path):
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        raise typer.Exit(code=1)
        
    logger.info(f"설정 파일 사용: {config_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for local models")

    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)

    for config_key, config_values in configs.items():
        results = {}
        for config in config_values:
            results[config_key] = {}
            task = config["task"]
            n_runs = config["n_runs"]
            run_results = {
                "predictions": [],
                "percent_successful": [],
                "metrics": [],
                "latencies": [],
                "llm_model": config["init_kwargs"].get("llm_model", "unknown"),  # 모델 정보 저장
                "llm_model_family": config["init_kwargs"].get("llm_model_family", "unknown"),  # 모델 패밀리 정보 저장
            }

            framework_instance = factory(
                config_key, task=task, device=device, **config["init_kwargs"]
            )
            logger.info(f"Using {type(framework_instance)}")
            
            # API 지연 시간 설정 확인
            api_delay_seconds = config["init_kwargs"].get("api_delay_seconds", 0)
            is_first_sample = True

            if isinstance(framework_instance.source_data, pd.DataFrame):
                for row in tqdm(
                    framework_instance.source_data.itertuples(),
                    desc=f"Running {framework_instance.task}",
                    total=len(framework_instance.source_data),
                ):
                    # 데이터 샘플 간 API 지연 적용 (첫 번째 샘플 제외)
                    if not is_first_sample and api_delay_seconds > 0:
                        time.sleep(api_delay_seconds)
                    else:
                        is_first_sample = False
                        
                    if isinstance(row.labels, list):
                        labels = set(row.labels)
                    else:
                        labels = row.labels

                    # logger.info(f"Actual Text: {row.text}")
                    # logger.info(f"Actual Labels: {labels}")
                    predictions, percent_successful, framework_metrics, latencies = (
                        framework_instance.run(
                            inputs={"text": row.text},
                            n_runs=n_runs,
                            expected_response=labels,
                            task=task,
                        )
                    )
                    # logger.info(f"Predicted Labels: {predictions}")
                    run_results["metrics"].append(framework_metrics)
                    run_results["predictions"].append(predictions)
                    run_results["percent_successful"].append(percent_successful)
                    run_results["latencies"].append(latencies)
            else:
                predictions, percent_successful, _, latencies = (
                    framework_instance.run(
                        n_runs=n_runs,
                        task=task,
                    )
                )
                # logger.info(f"Predicted Labels: {predictions}")
                run_results["predictions"].append(predictions)
                run_results["percent_successful"].append(percent_successful)
                run_results["latencies"].append(latencies)

            results[config_key] = run_results

            # logger.info(f"Results:\n{results}")

            directory = f"results/{task}"
            os.makedirs(directory, exist_ok=True)
            
            # 프레임워크 이름에 모델 이름 추가
            model_name = config["init_kwargs"].get("llm_model", "unknown")
            key_with_model = f"{config_key}_{model_name}"
            
            # 모델 이름이 포함된 키로 결과 저장
            with open(f"{directory}/{key_with_model}.pkl", "wb") as file:
                pickle.dump({key_with_model: run_results}, file)
                logger.info(f"Results saved to {directory}/{key_with_model}.pkl")


@app.command()
def generate_results(
    results_data_path: str = typer.Option(
        "",
        "--results-path",
        help="Path to directory containing benchmark results (default: ./results/[task])",
    ),
    task: str = typer.Option(
        "multilabel_classification",
        "--task", "-t", 
        help="Task to generate results for: multilabel_classification, ner, or synthetic_data_generation",
    ),
):
    allowed_tasks = ["multilabel_classification", "ner", "synthetic_data_generation"]
    if task not in allowed_tasks:
        raise ValueError(f"{task} is not allowed. Allowed values are {allowed_tasks}")
    
    if not results_data_path:
        results_data_path = f"./results/{task}"

    # Combine results from different frameworks
    results = {}
    framework_model_info = {}

    for file_name in os.listdir(results_data_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(results_data_path, file_name)
            with open(file_path, "rb") as file:
                framework_results = pickle.load(file)
                
                # 새로운 결과 파일 형식 (프레임워크_모델명.pkl)과 이전 형식(프레임워크.pkl) 모두 지원
                for key, value in framework_results.items():
                    # 모델 정보를 framework_model_info에 저장
                    if "llm_model" in value:
                        framework_model_info[key] = {
                            "model": value["llm_model"],
                            "family": value.get("llm_model_family", "unknown")
                        }
                    # 아직 모델 정보가 없는 이전 형식 결과의 경우 기본값 설정
                    else:
                        framework_model_info[key] = {
                            "model": "unknown",
                            "family": "unknown"
                        }
                
                results.update(framework_results)

    # 로그에 모델 정보 표시
    logger.info("Framework and Model Information:")
    for framework, info in framework_model_info.items():
        logger.info(f"{framework}: Model = {info['model']}, Family = {info['family']}")

    # Reliability
    percent_successful = {
        framework: value["percent_successful"]
        for framework, value in results.items()
    }
    logger.info(f"Reliability:\n{metrics.reliability_metric(percent_successful)}")

    # Latency
    latencies = {
        framework: value["latencies"]
        for framework, value in results.items()
    }
    logger.info(f"Latencies:\n{metrics.latency_metric(latencies, 95)}")

    # NER Micro Metrics
    if task == "ner":
        micro_metrics_df = metrics.ner_micro_metrics(results)
        micro_metrics_df = micro_metrics_df.sort_values(by="micro_f1", ascending=False)
        logger.info(f"NER Micro Metrics:\n{micro_metrics_df}")

    # Variety
    if task == "synthetic_data_generation":
        predictions = {
            framework: value["predictions"][0]
            for framework, value in results.items()
        }
        logger.info(f"Variety:\n{metrics.variety_metric(predictions)}")

if __name__ == "__main__":
    app()
