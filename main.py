import os
import pickle

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
    logger.info(f"사용 디바이스: {device} (로컬 모델용)")

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
            }

            framework_instance = factory(
                config_key, task=task, device=device, **config["init_kwargs"]
            )
            logger.info(f"Using {type(framework_instance)}")

            if isinstance(framework_instance.source_data, pd.DataFrame):
                for row in tqdm(
                    framework_instance.source_data.itertuples(),
                    desc=f"Running {framework_instance.task}",
                    total=len(framework_instance.source_data),
                ):
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

            with open(f"{directory}/{config_key}.pkl", "wb") as file:
                pickle.dump(results, file)
                logger.info(f"Results saved to {directory}/{config_key}.pkl")


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
    config_path: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to configuration YAML file used for the benchmark (for reference only)",
    ),
):
    """
    벤치마크 결과 생성: 벤치마크 결과를 분석하고 출력합니다.
    
    예시:
    - 기본 결과 생성: python -m main generate-results
    - 다른 태스크 지정: python -m main generate-results --task ner
    - 다른 결과 디렉토리 지정: python -m main generate-results --results-path ./custom_results/multilabel_classification
    - 참조용 설정 파일 지정: python -m main generate-results --config my_config.yaml
    """
    allowed_tasks = ["multilabel_classification", "ner", "synthetic_data_generation"]
    if task not in allowed_tasks:
        raise ValueError(f"{task} is not allowed. Allowed values are {allowed_tasks}")
    
    if not results_data_path:
        results_data_path = f"./results/{task}"
        
    logger.info(f"결과 분석 중: {results_data_path}")
    if not os.path.exists(results_data_path):
        logger.error(f"결과 디렉토리를 찾을 수 없습니다: {results_data_path}")
        raise typer.Exit(code=1)

    # Combine results from different frameworks
    results = {}

    for file_name in os.listdir(results_data_path):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(results_data_path, file_name)
            with open(file_path, "rb") as file:
                framework_results = pickle.load(file)
                results.update(framework_results)

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
