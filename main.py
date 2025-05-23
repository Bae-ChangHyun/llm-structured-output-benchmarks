import os
import sys
import pickle
import time
from datetime import datetime
import glob

import pandas as pd
import torch
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from frameworks import factory
from dotenv import load_dotenv
import shutil

import metrics

load_dotenv()

logger.remove()
logger.add(sys.stderr, level=os.environ.get("LOG_LEVEL", "INFO").upper())
pd.set_option('display.max_rows', None)

# Typer 앱 생성 시 자동 완성 기능 비활성화하고 도움말 옵션 설정
app = typer.Typer(
    help="LLM 구조화된 출력 벤치마크 도구. 다양한 프레임워크와 모델의 성능을 비교합니다.",
    add_help_option=True,
    rich_help_panel="rich",
    pretty_exceptions_enable=True,
    pretty_exceptions_short=False,
    no_args_is_help=True,  # 인수 없이 실행 시 도움말 표시
    add_completion=False   # 자동 완성 기능 비활성화
)

@app.command(help="벤치마크 실행: 지정된 프레임워크와 모델에 대해 성능 테스트를 실행하고 결과를 저장합니다.")
def run_benchmark(
    config_path: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="프레임워크 및 모델 설정 YAML 파일의 경로입니다.",
    ),
    results_path: str = typer.Option(
        f"results/{datetime.now().strftime('%Y-%m-%d')}",
        "--results","-r",
        help="벤치마크 결과를 저장할 폴더명입니다. 기본값: 현재날짜 (results 하위)",
    ),
):
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        raise typer.Exit(code=1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    with open(config_path, "r") as file:
        configs = yaml.safe_load(file)

    for config_key, config_values in configs.items():
            results = {}
            for config in config_values:
                results[config_key] = {}
                retries = config["retries"]
                run_results = {
                    "predictions": [],
                    "percent_successful": [],
                    "latencies": [],
                    "llm_model": config["init_kwargs"].get("llm_model", "unknown"),  # 실제 모델 정보 저장
                    "llm_model_alias": config["init_kwargs"].get("llm_model_alias", ""),  # 모델 별칭 저장
                    "llm_provider": config["init_kwargs"].get("llm_provider", "unknown"),  # 모델 패밀리 정보 저장
                    "source_data_path": config["init_kwargs"].get("source_data_pickle_path", ""),  # 소스 데이터 경로 저장
                }

                framework_instance = factory(
                    config_key, device=device, **config["init_kwargs"]
                )
                logger.info(f"Using {type(framework_instance)}")
                
                # API 지연 시간 설정 확인(for free API)
                api_delay_seconds = config["init_kwargs"].get("api_delay_seconds", 0)
                is_first_sample = True

                if isinstance(framework_instance.source_data, pd.DataFrame):
                    for row in tqdm(
                        framework_instance.source_data.itertuples(),
                        desc=f"Running NER benchmark",
                        total=len(framework_instance.source_data),
                    ):
                        if not is_first_sample and api_delay_seconds > 0:
                            time.sleep(api_delay_seconds)
                        else:
                            is_first_sample = False
                            
                        if isinstance(row.labels, list):
                            labels = set(row.labels)
                        else:
                            labels = row.labels

                        predictions, percent_successful, _, latencies = (
                            framework_instance.run(
                                inputs={"text": row.text},
                                retries=retries,
                                expected_response=labels,
                            )
                        )
                        run_results["predictions"].append(predictions)
                        run_results["percent_successful"].append(percent_successful)
                        run_results["latencies"].append(latencies)
                else:
                    predictions, percent_successful, _, latencies = (
                        framework_instance.run(
                            retries=retries,
                        )
                    )
                    run_results["predictions"].append(predictions)
                    run_results["percent_successful"].append(percent_successful)
                    run_results["latencies"].append(latencies)

                results[config_key] = run_results

                final_results_path = f"results/{results_path}"
                os.makedirs(final_results_path, exist_ok=True)
                shutil.copy(config_path, f"{final_results_path}/{config_key}.yaml")
                
                # llm_model_alias를 우선 사용하고, 없으면 llm_model 사용
                model_name = config["init_kwargs"].get("llm_model_alias", 
                                                    config["init_kwargs"].get("llm_model", "unknown"))
                key_with_model = f"{config_key}_{model_name}"
                
                # 모델 이름이 포함된 키로 결과 저장
                with open(f"{final_results_path}/{key_with_model}.pkl", "wb") as file:
                    pickle.dump({key_with_model: run_results}, file)
                    logger.info(f"Results saved to {final_results_path}/{key_with_model}.pkl")

@app.command(help="벤치마크 결과 분석: 저장된 벤치마크 결과를 로드하여 성능 지표를 분석하고 비교 테이블을 출력합니다.")
def show_results(
    ground_truth_path: str = typer.Option(
        "",
        "--ground-truth", "-g",
        help="정답 라벨이 포함된 PKL 파일 경로. 지정하지 않으면 벤치마크 실행시 사용된 소스 데이터 경로에서 로드합니다.",
    ),
    results_data_paths: list[str] = typer.Argument(
        None,
        help="하나 이상의 결과 폴더 경로들. 공백으로 구분하여 여러 경로 지정 가능: path1 path2 path3",
    ),
    sort_by: str = typer.Option(
        "f1",
        "--sort-by", "-s",
        help="정렬 기준 ('f1', 'recall', 'precision', 'reliability', 'latency' 중 하나)",
    ),
):
    
    # 결과 경로가 지정되지 않은 경우 기본값으로 results 디렉토리의 모든 하위 폴더 사용
    if results_data_paths is None or len(results_data_paths) == 0:
        results_dir = "results"
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            # results 디렉토리 내의 모든 하위 디렉토리를 가져옴
            subdirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                      if os.path.isdir(os.path.join(results_dir, d))]
            
            if subdirs:
                results_data_paths = subdirs
                logger.debug(f"결과 경로가 지정되지 않아 results 디렉토리의 모든 하위 폴더를 사용합니다: {results_data_paths}")
            else:
                # 하위 디렉토리가 없으면 results 디렉토리 자체를 사용
                results_data_paths = [results_dir]
                logger.debug(f"results 디렉토리에 하위 폴더가 없어 results 디렉토리 자체를 사용합니다.")
    
    # -r 옵션을 유지하기 위한 추가 코드 
    # typer.Argument 대신 typer.Option을 사용하여 -r 옵션 지원
    if len(results_data_paths) == 1 and ',' in results_data_paths[0]:
        results_data_paths = [path.strip() for path in results_data_paths[0].split(',')]
        logger.debug(f"콤마로 구분된 경로를 분리하여 사용합니다: {results_data_paths}")

    # Combine results from different frameworks
    results = {}
    framework_model_info = {}
    source_data_paths = {}

    # 여러 경로를 순회
    for results_data_path in results_data_paths:
        if not os.path.exists(results_data_path):
            logger.warning(f"결과 폴더를 찾을 수 없습니다: {results_data_path}")
            continue
        
        for file_name in os.listdir(results_data_path):
            if file_name.endswith(".pkl"):
                file_path = os.path.join(results_data_path, file_name)
                with open(file_path, "rb") as file:
                    try:
                        framework_results = pickle.load(file)
                        
                        for key, value in framework_results.items():
                            # 모델 정보를 framework_model_info에 저장
                            framework_model_info[key] = {
                                "model": value["llm_model"] if "llm_model" in value else "unknown",
                                "alias": value.get("llm_model_alias", ""),  # 별칭 정보 추가
                                "host": value.get("llm_provider", value.get("llm_model_host", "unknown")),
                            }
                            # 소스 데이터 경로 저장
                            source_data_path = value.get("source_data_path", "")
                            if source_data_path:
                                source_data_paths[key] = source_data_path
                        
                        results.update(framework_results)
                        logger.debug(f"로드된 결과: {file_path}")
                    except Exception as e:
                        logger.error(f"파일 로드 중 오류 발생: {file_path}: {e}")

    if not results:
        logger.error("처리할 결과 파일이 없습니다.")
        raise typer.Exit(code=1)

    # 로그에 모델 정보 표시
    logger.info(f"총 {len(results)} 개의 모델 결과 로드됨")
    logger.debug("Framework and Model Information:")
    for framework, info in framework_model_info.items():
        # llm_model_alias가 있으면 사용하고, 없거나 빈 문자열이면 llm_model 사용
        display_model = info["alias"] if info.get("alias") else info["model"]
        logger.debug(f"{framework}: Model = {display_model}({info['host']})")
    
    # --ground-truth 옵션이 지정된 경우
    if ground_truth_path:
        logger.info(f"Loaded custom ground truth {ground_truth_path}")
        source_path = ground_truth_path
    # --ground-truth 옵션이 지정되지 않은 경우, 저장된 source_data_path에서 데이터 로드
    else:
        if source_data_paths:
            # source_data_paths가 모두 동일한지 확인
            unique_paths = set(source_data_paths.values())
            
            if len(unique_paths) > 1:
                logger.warning(f"서로 다른 소스 데이터 경로가 발견되었습니다:")
                for framework, path in source_data_paths.items():
                    logger.warning(f" - {framework}: {path}")
                
                # 가장 많이 사용된 경로를 선택
                path_counts = {}
                for path in source_data_paths.values():
                    path_counts[path] = path_counts.get(path, 0) + 1
                
                source_path = max(path_counts.items(), key=lambda x: x[1])[0]
                logger.debug(f"가장 많이 사용된 소스 데이터 경로를 선택합니다: {source_path}")
            else:
                # 모든 경로가 동일한 경우
                source_path = next(iter(unique_paths))
                logger.info(f"Loaded ground truth labels from results: {source_path}")
        else:
            logger.error("소스 데이터 경로가 결과에 포함되어 있지 않으며, ground-truth도 지정되지 않았습니다.")
            raise typer.Exit(code=1)
            
    # 선택된 경로에서 데이터 로드
    try:
        source_data = pd.read_pickle(source_path)
        ground_truths = source_data["labels"].tolist()
        logger.info(f"Ground truth labels loaded successfully: {len(ground_truths)} entries")
        
        # 통합된 지표 계산 및 표시 (정렬 기준 적용)
        logger.info(f"모든 지표 (정렬 기준: {sort_by}):")
        combined_df = metrics.combined_metrics(results, ground_truths=ground_truths, sort_by=sort_by)
        logger.info(f"\n{combined_df}\n")
    except Exception as e:
        logger.error(f"Failed to load ground truth labels from source data: {e}")
        raise typer.Exit(code=1)

@app.command(help="Streamlit을 사용하여 예측 결과와 실제값 비교 시각화: 결과를 인터랙티브하게 시각화하고 분석합니다.")
def visualize(
    port: int = typer.Option(
        8501,
        "--port", "-p",
        help="Streamlit 서버 포트 번호입니다.",
    ),
    fix_arrow_error: bool = typer.Option(
        True,
        "--fix-arrow-error", "-f",
        help="PyArrow 변환 오류를 방지하기 위해 혼합 타입 컬럼을 문자열로 변환합니다.",
    ),
):
    """Streamlit 앱을 실행하여 예측 결과와 실제값을 비교 시각화합니다."""
    import subprocess
    import sys
    import os
    
    streamlit_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_app.py")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        streamlit_file,
        "--server.port", str(port)
    ]
    
    # 데이터 변환 플래그 추가
    if fix_arrow_error:
        cmd.extend(["--", "--fix-arrow-error"])
    
    try:
        logger.info(f"Streamlit 시각화 도구를 포트 {port}에서 시작합니다...")
        if fix_arrow_error:
            logger.info("데이터 변환 오류 수정 활성화: 혼합 타입 컬럼을 문자열로 변환합니다.")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Streamlit 앱이 종료되었습니다.")
    except Exception as e:
        logger.error(f"Streamlit 앱 실행 중 오류가 발생했습니다: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
