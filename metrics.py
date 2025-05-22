import itertools
import re

import numpy as np
import pandas as pd


def format_framework_name(framework_name, model_host=None):
    """프레임워크 이름과 모델 이름을 포맷팅하는 함수.
    
    Args:
        framework_name (str): '프레임워크_모델이름' 형식의 문자열
        model_host (str, optional): 모델 패밀리 이름
        
    Returns:
        tuple: (프레임워크 이름, 모델 이름) 형식의 튜플
    """
    if '_' in framework_name:
        # 첫 번째 _ 를 기준으로 프레임워크 이름과 모델 이름 분리
        parts = framework_name.split('_', 1)
        framework = parts[0].replace("Framework", "")
        model = parts[1]
        return framework, model
    else:
        return framework_name.replace("Framework", ""), "unknown"


def reliability_metric(percent_successful: dict[str, list[float]], model_hosts=None):
    # 프레임워크와 모델 이름 분리
    frameworks = []
    models = []
    reliabilities = []
    
    if model_hosts is None:
        model_hosts = {}
    
    for key, values in percent_successful.items():
        framework, model = format_framework_name(key)
        
        # 모델 패밀리 정보가 있으면 추가
        model_host = model_hosts.get(key, {}).get("host", "")
        if model_host:
            model_display = f"{model}({model_host})"
        else:
            model_display = model
            
        frameworks.append(framework)
        models.append(model_display)
        # 빈 배열 체크 추가
        if len(values) > 0:
            reliabilities.append(np.mean(values))
        else:
            reliabilities.append(0.0)  # 빈 배열인 경우 0으로 처리
    
    # 데이터프레임 생성 (프레임워크와 모델을 별도 컬럼으로)
    reliability_df = pd.DataFrame({
        "Framework": frameworks,
        "Model(host)": models,
        "Reliability": reliabilities
    })
    
    reliability_df = reliability_df.round(3)
    reliability_df = reliability_df.sort_values(by="Reliability", ascending=False)
    return reliability_df


def latency_metric(latencies: dict[str, list[float]], percentile: int = 95, model_hosts=None):
    # Flatten the list of latencies
    latencies = {
        key: list(itertools.chain.from_iterable(value))
        for key, value in latencies.items()
    }

    # 프레임워크와 모델 이름 분리하고 백분위 계산
    frameworks = []
    models = []
    latency_values = []
    
    if model_hosts is None:
        model_hosts = {}
    
    for key, values in latencies.items():
        framework, model = format_framework_name(key)
        
        # 모델 패밀리 정보가 있으면 추가
        model_host = model_hosts.get(key, {}).get("host", "unknown")

        model_display = f"{model}({model_host})"
            
        frameworks.append(framework)
        models.append(model_display)
        # 빈 배열 체크 추가
        if len(values) > 0:
            latency_values.append(np.percentile(values, percentile))
        else:
            latency_values.append(0.0)  # 빈 배열인 경우 0으로 처리
    
    # 데이터프레임 생성 (프레임워크와 모델을 별도 컬럼으로)
    latency_df = pd.DataFrame({
        "Framework": frameworks,
        "Model(host)": models,
        f"Latency_p{percentile}(s)": latency_values
    })
    
    latency_df = latency_df.round(3)
    latency_df = latency_df.sort_values(by=f"Latency_p{percentile}(s)", ascending=True)
    return latency_df


def ner_micro_metrics(results: dict[str, dict], ground_truths=None):
    """NER 작업에 대한 마이크로 평가 지표를 계산합니다.
    
    Args:
        results (dict): 프레임워크별 예측 결과가 포함된 딕셔너리
        ground_truths (list, optional): 정답 데이터. None인 경우 results에 저장된 metrics 사용
        
    Returns:
        pd.DataFrame: 마이크로 정밀도, 재현율, F1 점수가 포함된 데이터프레임
    """
    micro_metrics = {
            "Framework": [],
            "Model(host)": [],
            "micro_precision": [],
            "micro_recall": [],
            "micro_f1": []
        }
    
    for framework, values in results.items():
        # ground_truths가 제공되지 않은 경우 저장된 metrics 사용
       
        tp_total, fp_total, fn_total = 0, 0, 0
        predictions = values["predictions"]
        
        # predictions가 비어 있는지 확인
        if not predictions or not ground_truths:
            # 빈 예측 결과 처리
            framework_name, model_name = format_framework_name(framework)
            model_host = values.get("llm_provider", values.get("llm_model_host", "unknown"))
            model_display = f"{model_name}({model_host})"
            
            micro_metrics["Framework"].append(framework_name)
            micro_metrics["Model(host)"].append(model_display)
            micro_metrics["micro_precision"].append(0.0)
            micro_metrics["micro_recall"].append(0.0)
            micro_metrics["micro_f1"].append(0.0)
            continue
        
        # 각 예측마다 true positives, false positives, false negatives 계산
        for i, pred_runs in enumerate(predictions):
            # i가 ground_truths의 범위를 벗어나지 않는지 확인
            if i >= len(ground_truths):
                continue
                
            truth = ground_truths[i]
            
            # 각 실행마다 집계
            for pred in pred_runs:
                # 카테고리별로 집계
                for category in set(list(pred.keys()) + list(truth.keys())):
                    pred_entities = set(pred.get(category, []))
                    true_entities = set(truth.get(category, []))
                    
                    # True Positives: 예측과 실제가 일치하는 항목
                    true_positives = len(pred_entities.intersection(true_entities))
                    
                    # False Positives: 예측은 했지만 실제론 없는 항목
                    false_positives = len(pred_entities - true_entities)
                    
                    # False Negatives: 예측하지 못한 실제 항목
                    false_negatives = len(true_entities - pred_entities)
                    
                    tp_total += true_positives
                    fp_total += false_positives
                    fn_total += false_negatives

        # 정밀도, 재현율, F1 계산
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0
        )

        # 프레임워크 이름과 모델 이름 분리
        framework_name, model_name = format_framework_name(framework)
        
        # 모델 패밀리 정보가 있으면 추가
        model_host = values.get("llm_provider",  values.get("llm_model_host", "unknown"))
        model_display = f"{model_name}({model_host})"
        
        micro_metrics["Framework"].append(framework_name)
        micro_metrics["Model(host)"].append(model_display)
        micro_metrics["micro_precision"].append(micro_precision)
        micro_metrics["micro_recall"].append(micro_recall)
        micro_metrics["micro_f1"].append(micro_f1)

    return pd.DataFrame(micro_metrics)


def combined_metrics(results: dict[str, dict], ground_truths=None, percentile: int = 95, sort_by: str = "micro_f1"):
    """모든 평가 지표(정밀도, 재현율, F1, 신뢰성, 지연 시간)를 하나의 표로 통합합니다.
    
    Args:
        results (dict): 프레임워크별 예측 결과가 포함된 딕셔너리
        ground_truths (list, optional): 정답 데이터. None인 경우 results에 저장된 metrics 사용
        percentile (int, optional): 지연 시간 백분위 기준. 기본값은 95
        sort_by (str, optional): 정렬 기준 ('micro_f1', 'micro_recall', 'micro_precision', 'reliability', 'latency' 중 하나). 기본값은 'micro_f1'
        
    Returns:
        pd.DataFrame: 모든 지표가 포함된 데이터프레임
    """
    # 결과가 비어있는지 확인
    if not results:
        return pd.DataFrame(columns=["Framework", "Model(host)", "micro_precision", "micro_recall", 
                                   "micro_f1", "Reliability", "Latency"])
    
    # ground_truths가 None인지 확인
    if ground_truths is None:
        # 여기서 ground_truths가 필요한 경우의 처리
        # 예를 들어, 각 프레임워크의 results에서 정답을 가져올 수 있다면:
        # ground_truths = next(iter(results.values())).get("ground_truth", [])
        # 아니면 빈 리스트로 처리:
        ground_truths = []
    
    # 데이터 준비
    percent_successful = {
        framework: value.get("percent_successful", [])
        for framework, value in results.items()
    }
    
    latencies = {
        framework: value.get("latencies", [])
        for framework, value in results.items()
    }
    
    # 모델 호스트 정보 추출
    model_hosts = {}
    for framework, value in results.items():
        model_hosts[framework] = {
            "model": value.get("llm_model", "unknown"),
            "host": value.get("llm_provider", value.get("llm_model_host", "unknown"))
        }
    
    try:
        # 각 지표 계산
        ner_df = ner_micro_metrics(results, ground_truths)
        reliability_df = reliability_metric(percent_successful, model_hosts)
        latency_df = latency_metric(latencies, percentile, model_hosts)
        
        # 데이터프레임이 비어있는지 확인
        if ner_df.empty or reliability_df.empty or latency_df.empty:
            print("경고: 하나 이상의 지표 데이터프레임이 비어 있습니다.")
        
        # 데이터프레임 병합
        combined_df = ner_df.merge(
            reliability_df, on=["Framework", "Model(host)"], how="outer"
        ).merge(
            latency_df, on=["Framework", "Model(host)"], how="outer"
        )
        
        # 컬럼명 정리
        combined_df = combined_df.rename(columns={
            f"Latency_p{percentile}(s)": "Latency"
        })
        
        # 누락된 값 처리
        combined_df = combined_df.fillna(0)
        
        # 숫자 반올림
        numeric_cols = ["micro_precision", "micro_recall", "micro_f1", "Reliability", "Latency"]
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].round(3)
        
        # 사용자 정의 정렬 기준으로 정렬
        sort_column = {
            "f1": "micro_f1",
            "micro_f1": "micro_f1",
            "recall": "micro_recall",
            "micro_recall": "micro_recall",
            "precision": "micro_precision",
            "micro_precision": "micro_precision",
            "reliability": "Reliability",
            "latency": "Latency"
        }.get(sort_by.lower(), "micro_f1")
        
        # sort_column이 존재하는지 확인
        if sort_column in combined_df.columns:
            # 지연 시간은 낮을수록 좋으므로 오름차순 정렬, 나머지는 내림차순 정렬
            ascending = (sort_column == "Latency")
            combined_df = combined_df.sort_values(by=sort_column, ascending=ascending)
        else:
            print(f"경고: 정렬 컬럼 '{sort_column}'이 데이터프레임에 존재하지 않습니다.")
        
        # 인덱스 재설정 (인덱스를 순차적으로 새로 부여하고 인덱스 컬럼 제거)
        combined_df = combined_df.reset_index(drop=True)
        
        return combined_df
    
    except Exception as e:
        print(f"combined_metrics 함수 실행 중 오류 발생: {e}")
        # 오류 발생시 빈 데이터프레임 반환
        return pd.DataFrame(columns=["Framework", "Model(host)", "micro_precision", "micro_recall", 
                                   "micro_f1", "Reliability", "Latency"])

