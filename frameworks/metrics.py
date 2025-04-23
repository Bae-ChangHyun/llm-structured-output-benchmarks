import itertools
import re

import numpy as np
import pandas as pd


def format_framework_name(framework_name, model_family=None):
    """프레임워크 이름과 모델 이름을 포맷팅하는 함수.
    
    Args:
        framework_name (str): '프레임워크_모델이름' 형식의 문자열
        model_family (str, optional): 모델 패밀리 이름
        
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


def reliability_metric(percent_successful: dict[str, list[float]], model_families=None):
    # 프레임워크와 모델 이름 분리
    frameworks = []
    models = []
    reliabilities = []
    
    if model_families is None:
        model_families = {}
    
    for key, values in percent_successful.items():
        framework, model = format_framework_name(key)
        
        # 모델 패밀리 정보가 있으면 추가
        model_family = model_families.get(key, {}).get("family", "")
        if model_family:
            model_display = f"{model}({model_family})"
        else:
            model_display = model
            
        frameworks.append(framework)
        models.append(model_display)
        reliabilities.append(np.mean(values))
    
    # 데이터프레임 생성 (프레임워크와 모델을 별도 컬럼으로)
    reliability_df = pd.DataFrame({
        "Framework": frameworks,
        "Model(host)": models,
        "Reliability": reliabilities
    })
    
    reliability_df = reliability_df.round(3)
    reliability_df = reliability_df.sort_values(by="Reliability", ascending=False)
    return reliability_df


def latency_metric(latencies: dict[str, list[float]], percentile: int = 95, model_families=None):
    # Flatten the list of latencies
    latencies = {
        key: list(itertools.chain.from_iterable(value))
        for key, value in latencies.items()
    }

    # 프레임워크와 모델 이름 분리하고 백분위 계산
    frameworks = []
    models = []
    latency_values = []
    
    if model_families is None:
        model_families = {}
    
    for key, values in latencies.items():
        framework, model = format_framework_name(key)
        
        # 모델 패밀리 정보가 있으면 추가
        model_family = model_families.get(key, {}).get("family", "unknown")

        model_display = f"{model}({model_family})"

            
        frameworks.append(framework)
        models.append(model_display)
        latency_values.append(np.percentile(values, percentile))
    
    # 데이터프레임 생성 (프레임워크와 모델을 별도 컬럼으로)
    latency_df = pd.DataFrame({
        "Framework": frameworks,
        "Model(host)": models,
        f"Latency_p{percentile}(s)": latency_values
    })
    
    latency_df = latency_df.round(3)
    latency_df = latency_df.sort_values(by=f"Latency_p{percentile}(s)", ascending=True)
    return latency_df


def ner_micro_metrics(results: dict[str, list[float]]):
    micro_metrics = {
            "Framework": [],
            "Model(host)": [],
            "micro_precision": [],
            "micro_recall": [],
            "micro_f1": []
        }
    
    for framework, values in results.items():
        tp_total, fp_total, fn_total = 0, 0, 0
        runs = values["metrics"]

        for run in runs:
            for metric in run:
                tp_total += sum(metric["true_positives"].values())
                fp_total += sum(metric["false_positives"].values())
                fn_total += sum(metric["false_negatives"].values())

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
        model_family = values.get("llm_model_family", "unknown")
        model_display = f"{model_name}({model_family})"
        
        micro_metrics["Framework"].append(framework_name)
        micro_metrics["Model(host)"].append(model_display)
        micro_metrics["micro_precision"].append(micro_precision)
        micro_metrics["micro_recall"].append(micro_recall)
        micro_metrics["micro_f1"].append(micro_f1)

    return pd.DataFrame(micro_metrics)
