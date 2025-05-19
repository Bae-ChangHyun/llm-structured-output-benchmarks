import os
import pickle
import glob
import argparse
import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger

# 명령줄 인수 파서 설정
parser = argparse.ArgumentParser(description='Streamlit NER 벤치마크 시각화 도구')
parser.add_argument('--fix-arrow-error', action='store_true', 
                    help='데이터프레임 컬럼의 혼합 타입을 문자열로 변환하여 Arrow 변환 오류 방지')
args = parser.parse_args()

st.set_page_config(
    page_title="LLM NER 벤치마크 시각화", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("LLM NER 벤치마크 결과 시각화")

# 데이터프레임 컬럼 타입 전처리 함수
def preprocess_dataframe(df):
    """리스트와 비리스트가 혼합된 컬럼을 문자열로 변환하여 Arrow 변환 오류 방지"""
    if not args.fix_arrow_error:
        return df
    
    # 데이터프레임 복사
    processed_df = df.copy()
    
    # 각 컬럼 검사
    for col in processed_df.columns:
        # 샘플 데이터에서 리스트 타입 존재 여부 확인
        has_list = False
        has_non_list = False
        
        sample_values = processed_df[col].dropna().head(100).values
        for val in sample_values:
            if isinstance(val, (list, tuple, np.ndarray)):
                has_list = True
            elif not pd.isna(val):
                has_non_list = True
            
            # 리스트와 비리스트가 모두 발견되면 반복 종료
            if has_list and has_non_list:
                break
        
        # 리스트와 비리스트가 혼합된 컬럼은 문자열로 변환
        if has_list and has_non_list:
            logger.info(f"컬럼 '{col}'에 혼합 타입 발견, 문자열로 변환 중...")
            processed_df[col] = processed_df[col].apply(
                lambda x: str(x) if not pd.isna(x) else x
            )
    
    return processed_df

# 결과 로딩 함수
def load_results():
    results_dir = "results"
    all_results = {}
    framework_model_info = {}
    
    # 결과 디렉토리의 모든 하위 폴더 검색
    result_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
    
    # 결과 디렉토리가 없으면 results 디렉토리 자체 사용
    if not result_dirs:
        result_dirs = [results_dir]
    
    for result_dir in result_dirs:
        # 각 결과 디렉토리에서 PKL 파일 검색
        pkl_files = glob.glob(os.path.join(result_dir, "*.pkl"))
        
        for file_path in pkl_files:
            try:
                with open(file_path, "rb") as file:
                    framework_results = pickle.load(file)
                    all_results.update(framework_results)
                    
                    # 모델 정보 추출
                    for key, value in framework_results.items():
                        framework_model_info[key] = {
                            "model": value.get("llm_model", "unknown"),
                            "host": value.get("llm_model_host", "unknown"),
                            "source_data": value.get("source_data_path", ""),
                            "file_path": file_path  # 파일 경로 저장
                        }
            except Exception as e:
                st.warning(f"파일 로드 중 오류 발생: {file_path}: {str(e)}")
    
    return all_results, framework_model_info

# 메인 함수
def main():
    # 결과 로딩
    with st.spinner("벤치마크 결과 로딩 중..."):
        results, model_info = load_results()
    
    if not results:
        st.error("처리할 결과 파일이 없습니다. results 폴더에 결과가 있는지 확인하세요.")
        return
    
    # 사이드바 - 모델 선택
    st.sidebar.header("모델 선택")
    
    # 파일 경로를 포함한 옵션 생성
    model_options = {}
    for key in results.keys():
        file_path = model_info.get(key, {}).get("file_path", "unknown")
        display_name = f"{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)} - {key}"
        model_options[display_name] = key
    
    selected_display_names = st.sidebar.multiselect(
        "비교할 프레임워크/모델 선택",
        options=list(model_options.keys()),
        default=list(model_options.keys())[:min(3, len(model_options.keys()))]
    )
    
    # 선택된 표시 이름을 실제 키로 변환
    selected_frameworks = [model_options[name] for name in selected_display_names]
    
    # 모델 정보 표시
    st.header("모델 정보")
    model_info_df = pd.DataFrame({
        "Framework": [f for f in selected_frameworks],
        "Model": [model_info.get(f, {}).get("model", "unknown") for f in selected_frameworks],
        "Host": [model_info.get(f, {}).get("host", "unknown") for f in selected_frameworks]
    })
    
    # Arrow 변환 오류 방지를 위한 전처리
    model_info_df = preprocess_dataframe(model_info_df)
    st.dataframe(model_info_df)
    
    if not selected_frameworks:
        st.warning("비교할 모델을 선택하세요.")
        return
    
    # 소스 데이터 로드
    st.header("예측 결과 비교")
    
    # 소스 데이터 경로 선택
    source_paths = {model_info.get(f, {}).get("source_data", "") for f in selected_frameworks}
    source_paths = [p for p in source_paths if p]
    
    if not source_paths:
        st.error("소스 데이터 경로를 찾을 수 없습니다.")
        return
    
    source_path = st.selectbox("소스 데이터 선택", options=source_paths)
    
    try:
        source_data = pd.read_pickle(source_path)
        ground_truths = source_data["labels"].tolist()
        texts = source_data["text"].tolist()
    except Exception as e:
        st.error(f"소스 데이터 로드 중 오류 발생: {str(e)}")
        return
    
    # 샘플 선택
    sample_idx = st.slider("샘플 인덱스 선택", 0, len(texts), 0)
    
    # 현재 샘플의 Ground Truth와 예측 결과 가져오기
    ground_truth = ground_truths[sample_idx]
    
    # 각 모델의 예측 결과를 저장할 딕셔너리
    model_predictions = {}
    
    for framework in selected_frameworks:
        try:
            predictions = results[framework]["predictions"][sample_idx]
            # 여러 실행 결과가 있는 경우 첫 번째 결과만 사용
            if isinstance(predictions, list) and len(predictions) > 0:
                model_predictions[framework] = predictions[0]
            else:
                model_predictions[framework] = predictions
        except (IndexError, KeyError) as e:
            model_predictions[framework] = {}
    
    # Ground Truth와 모든 모델의 예측 결과에서 모든 카테고리(키) 수집
    all_categories = set()
    all_categories.update(ground_truth.keys())
    for framework, pred in model_predictions.items():
        # 예측 결과가 딕셔너리인 경우에만 keys() 호출
        if isinstance(pred, dict):
            all_categories.update(pred.keys())
        elif isinstance(pred, list):
            # 리스트인 경우 각 항목이 딕셔너리인지 확인
            st.warning(f"{framework}의 예측 결과가 리스트 형식입니다. 첫 번째 항목만 사용합니다.")
            if pred and isinstance(pred[0], dict):
                all_categories.update(pred[0].keys())
        else:
            st.warning(f"{framework}의 예측 결과 타입({type(pred)})이 예상과 다릅니다.")
    
    # 카테고리별 결과 표시
    st.subheader("카테고리별 라벨 비교")
    
    # 모든 카테고리에 대해 표 생성
    for category in sorted(all_categories):
        st.write(f"**카테고리: {category}**")
        
        # Ground Truth의 해당 카테고리 엔티티
        gt_entities = set(ground_truth.get(category, []))
        
        # 모든 모델과 Ground Truth를 포함한 표 데이터
        comparison_data = []
        
        # Ground Truth 데이터 추가
        comparison_data.append({
            "모델": "Ground Truth",
            "엔티티": ", ".join(gt_entities) if gt_entities else "(없음)"
        })
        
        # 각 모델의 예측 결과 추가
        for framework, prediction in model_predictions.items():
            # 예측이 딕셔너리인지 확인
            if not isinstance(prediction, dict):
                comparison_data.append({
                    "모델": framework,
                    "엔티티": f"(잘못된 형식: {type(prediction)})",
                    "정확히 맞춘 것(TP)": "(없음)",
                    "잘못 예측(FP)": "(없음)",
                    "놓친 것(FN)": ", ".join(gt_entities) if gt_entities else "(없음)"
                })
                continue
                
            pred_entities = set(prediction.get(category, []))
            
            # TP, FP, FN 계산 (metrics.py와 동일한 방식)
            true_positives = pred_entities.intersection(gt_entities)
            false_positives = pred_entities - gt_entities
            false_negatives = gt_entities - pred_entities
            
            # 데이터 추가
            comparison_data.append({
                "모델": framework,
                "엔티티": ", ".join(pred_entities) if pred_entities else "(없음)",
                "정확히 맞춘 것(TP)": ", ".join(true_positives) if true_positives else "(없음)",
                "잘못 예측(FP)": ", ".join(false_positives) if false_positives else "(없음)",
                "놓친 것(FN)": ", ".join(false_negatives) if false_negatives else "(없음)"
            })
        
        # 데이터프레임 생성
        comparison_df = pd.DataFrame(comparison_data)
        
        # Arrow 변환 오류 방지를 위한 전처리
        comparison_df = preprocess_dataframe(comparison_df)
        
        # 표 표시
        st.dataframe(comparison_df, use_container_width=True)
    
    # 모델별 전체 예측 결과 표시 (접을 수 있는 영역)
    with st.expander("전체 예측 결과 보기"):
        for framework in selected_frameworks:
            st.write(f"**{framework}**")
            try:
                predictions = results[framework]["predictions"][sample_idx]
                
                # 여러 실행 결과가 있는 경우
                if isinstance(predictions, list) and len(predictions) > 0:
                    for i, pred in enumerate(predictions):
                        st.write(f"Run {i+1}:")
                        st.json(pred)
                else:
                    st.json(predictions)
            except (IndexError, KeyError) as e:
                st.write(f"해당 샘플에 대한 예측 결과 없음 ({str(e)})")

    # 메트릭 비교
    st.header("성능 메트릭")
    metrics_data = {}
    
    for framework in selected_frameworks:
        try:
            latencies = results[framework]["latencies"][sample_idx]
            success_rate = results[framework]["percent_successful"][sample_idx]
            
            metrics_data[framework] = {
                "평균 지연시간(초)": sum(latencies) / len(latencies) if latencies else 0,
                "성공률(%)": success_rate * 100 if isinstance(success_rate, (int, float)) else 0
            }
        except (IndexError, KeyError, ZeroDivisionError) as e:
            metrics_data[framework] = {
                "평균 지연시간(초)": 0,
                "성공률(%)": 0
            }
            st.warning(f"{framework}의 메트릭 계산 중 오류: {str(e)}")
    
    metrics_df = pd.DataFrame(metrics_data).T
    
    # Arrow 변환 오류 방지를 위한 전처리
    metrics_df = preprocess_dataframe(metrics_df)
    st.dataframe(metrics_df)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
        logger.exception("Streamlit 앱 실행 오류")
