import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import metrics
from loguru import logger

# 페이지 설정
st.set_page_config(
    page_title="LLM NER 벤치마크 시각화 도구",
    page_icon="📊",
    layout="wide",
)

# 타이틀 및 설명
st.title("LLM NER 벤치마크 결과 시각화")
st.markdown("결과 폴더를 선택하고 분석할 모델을 선택하세요.")

# 결과 폴더 탐색 함수
def get_result_folders():
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.exists(results_path):
        return []
    
    folders = [f for f in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, f))]
    return folders

# PKL 파일 탐색 함수
def get_pkl_files(folder):
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", folder)
    if not os.path.exists(folder_path):
        return []
    
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    return pkl_files

# PKL 파일 로드 함수
def load_pkl_file(folder, filename):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", folder, filename)
    
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {file_path}: {e}")
        return None

# 소스 데이터 로드 함수
def load_source_data(path):
    try:
        return pd.read_pickle(path)
    except Exception as e:
        st.error(f"소스 데이터 로드 중 오류 발생: {path}: {e}")
        return None

# 사이드바에 폴더 선택 드롭다운 추가
result_folders = get_result_folders()
if not result_folders:
    st.error("results 폴더가 없거나 하위 폴더가 없습니다.")
    st.stop()

selected_folder = st.sidebar.selectbox(
    "결과 폴더 선택:",
    result_folders,
    index=0
)

# 선택된 폴더의 PKL 파일들을 체크박스로 표시
pkl_files = get_pkl_files(selected_folder)
if not pkl_files:
    st.error(f"선택한 폴더 ({selected_folder})에 PKL 파일이 없습니다.")
    st.stop()

st.sidebar.markdown("## 분석할 모델 선택")
selected_files = []
for pkl_file in pkl_files:
    if st.sidebar.checkbox(pkl_file, key=f"checkbox_{pkl_file}"):
        selected_files.append(pkl_file)

# 정렬 옵션
sort_by = st.sidebar.selectbox(
    "정렬 기준:",
    ["f1", "precision", "recall", "reliability", "latency"],
    index=0
)

# 계산 버튼
calculate_button = st.sidebar.button("메트릭 계산 및 시각화")

if calculate_button and selected_files:
    st.markdown("## 선택된 모델들의 성능 비교")
    
    # 결과 로드
    results = {}
    source_data_paths = {}
    
    for filename in selected_files:
        data = load_pkl_file(selected_folder, filename)
        if data:
            results.update(data)
            
            # 소스 데이터 경로 추출
            for key, value in data.items():
                source_data_path = value.get("source_data_path", "")
                if source_data_path:
                    source_data_paths[key] = source_data_path
    
    # 소스 데이터 경로 검증 및 선택
    if not source_data_paths:
        st.error("선택한 모델 결과에 소스 데이터 경로가 포함되어 있지 않습니다.")
        st.stop()
    
    # 다른 소스 데이터 경로가 있는 경우 사용자에게 알림
    unique_paths = set(source_data_paths.values())
    if len(unique_paths) > 1:
        st.warning(f"서로 다른 소스 데이터 경로가 발견되었습니다:")
        for framework, path in source_data_paths.items():
            st.write(f" - {framework}: {path}")
        
        # 가장 많이 사용된 경로를 선택
        path_counts = {}
        for path in source_data_paths.values():
            path_counts[path] = path_counts.get(path, 0) + 1
        
        source_path = max(path_counts.items(), key=lambda x: x[1])[0]
        st.info(f"가장 많이 사용된 소스 데이터 경로를 선택합니다: {source_path}")
    else:
        # 모든 경로가 동일한 경우
        source_path = next(iter(unique_paths))
    
    # 소스 데이터 로드
    source_data = load_source_data(source_path)
    if source_data is None:
        st.error("소스 데이터를 로드할 수 없습니다.")
        st.stop()
    
    ground_truths = source_data["labels"].tolist()
    
    # 메트릭 계산
    try:
        combined_df = metrics.combined_metrics(results, ground_truths=ground_truths, sort_by=sort_by)
        
        # 결과 표시
        st.dataframe(combined_df)
        
        # 시각화 - 바 차트
        st.markdown("### 메트릭 비교")
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        
        # F1 점수
        sns.barplot(x=combined_df.index, y=combined_df['micro_f1'], ax=axes[0])
        axes[0].set_title('F1 Score')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # Precision
        sns.barplot(x=combined_df.index, y=combined_df['micro_precision'], ax=axes[1])
        axes[1].set_title('Precision')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        
        # Recall
        sns.barplot(x=combined_df.index, y=combined_df['micro_recall'], ax=axes[2])
        axes[2].set_title('Recall')
        axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
        
        # Latency
        sns.barplot(x=combined_df.index, y=combined_df['Latency'], ax=axes[3])
        axes[3].set_title('Latency (seconds)')
        axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 예측 결과와 실제 값 비교
        st.markdown("### 예측 결과와 실제 값 비교")
        
        # 텍스트 샘플 선택
        max_index = len(ground_truths) - 1
        sample_index = st.slider("텍스트 샘플 선택", 0, max_index + 1, 0)
        
        # 결과 비교 테이블 생성 - 스키마 기반으로 행렬 형태로 표시
        # Ground Truth 라벨 가져오기
        ground_truth_labels = ground_truths[sample_index]
        if isinstance(ground_truth_labels, set):
            ground_truth_labels = list(ground_truth_labels)
        
        # 모든 가능한 키(스키마) 수집
        all_keys = set()
        # Ground Truth 키 추가
        for label in ground_truth_labels:
            if isinstance(label, dict):
                all_keys.update(label.keys())
            elif isinstance(label, str):
                all_keys.add(label)
                
        # 모든 모델 예측에서 키 추가
        for framework, data in results.items():
            try:
                predictions = data['predictions'][sample_index][0]
                if predictions:
                    for pred in predictions:
                        if isinstance(pred, dict):
                            all_keys.update(pred.keys())
                        elif isinstance(pred, str):
                            all_keys.add(pred)
            except (IndexError, KeyError):
                pass
        
        # 결과 비교 테이블 생성
        comparison_data = []
        for key in sorted(all_keys):
            row = {'Schema': key, 'Ground Truth': ''}
  
            row['Ground Truth'] = ground_truths[sample_index][key]  # 딕셔너리에서 키에 해당하는 값
            
            # 각 모델의 예측 값 추가
            for framework, data in results.items():
                try:
                    predictions = data['predictions'][sample_index][0]
                    row[framework] = predictions[key]  # 딕셔너리에서 실제 값 표시
                except (IndexError, KeyError):
                    row[framework] = 'N/A'
            comparison_data.append(row)
        # 결과 테이블 표시
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
        else:
            st.warning("이 샘플에 대한 예측 결과가 없습니다.")
        
        # 이전/다음 샘플 이동 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("이전 샘플") and sample_index > 0:
                st.experimental_rerun()
        with col2:
            if st.button("다음 샘플") and sample_index < len(ground_truths) - 1:
                st.experimental_rerun()
                
    except Exception as e:
        st.error(f"메트릭 계산 중 오류가 발생했습니다: {e}")
else:
    if calculate_button and not selected_files:
        st.warning("분석할 모델을 하나 이상 선택하세요.")
    else:
        st.info("메트릭을 계산하고 시각화하려면 왼쪽 사이드바에서 모델을 선택하고 '메트릭 계산 및 시각화' 버튼을 클릭하세요.")
