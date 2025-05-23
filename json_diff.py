import os
import json
import pickle
import glob
import random
import streamlit as st
import pandas as pd
import numpy as np
from loguru import logger
from jycm.jycm import YouchamaJsonDiffer
from jycm.helper import dump_html_output, open_url

results_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")

# 페이지 설정
st.set_page_config(
    page_title="LLM NER 벤치마크 시각화", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'analysis_url' not in st.session_state:
    st.session_state.analysis_url = None

st.title("LLM NER 벤치마크 결과 시각화")

# 결과 로딩 함수
def load_results():
    results_dir = "results"
    all_results = {}
    framework_model_info = {}
    folder_structure = {}
    
    # 결과 디렉토리의 모든 하위 폴더 검색
    result_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
    
    # 결과 디렉토리가 없으면 results 디렉토리 자체 사용
    if not result_dirs:
        result_dirs = [results_dir]
    
    for result_dir in result_dirs:
        # 각 결과 디렉토리에서 PKL 파일 검색
        pkl_files = glob.glob(os.path.join(result_dir, "*.pkl"))
        dir_name = os.path.basename(result_dir)
        
        folder_structure[dir_name] = {}
        
        for file_path in pkl_files:
            try:
                with open(file_path, "rb") as file:
                    framework_results = pickle.load(file)
                    
                    # 파일별로 모델 정보 구성
                    file_name = os.path.basename(file_path)
                    folder_structure[dir_name][file_name] = []
                    
                    # 모델 정보 추출 및 고유 키 생성
                    for key, value in framework_results.items():
                        # 폴더명을 포함한 고유 키 생성
                        unique_key = f"{dir_name}/{key}"
                        
                        # 새로운 키로 결과 저장
                        all_results[unique_key] = value
                        
                        framework_model_info[unique_key] = {
                            "model": value.get("llm_model", "unknown"),
                            "host": value.get("llm_provider", "unknown"),
                            "source_data": value.get("source_data_path", ""),
                            "file_path": file_path,
                            "folder": dir_name,
                            "file_name": file_name,
                            "original_key": key  # 원본 키도 보존
                        }
                        # 폴더 구조에 모델 키 추가
                        folder_structure[dir_name][file_name].append(unique_key)
            except Exception as e:
                st.warning(f"파일 로드 중 오류 발생: {file_path}: {str(e)}")
    
    return all_results, framework_model_info, folder_structure

# 메인 함수
def main():
    
    with st.spinner("로딩 중..."):
        results, model_info, folder_structure = load_results()
    
    if not results:
        st.error("처리할 결과 파일이 없습니다. results 폴더에 결과가 있는지 확인하세요.")
        return
    
    st.sidebar.header("결과 선택")
    
    folder_options = list(folder_structure.keys())
    if not folder_options:
        st.error("결과 폴더를 찾을 수 없습니다.")
        return
        
    selected_folder = st.sidebar.selectbox(
        "폴더 선택",
        options=folder_options,
        index=0
    )
    
    models_in_folder = {}
    for file_name, models in folder_structure[selected_folder].items():
        for model in models:
            original_key = model_info[model].get("original_key", "")
            display_name = f"{original_key}"
            models_in_folder[display_name] = model
    
    if not models_in_folder:
        st.error(f"선택한 폴더 '{selected_folder}'에 모델 결과가 없습니다.")
        return
    
    selected_model_display = st.sidebar.selectbox(
        "모델 선택",
        options=list(models_in_folder.keys()),
        index=0
    )

    selected_framework = models_in_folder[selected_model_display]
    
    with st.sidebar.expander("선택한 모델 정보", expanded=False):
        st.write(f"**모델명**: {model_info.get(selected_framework, {}).get('model', 'unknown')}")
        st.write(f"**제공자**: {model_info.get(selected_framework, {}).get('llm_provider', model_info.get(selected_framework, {}).get('host', ''))}")
        st.write(f"**파일**: {model_info.get(selected_framework, {}).get('file_name', 'unknown')}")
        st.write(f"**폴더**: {model_info.get(selected_framework, {}).get('folder', 'unknown')}")
    
    st.header("Ground Truth vs 예측 결과 비교")
    
    source_path = model_info.get(selected_framework, {}).get("source_data", "")
    
    if not source_path:
        st.error("소스 데이터 경로를 찾을 수 없습니다.")
        return
    
    try:
        source_data = pd.read_pickle(source_path)
        ground_truths = source_data["labels"].tolist()
        texts = source_data["text"].tolist()
    except Exception as e:
        st.error(f"소스 데이터 로드 중 오류 발생: {str(e)}")
        return
    
    # 샘플 선택
    sample_idx = st.slider("샘플 인덱스 선택", 0, len(texts) - 1, 0)
    
    # 현재 샘플의 Ground Truth 로드
    ground_truth_path = ground_truths[sample_idx]
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
        
    # 예측 결과 가져오기
    try:
        prediction = results[selected_framework]["predictions"][sample_idx]
        # 여러 실행 결과가 있는 경우 첫 번째 결과만 사용
        if isinstance(prediction, list) and len(prediction) > 0:
            # 첫 번째 실행 결과만 사용
            first_run = prediction[0]
            # 리스트 내의 첫 번째 항목이 딕셔너리이면 그대로 사용, 아니면 첫 번째 실행 결과 전체를 사용
            if isinstance(first_run, list) and len(first_run) > 0:
                prediction = first_run[0]
            else:
                prediction = first_run
    except (IndexError, KeyError) as e:
        st.error(f"예측 결과 로드 중 오류 발생: {str(e)}")
        prediction = {}
    
    with st.expander("원본 텍스트", expanded=False):
        st.text(texts[sample_idx])
    
    with st.expander("JSON 데이터 보기", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ground Truth (정답 데이터)")
            st.json(ground_truth)
            
        with col2:
            st.subheader(f"예측 결과 ({selected_framework})")
            st.json(prediction)
    
    analyze_button = st.button("JSON 구조 분석")
    visualize_button = st.button("분석 결과 보기")
    
    if analyze_button:
        import shutil
        shutil.rmtree(results_dir_path, ignore_errors=True)
        
        # jycm을 사용한 JSON 비교 분석
        try:
            ycm = YouchamaJsonDiffer(ground_truth, prediction)
            diff_result = ycm.get_diff()  # 새 API 사용
            
            os.makedirs(results_dir_path, exist_ok=True) 
            
            while True:
                random_id = random.randint(1, 100000)
                tmp_results_dir = os.path.join(results_dir_path, str(random_id))
                if not os.path.exists(tmp_results_dir):
                    break
            url = dump_html_output(ground_truth, prediction, diff_result, tmp_results_dir, left_title="Ground Truth", right_title=selected_model_display)

            st.session_state.analysis_url = url
            
            st.success("JSON 구조 분석이 완료되었습니다. '분석 결과 보기' 버튼을 클릭하여 결과를 확인하세요.")
            
        except Exception as e:
            st.error(f"JSON 구조 분석 중 오류 발생: {str(e)}")
            logger.exception("jycm 분석 오류")
            st.info("jycm 라이브러리 사용 중 문제가 발생했습니다. 라이브러리가 올바르게 설치되어 있는지 확인하세요.")
                
    if visualize_button:
        if st.session_state.analysis_url:
            try:
                open_url(st.session_state.analysis_url)
                st.success("분석 결과 페이지를 외부 브라우저에서 열었습니다.")
            except Exception as e:
                st.error(f"외부 브라우저를 여는 중 오류가 발생했습니다: {str(e)}")
                st.info(f"분석 결과 URL: {st.session_state.analysis_url}")
        else:
            st.warning("먼저 'JSON 구조 분석' 버튼을 클릭하여 분석을 실행해주세요.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
        logger.exception("Streamlit 앱 실행 오류")
