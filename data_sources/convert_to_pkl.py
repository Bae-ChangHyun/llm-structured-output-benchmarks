from cgitb import text
import os
import json
import pandas as pd
import glob
import importlib
from typing import Dict, List, Any
import inspect

# output_scheme.py에서 ResumeInfo 모델만 가져오기
from output_scheme import ResumeInfo, PersonalInfo, Education, Activity, Certificate

def convert_json_to_dataframe(result_dir):
    """
    지정된 result 디렉토리 내의 모든 candidate 폴더에서 
    temp2.json과 candidate.json 파일을 읽어 DataFrame으로 변환한 후 
    pickle 파일로 저장합니다.
    
    Args:
        result_dir (str): result 디렉토리의 경로
    """
    # 결과 저장을 위한 빈 리스트 생성
    data_list = []
    
    # 모든 candidate 폴더 탐색
    candidate_dirs = [os.path.join(result_dir,d) for d in os.listdir(result_dir) if d.startswith('candidate')]
    #print(candidate_dirs)
    
    for candidate_dir in candidate_dirs:
        candidate_name = os.path.basename(candidate_dir)
        
        # candidate_temp2.json 파일 경로  -> markdown output
        text_path = os.path.join(candidate_dir, f"{candidate_name}_temp2.json")
        print(text_path)
        # candidate.json 파일 경로 -> structured output
        labels_path = os.path.join(candidate_dir, f"{candidate_name}.json")
        print(labels_path)
        
        if os.path.exists(text_path) and os.path.exists(labels_path):
            try:
                # temp2.json 파일에서 텍스트 읽기
                with open(text_path, 'r', encoding='utf-8') as f:
                    text_content = json.load(f)
                
                # candidate.json 파일에서 레이블 정보 읽기
                with open(labels_path, 'r', encoding='utf-8') as f:
                    label_json = json.load(f)
                
                # 레이블 정보를 원하는 형식으로 변환
                labels = convert_to_label_format(label_json)
                
                # 데이터를 리스트에 추가
                data_list.append({
                    'text': text_content,
                    'labels': labels,
                    'candidate_name': candidate_name,
                    'original_path': candidate_dir
                })
                
                print(f"처리 완료: {candidate_name}")
            
            except Exception as e:
                print(f"오류 발생 ({candidate_name}): {e}")
    
    # DataFrame 생성
    df = pd.DataFrame(data_list)
    
    # pickle 파일로 저장
    output_path = os.path.join(result_dir, "resume_data.pkl")
    print(df)
    df.to_pickle(output_path)
    print(f"데이터가 성공적으로 저장되었습니다: {output_path}")
    
    return df


def convert_to_label_format(label_json: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    candidate.json 파일의 내용을 지정된 레이블 형식으로 변환합니다.
    대분류_소분류 형식의 키를 가진 사전으로 변환합니다.
    
    Args:
        label_json (dict): candidate.json 파일의 내용
        
    Returns:
        dict: 변환된 레이블 형식 {대분류_소분류: [값1, 값2, ...]}
    """
    labels = {}
    # JSON 파일의 모든 항목을 순회
    for category_name, category_data in label_json.items():
        # 객체 타입 처리 (personal_info와 같은 단일 객체)
        if isinstance(category_data, dict):
            for field_name, field_value in category_data.items():
                if field_value is not None and field_value != "":
                    # 대분류_소분류 형식의 키 생성
                    label_key = f"{category_name}_{field_name}"
                    # 값은 항상 리스트로 저장
                    labels[label_key] = [field_value]
        
        # 리스트 타입 처리 (education, activities, certificates와 같은 리스트)
        elif isinstance(category_data, list):
            for item in category_data:
                if isinstance(item, dict):
                    for field_name, field_value in item.items():
                        if field_value is not None and field_value != "":
                            # 대분류_소분류 형식의 키 생성
                            label_key = f"{category_name}_{field_name}"
                            
                            # 이미 키가 존재하면 값을 추가, 아니면 새 리스트 생성
                            if label_key in labels:
                                labels[label_key].append(field_value)
                            else:
                                labels[label_key] = [field_value]
    print(labels)
    return labels

def main():
    """
    메인 함수: 결과 디렉토리를 지정하고 변환 함수를 실행합니다.
    """
    # 현재 스크립트가 있는 경로에서 result 디렉토리 경로 구성
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, "result")
    
    # 사용자에게 특정 결과 폴더를 선택하도록 안내
    result_folders = [f for f in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, f))]
    
    if not result_folders:
        print("결과 폴더가 없습니다.")
        return
    
    print("사용 가능한 결과 폴더:")
    for i, folder in enumerate(result_folders):
        print(f"{i+1}. {folder}")
    
    choice = input("처리할 폴더 번호를 선택하세요 (모든 폴더를 처리하려면 'all' 입력): ")
    
    if choice.lower() == 'all':
        for folder in result_folders:
            folder_path = os.path.join(result_dir, folder)
            print(f"\n'{folder}' 폴더 처리 중...")
            convert_json_to_dataframe(folder_path)
    else:
        try:
            folder_idx = int(choice) - 1
            if 0 <= folder_idx < len(result_folders):
                folder_path = os.path.join(result_dir, result_folders[folder_idx])
                print(f"\n'{result_folders[folder_idx]}' 폴더 처리 중...")
                convert_json_to_dataframe(folder_path)
            else:
                print("잘못된 폴더 번호입니다.")
        except ValueError:
            print("유효한 숫자를 입력하거나 'all'을 입력하세요.")

if __name__ == "__main__":
    main()