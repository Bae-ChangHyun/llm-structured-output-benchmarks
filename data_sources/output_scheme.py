from pydantic import BaseModel, Field
from typing import List

class PersonalInfo(BaseModel):
    name: str = Field(description="지원자의 이름")
    email: str = Field(description="지원자의 이메일")
    phone: str = Field(description="지원자의 전화번호")
    address: str = Field(description="지원자의 주소")
    age: str = Field(description="지원자의 나이(년도로 되어있으면 현재 2025년기준으로 계산하여 작성할것)")

# 모델 정의
class Education(BaseModel):
    institution: str = Field(description="최종학력")
    degree: str = Field(description="학위")
    score: str = Field(description="학점 (예: 4.5/4.5)")
    major: str = Field(description="전공")
    period: str = Field(description="기간 (예: 2015-2019) 없으면 비워둘 것")

class Activity(BaseModel):
    title: str = Field(description="활동/경험/포트폴리오 제목")
    description: str = Field(description="활동/경험/포트폴리오/프로젝트경험 내용")
    period: str = Field(description="활동 기간")

class Certificate(BaseModel):
    name: str = Field(description="자격증 이름")
    score: str = Field(description="자격증 점수(없으면 Pass로 표기)")
    issuer: str = Field(description="발급 기관")
    date: str = Field(description="취득일")

class ResumeInfo(BaseModel):
    personal_info: PersonalInfo = Field(description="지원자의 개인 정보")
    education: List[Education] = Field(description="지원자의 학력 정보", default=[])
    activities: List[Activity] = Field(description="지원자의 경험/활동/프로젝트/포트폴리오 내역", default=[])
    certificates: List[Certificate] = Field(description="지원자의 자격증 정보", default=[])
