from pydantic import BaseModel, Field
from typing import List, Optional

class PersonalInfo(BaseModel):
    name: Optional[str] = Field(default="", description="지원자의 이름")
    email: Optional[str] = Field(default="", description="지원자의 이메일")
    phone: Optional[str] = Field(default="", description="지원자의 전화번호")
    address: Optional[str] = Field(default="", description="지원자의 주소")
    age: Optional[str] = Field(default="", description="지원자의 나이(년도로 되어있으면 현재 2025년기준으로 계산하여 작성할것)")

# 모델 정의
class Education(BaseModel):
    institution: Optional[str] = Field(default="", description="최종학력")
    degree: Optional[str] = Field(default="", description="학위")
    score: Optional[str] = Field(default="", description="학점 (예: 4.5/4.5)")
    major: Optional[str] = Field(default="", description="전공")
    period: Optional[str] = Field(default="", description="기간 (예: 2015-2019) 없으면 비워둘 것")

class Activity(BaseModel):
    title: Optional[str] = Field(default="", description="활동/경험/포트폴리오 제목")
    description: Optional[str] = Field(default="", description="활동/경험/포트폴리오/프로젝트경험 내용")
    period: Optional[str] = Field(default="", description="활동 기간")

class Certificate(BaseModel):
    name: Optional[str] = Field(default="", description="자격증 이름")
    score: Optional[str] = Field(default="Pass", description="자격증 점수(없으면 Pass로 표기)")
    issuer: Optional[str] = Field(default="", description="발급 기관")
    date: Optional[str] = Field(default="", description="취득일")

class ResumeInfo(BaseModel):
    personal_info: PersonalInfo = Field(description="지원자의 개인 정보")
    education: List[Education] = Field(description="지원자의 학력 정보", default=[])
    activities: List[Activity] = Field(description="지원자의 경험/활동/프로젝트/포트폴리오 내역", default=[])
    certificates: List[Certificate] = Field(description="지원자의 자격증 정보", default=[])
