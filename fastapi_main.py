from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient
import os
from typing import Optional
from contextlib import asynccontextmanager

# MongoDB ObjectId를 문자열로 변환하는 함수
def serialize_doc(doc):
    if doc is None:
        return None
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        serialized = {}
        for key, value in doc.items():
            if key == "_id":
                continue  # _id 필드는 제외
            serialized[key] = serialize_doc(value)
        return serialized
    return doc

# MongoDB 설정 - 올바른 컬렉션 이름으로 수정
MONGODB_URL = "mongodb+srv://mingyu4796:qwert12345@cluster0.nnr0q.mongodb.net/"
DATABASE_NAME = "Code_Reading"
USERS_COLLECTION = "User_info"  # 수정: I를 소문자로 변경
SUBMISSIONS_COLLECTION = "Submission"
PROBLEMS_COLLECTION = "Question_info"

# JWT 설정
SECRET_KEY = "Computer_Science_Education_2025_Secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# MongoDB 클라이언트
client = AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]

# 패스워드 해싱
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# lifespan 이벤트 핸들러 - 간단하게 정리
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 실행
    print("FastAPI 서버 시작됨")
    try:
        # MongoDB 연결 테스트
        await client.admin.command('ping')
        print("MongoDB 연결 성공")
        
        # 사용자 컬렉션 확인
        user_count = await db[USERS_COLLECTION].count_documents({})
        print(f"User_info 컬렉션의 문서 개수: {user_count}")
        
        if user_count > 0:
            print("사용자 데이터베이스 정상 연결됨")
        else:
            print("경고: 사용자 컬렉션이 비어있습니다")
        
    except Exception as e:
        print(f"MongoDB 연결 오류: {e}")
    
    yield
    # 종료 시 실행
    print("FastAPI 서버 종료됨")

# FastAPI 앱 생성
app = FastAPI(title="학생 답안 제출 시스템", lifespan=lifespan)

# 정적 파일 제공
app.mount("/static", StaticFiles(directory="static"), name="static")

# 보안 설정
security = HTTPBearer()

# Pydantic 모델들
class LoginRequest(BaseModel):
    user_id: str
    password: str

class SubmissionRequest(BaseModel):
    problem_id: str
    answer: str

class User(BaseModel):
    user_id: str
    name: str
    hashed_password: str

class Problem(BaseModel):
    problem_id: str
    title: str
    content: str
    max_score: int

class Submission(BaseModel):
    user_id: str
    problem_id: str
    answer: str
    submitted_at: datetime

# JWT 토큰 생성
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# JWT 토큰 검증 수정 - 실제 DB 구조에 맞춤
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="로그인이 필요합니다",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    
    # user_id를 정수로 변환해서 검색
    try:
        user_id_int = int(user_id)
    except ValueError:
        raise credentials_exception
    
    user = await db[USERS_COLLECTION].find_one({"ID": user_id_int})
    if user is None:
        raise credentials_exception
    return user

# 디버깅용 API 수정
@app.get("/api/debug/user/{user_id}")
async def debug_get_user(user_id: int):
    """디버깅용: 특정 사용자 조회"""
    user = await db[USERS_COLLECTION].find_one({"ID": user_id})
    return serialize_doc(user) if user else {"error": "User not found"}

@app.get("/api/debug/users")
async def debug_get_all_users():
    """디버깅용: 모든 사용자 조회 (최대 10명)"""
    users = await db[USERS_COLLECTION].find({}).limit(10).to_list(length=10)
    return serialize_doc(users)

@app.get("/api/debug/questions")
async def debug_get_questions():
    """디버깅용: 모든 문제 조회"""
    questions = await db[PROBLEMS_COLLECTION].find({}).to_list(length=10)
    return serialize_doc(questions)

# 라우트 정의
@app.get("/")
async def root():
    return FileResponse("static/login.html")

# 튜토리얼 페이지 라우트 추가
@app.get("/tutorial1")
async def tutorial1_page():
    return FileResponse("static/Tutorial1.html")

@app.get("/tutorial2")
async def tutorial2_page():
    return FileResponse("static/Tutorial2.html")

# 로그인 함수 수정 - 디버깅 정보 추가
@app.post("/api/login")
async def login(login_data: LoginRequest):
    print(f"로그인 시도: {login_data.user_id}, {login_data.password}")
    
    # ID는 정수로 변환해서 검색
    try:
        user_id_int = int(login_data.user_id)
        print(f"변환된 ID: {user_id_int}")
    except ValueError:
        print("ID 변환 실패")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 학번입니다"
        )
    
    user = await db[USERS_COLLECTION].find_one({"ID": user_id_int})
    print(f"DB에서 찾은 사용자: {user}")
    
    if not user:
        print("사용자를 찾을 수 없음")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="존재하지 않는 학번입니다"
        )
    
    # 비밀번호 확인 (여러 가지 방법으로 시도)
    password_int = int(login_data.password)
    print(f"입력된 비밀번호: {password_int}, DB PW: {user.get('PW')}")
    
    # 여러 조건으로 비밀번호 확인
    if (user.get("PW") == password_int or 
        user.get("PW") == str(password_int) or 
        user.get("PW") == login_data.password):
        print("비밀번호 일치!")
    else:
        print(f"비밀번호 불일치: DB={user.get('PW')}, 입력={password_int}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="비밀번호가 잘못되었습니다"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["ID"])}, expires_delta=access_token_expires
    )
    
    # 사용자 이름을 Hash_ID로 설정
    user_name = f"학생{user.get('Hash_ID', user['ID'])}"
    print(f"로그인 성공: {user_name}")
    
    return {"access_token": access_token, "token_type": "bearer", "user_name": user_name}

# 문제 목록 조회 수정 - 코드 표시 방식 개선
@app.get("/api/problems")
async def get_problems(current_user: dict = Depends(get_current_user)):
    # Question_id가 있는 기존 문제들을 조회
    problems = await db[PROBLEMS_COLLECTION].find({
        "Question_id": {"$exists": True}
    }).sort("Question_id", 1).to_list(length=None)
    
    # 프론트엔드에서 사용할 수 있도록 필드명을 변환
    converted_problems = []
    for problem in problems:
        # Code 필드를 그대로 사용 (``` 없이)
        converted_problem = {
            "problem_id": f"q{problem['Question_id']}",  # 1, 2, 3... -> q1, q2, q3...
            "title": f"문제 {problem['Question_id']}",
            "content": f"{problem['Question Description']}\n\n{problem['Code']}",  # 코드 블록 표시 제거
            "max_score": 100,  # 기본값
            "original_id": problem['Question_id']
        }
        converted_problems.append(converted_problem)
    
    return serialize_doc(converted_problems)

# 개별 문제 조회 수정 - 코드 표시 방식 개선
@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str, current_user: dict = Depends(get_current_user)):
    # q1, q2 형태에서 숫자만 추출
    if problem_id.startswith('q'):
        try:
            question_id = int(problem_id[1:])  # q1 -> 1, q2 -> 2
        except ValueError:
            raise HTTPException(status_code=404, detail="잘못된 문제 ID입니다")
    else:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")
    
    # Question_id로 문제 검색
    problem = await db[PROBLEMS_COLLECTION].find_one({"Question_id": question_id})
    if not problem:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")
    
    # 프론트엔드 형식으로 변환 - Code를 더 명확하게 분리
    converted_problem = {
        "problem_id": problem_id,
        "title": f"문제 {problem['Question_id']}",
        "description": problem['Question Description'],  # 설명만 따로
        "code": problem['Code'],  # 코드만 따로
        "content": f"{problem['Question Description']}\n\n{problem['Code']}",  # 전체 (호환성)
        "max_score": 100,
        "original_id": problem['Question_id']
    }
    
    return serialize_doc(converted_problem)

# 답안 제출 수정 - Hash_ID 필드 추가
@app.post("/api/submit")
async def submit_answer(submission: SubmissionRequest, current_user: dict = Depends(get_current_user)):
    # problem_id가 q1, q2 형태인지 확인하고 Question_id로 변환
    if submission.problem_id.startswith('q'):
        try:
            question_id = int(submission.problem_id[1:])  # q1 -> 1
        except ValueError:
            raise HTTPException(status_code=404, detail="잘못된 문제 ID입니다")
    else:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")
    
    # 문제 존재 확인
    problem = await db[PROBLEMS_COLLECTION].find_one({"Question_id": question_id})
    if not problem:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다")
    
    # 제출 데이터 생성 - Hash_ID 필드 추가
    submission_data = {
        "user_id": current_user["ID"],              # 학번 (예: 250100)
        "Hash_ID": current_user["Hash_ID"],         # Hash_ID (예: 327)
        "problem_id": submission.problem_id,        # q1, q2 형태 그대로 저장
        "question_id": question_id,                 # 원본 Question_id도 저장
        "answer": submission.answer,                # 학생이 작성한 답안
        "submitted_at": datetime.utcnow()           # 제출 시간
    }
    
    # 기존 제출이 있는 경우 업데이트, 없는 경우 새로 생성
    await db[SUBMISSIONS_COLLECTION].replace_one(
        {"user_id": current_user["ID"], "problem_id": submission.problem_id},
        submission_data,
        upsert=True
    )
    
    return {"message": "답안이 성공적으로 제출되었습니다"}

# 내 제출 내역 조회 수정 - user_id 필드명 변경
@app.get("/api/my-submissions")
async def get_my_submissions(current_user: dict = Depends(get_current_user)):
    submissions = await db[SUBMISSIONS_COLLECTION].find(
        {"user_id": current_user["ID"]}  # ID 필드 사용
    ).to_list(length=None)
    return serialize_doc(submissions)

@app.get("/problems")
async def problems_page():
    return FileResponse("static/problems.html")

@app.get("/problem/{problem_id}")
async def problem_page(problem_id: str):
    return FileResponse("static/problem.html")

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
