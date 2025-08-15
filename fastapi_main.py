# fastapi_main.py

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from contextlib import asynccontextmanager
from collections import defaultdict
import os

# MongoDB ObjectId를 문자열로 변환하는 함수
def serialize_doc(doc):
    if doc is None: return None
    if isinstance(doc, list): return [serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {k: serialize_doc(v) for k, v in doc.items() if k != "_id"}
    return doc

# 설정 - 환경변수 사용
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://mingyu4796:qwert12345@cluster0.nnr0q.mongodb.net/")
DATABASE_NAME = "Code_Reading"
USERS_COLLECTION = "User_info"
SUBMISSIONS_COLLECTION = "Submission"
PROBLEMS_COLLECTION = "Question_info"
SECRET_KEY = os.getenv("SECRET_KEY", "Computer_Science_Education_2025_Secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# 전역 변수로 클라이언트 관리
client = None
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 MongoDB 연결
    global client, db
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
        # 연결 테스트
        await client.admin.command('ping')
        print("✅ MongoDB 연결 성공")
    except Exception as e:
        print(f"❌ MongoDB 연결 실패: {e}")
        raise
    
    yield
    
    # 종료 시 연결 정리
    if client:
        client.close()
        print("📦 MongoDB 연결 종료")

# FastAPI 앱 생성 - lifespan 추가
app = FastAPI(
    title="학생 답안 제출 및 교사 관리 시스템",
    lifespan=lifespan
)
security = HTTPBearer()

# Pydantic 모델
class LoginRequest(BaseModel): 
    user_id: str
    password: str

class SubmissionRequest(BaseModel): 
    problem_id: str
    answer: str

# 헬퍼 함수
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def create_no_cache_file_response(file_path: str) -> FileResponse:
    return FileResponse(file_path, headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "Expires": "0"})

def parse_student_id(student_id: int) -> dict:
    s = str(student_id)
    return {"grade": int(s[:2]), "class": int(s[2:4]), "number": int(s[4:])} if len(s) == 6 else {}

# 코드 하이라이팅 처리 함수 추가
def process_code_highlighting(code: str, highlight_start: Optional[int], highlight_end: Optional[int]) -> str:
    """
    코드에서 지정된 라인 범위를 하이라이팅하여 HTML로 반환
    """
    if not code:
        return ""
    
    lines = code.split('\n')
    
    # 하이라이팅 범위가 지정되지 않은 경우 원본 반환
    if highlight_start is None or highlight_end is None:
        return '\n'.join(f'<span class="code-line">{line}</span>' for line in lines)
    
    # 1-based index를 0-based로 변환하고 범위 검증
    start_idx = max(0, highlight_start - 1)
    end_idx = min(len(lines) - 1, highlight_end - 1)
    
    result_lines = []
    for i, line in enumerate(lines):
        if start_idx <= i <= end_idx:
            # 하이라이팅 구간
            if i == start_idx and i == end_idx:
                # 단일 라인 하이라이팅
                result_lines.append(f'<div class="highlighted-section"><span class="code-line">{line}</span></div>')
            elif i == start_idx:
                # 하이라이팅 시작
                result_lines.append(f'<div class="highlighted-section"><span class="code-line">{line}</span>')
            elif i == end_idx:
                # 하이라이팅 끝
                result_lines.append(f'<span class="code-line">{line}</span></div>')
            else:
                # 하이라이팅 중간
                result_lines.append(f'<span class="code-line">{line}</span>')
        else:
            # 일반 라인
            result_lines.append(f'<span class="code-line">{line}</span>')
    
    return '\n'.join(result_lines)

# 인증 및 권한
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="로그인이 필요합니다", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None: raise exc
        
        user_query = {"ID": int(user_id)} if user_id.isdigit() else {"ID": user_id}
        user = await db[USERS_COLLECTION].find_one(user_query)
        if user is None: raise exc
        
        user["user_type"] = payload.get("user_type", "student")
        return user
    except (jwt.PyJWTError, ValueError):
        raise exc

async def get_current_teacher(user: dict = Depends(get_current_user)):
    if user.get("user_type") != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="교사 권한이 필요합니다.")
    return user

# --- API 엔드포인트들 먼저 정의 ---

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    try:
        await client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": "disconnected", "error": str(e)}

# 로그인 API
@app.post("/api/login")
async def login(req: LoginRequest):
    uid = req.user_id
    user_query = {"ID": int(uid)} if uid.isdigit() else {"ID": uid}
    user = await db[USERS_COLLECTION].find_one(user_query)

    if not user or str(user.get("PW")) != req.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="ID 또는 비밀번호가 잘못되었습니다.")

    is_teacher = user.get("IsTeacher") == 1
    user_type = "teacher" if is_teacher else "student"
    name = user.get("Name", f"사용자{user.get('Hash_ID', user['ID'])}")
    if is_teacher: name += " 교사"

    token = create_access_token(data={"sub": str(user["ID"]), "user_type": user_type}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer", "user_name": name, "user_type": user_type}

# 문제 목록 API
@app.get("/api/problems")
async def get_problems(user: dict = Depends(get_current_user)):
    problems = await db[PROBLEMS_COLLECTION].find({}).sort("Question_id", 1).to_list(None)
    return [{"problem_id": f"q{p['Question_id']}", "title": p.get('title', f"문제 {p['Question_id']}"), "max_score": p.get('max_score', 100)} for p in problems]

# 개별 문제 API - 수정된 버전
@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str, user: dict = Depends(get_current_user)):
    qid = int(problem_id[1:]) if problem_id.startswith('q') else None
    if not qid:
        raise HTTPException(status_code=400, detail="잘못된 문제 ID입니다.")
    
    problem = await db[PROBLEMS_COLLECTION].find_one({"Question_id": qid})
    if not problem:
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다.")
    
    # MongoDB 컬럼명에 맞게 수정
    raw_code = problem.get('Code', '')
    highlight_start = problem.get('Highlight_Start_Line')
    highlight_end = problem.get('Highlight_End_Line')
    
    # 코드 하이라이팅 처리
    processed_code = process_code_highlighting(raw_code, highlight_start, highlight_end)
    
    return {
        "problem_id": problem_id,
        "title": problem.get('title', f"문제 {qid}"),
        "description": problem.get('Question Description', ''),  # 실제 MongoDB 컬럼명 사용
        "code": processed_code,  # 하이라이팅 처리된 코드
        "raw_code": raw_code,  # 원본 코드도 함께 전송 (필요시 사용)
        "highlight_start": highlight_start,
        "highlight_end": highlight_end,
        "max_score": problem.get('max_score', 100)
    }

# 내 제출 내역 API
@app.get("/api/my-submissions")
async def get_my_submissions(user: dict = Depends(get_current_user)):
    submissions = await db[SUBMISSIONS_COLLECTION].find({"user_id": user["ID"]}).to_list(None)
    return [serialize_doc(sub) for sub in submissions]

# 답안 제출 API
@app.post("/api/submit")
async def submit_answer(submission: SubmissionRequest, user: dict = Depends(get_current_user)):
    qid = int(submission.problem_id[1:]) if submission.problem_id.startswith('q') else None
    if not qid or not await db[PROBLEMS_COLLECTION].find_one({"Question_id": qid}):
        raise HTTPException(status_code=404, detail="문제를 찾을 수 없습니다.")
    
    await db[SUBMISSIONS_COLLECTION].replace_one(
        {"user_id": user["ID"], "problem_id": submission.problem_id},
        {"user_id": user["ID"], "Hash_ID": user.get("Hash_ID"), "problem_id": submission.problem_id, 
         "question_id": qid, "answer": submission.answer, "submitted_at": datetime.utcnow()},
        upsert=True
    )
    return {"message": "답안이 성공적으로 제출되었습니다."}

# 교사용 API들
@app.get("/api/teacher/statistics")
async def get_teacher_statistics(_: dict = Depends(get_current_teacher)):
    students_count = await db[USERS_COLLECTION].count_documents({"IsTeacher": 0})
    submitted_students_count = len(await db[SUBMISSIONS_COLLECTION].distinct("user_id"))
    rate = (submitted_students_count / students_count * 100) if students_count > 0 else 0
    return {"total_students": students_count, "submitted_students": submitted_students_count, "submission_rate": round(rate, 2)}

@app.get("/api/teacher/classes")
async def get_teacher_classes(_: dict = Depends(get_current_teacher)):
    students = await db[USERS_COLLECTION].find({"IsTeacher": 0}).to_list(None)
    submissions = await db[SUBMISSIONS_COLLECTION].find({}).to_list(None)
    
    class_data = defaultdict(lambda: {"total": 0, "submitted_ids": set()})
    for s in students:
        pid = parse_student_id(s['ID'])
        if pid: class_data[f"{pid['grade']}-{pid['class']}"]["total"] += 1

    for sub in submissions:
        pid = parse_student_id(sub['user_id'])
        if pid: class_data[f"{pid['grade']}-{pid['class']}"]["submitted_ids"].add(sub['user_id'])

    return sorted([
        {"class_id": cid, "class_name": f"{cid.split('-')[0]}학년 {cid.split('-')[1]}반",
         "total_students": data['total'], "submitted_students": len(data['submitted_ids']),
         "submission_rate": round(len(data['submitted_ids']) / data['total'] * 100, 2) if data['total'] > 0 else 0}
        for cid, data in class_data.items()
    ], key=lambda x: x['class_id'])

@app.get("/api/teacher/class/{class_id}")
async def get_class_details(class_id: str, _: dict = Depends(get_current_teacher)):
    try:
        grade, class_num = map(int, class_id.split('-'))
        id_start, id_end = int(f"{grade:02d}{class_num:02d}00"), int(f"{grade:02d}{class_num:02d}99")
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 반 ID 형식입니다.")

    students = await db[USERS_COLLECTION].find({"IsTeacher": 0, "ID": {"$gt": id_start, "$lt": id_end}}).to_list(None)
    student_ids = [s['ID'] for s in students]
    
    submissions = await db[SUBMISSIONS_COLLECTION].find({"user_id": {"$in": student_ids}}).to_list(None)
    problems = await db[PROBLEMS_COLLECTION].find({}).sort("Question_id", 1).to_list(None)

    subs_by_student = defaultdict(dict)
    for sub in submissions: subs_by_student[sub['user_id']][sub['problem_id']] = sub

    student_details = [{
        "id": s['ID'], "name": s.get("Name", f"학생 {s.get('Hash_ID')}"), "number": parse_student_id(s['ID']).get('number'),
        "submissions": {f"q{p['Question_id']}": subs_by_student[s['ID']].get(f"q{p['Question_id']}", {}) for p in problems}
    } for s in students]

    return {
        "class_name": f"{grade}학년 {class_num}반",
        "students": sorted(student_details, key=lambda x: x['number']),
        "problems": [{"id": f"q{p['Question_id']}", "title": p.get('title', f"문제 {p['Question_id']}")} for p in problems]
    }

# --- 정적 파일 및 페이지 라우팅 (수정됨) ---

# 정적 파일 먼저 마운트
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# 특정 HTML 페이지 라우팅 (수정된 경로들)
@app.get("/")
async def root(): 
    return create_no_cache_file_response("static/login.html")

# 학생용 페이지들 (수정된 라우팅)
@app.get("/tutorial0")
async def tutorial0(): 
    return create_no_cache_file_response("static/Tutorial0.html")

@app.get("/tutorial1")
async def tutorial1(): 
    return create_no_cache_file_response("static/Tutorial1.html")

@app.get("/tutorial1.5")  # 수정: tutorial1_5 -> tutorial1.5
async def tutorial1_5(): 
    return create_no_cache_file_response("static/Tutorial1_5.html")

@app.get("/tutorial2")
async def tutorial2(): 
    return create_no_cache_file_response("static/Tutorial2.html")

@app.get("/problems")
async def problems(): 
    return create_no_cache_file_response("static/problems.html")

@app.get("/problem/{problem_id}")
async def problem_page(problem_id: str): 
    return create_no_cache_file_response("static/problem.html")

# 교사용 페이지들
@app.get("/teacher/dashboard")
async def teacher_dashboard(): 
    return create_no_cache_file_response("static/teacher_dashboard.html")

@app.get("/teacher/class/{class_id}")
async def teacher_class(class_id: str): 
    return create_no_cache_file_response("static/teacher_class.html")

# 서버 시작 부분 추가
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("fastapi_main:app", host="0.0.0.0", port=port, reload=False)
