"""Authentication routes — simple JWT-based auth."""

import os
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel

router = APIRouter()

SECRET_KEY = os.environ.get("PHARMAAI_SECRET_KEY", "pharmaai-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

# Hardcoded credentials for small-scale use (under 10 doctors)
# In production, move to a database or external auth provider
USERS = {
    "doctor@pharmaai.com": {"password": "pharmaai2025", "name": "Dr. Demo"},
    "admin@pharmaai.com": {"password": "admin2025", "name": "Admin"},
}

# Allow override via environment variable: "email:password:name,email2:password2:name2"
env_users = os.environ.get("PHARMAAI_USERS", "")
if env_users:
    for entry in env_users.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3:
            USERS[parts[0]] = {"password": parts[1], "name": parts[2]}


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str
    name: str
    email: str


class VerifyResponse(BaseModel):
    valid: bool
    email: str
    name: str


def create_token(email: str, name: str) -> str:
    payload = {
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = USERS.get(request.email)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(request.email, user["name"])
    return LoginResponse(token=token, name=user["name"], email=request.email)


@router.get("/verify", response_model=VerifyResponse)
async def verify(user: dict = Depends(verify_token)):
    return VerifyResponse(valid=True, email=user["email"], name=user["name"])
