from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from retrieval_pipeline.api.security import verify_password, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from retrieval_pipeline.api.database import get_user, MOCK_USERS
from retrieval_pipeline.api.schemas import Token

router = APIRouter(tags=["auth"])

@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = get_user(form_data.username)
    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not verify_password(form_data.password, user_dict["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Embed role and department in the token
    access_token = create_access_token(
        data={"sub": user_dict["username"], "role": user_dict["role"], "department": user_dict["department"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ---------------------------------------------------------------------------
# Prototype role switching.
#
# The workbench has no login screen: picking a role signs you in as that role
# so RBAC can be demonstrated in one click. Previously the shared demo
# password was embedded in the frontend bundle. Issuing the token here keeps
# credentials out of the client entirely.
#
# Gated behind PROTOTYPE_AUTH so it cannot be left enabled by accident: with
# the flag off this 404s and the normal password flow is the only way in.
# ---------------------------------------------------------------------------
import os

from pydantic import BaseModel

PROTOTYPE_AUTH = os.getenv("PROTOTYPE_AUTH", "1").lower() in ("1", "true", "yes")


class DemoLogin(BaseModel):
    role: str


@router.post("/login/demo", response_model=Token)
async def login_as_role(body: DemoLogin):
    if not PROTOTYPE_AUTH:
        raise HTTPException(status_code=404, detail="Not found")

    user = next((u for u in MOCK_USERS.values() if u["role"] == body.role), None)
    if not user:
        raise HTTPException(status_code=404, detail=f"Unknown role: {body.role}")

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"],
              "department": user["department"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": token, "token_type": "bearer"}
