from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from retrieval_pipeline.api.security import SECRET_KEY, ALGORITHM
from retrieval_pipeline.api.schemas import TokenData, User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        department: str = payload.get("department")
        if username is None or role is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role, department=department)
    except JWTError:
        raise credentials_exception

    # Construct the user model
    user = User(username=token_data.username, role=token_data.role, department=token_data.department)
    return user
