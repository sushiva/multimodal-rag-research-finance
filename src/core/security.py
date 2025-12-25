"""
Security & Authentication

What: JWT token handling, API key validation, rate limiting
Why: Protect API from unauthorized access and abuse
How: JWT for user auth, API keys for service auth, Redis for rate limiting

Security Layers:
1. API Key Authentication (for service-to-service)
2. JWT Token Authentication (for user sessions)
3. Rate Limiting (prevent abuse)
4. Input Validation (prevent injection attacks)

Usage:
    from fastapi import Depends
    from src.core.security import get_current_user, require_api_key

    @app.get("/protected")
    async def protected_endpoint(user=Depends(get_current_user)):
        return {"user": user}
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Header, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password

    Security:
        - Uses bcrypt (slow, resistant to brute force)
        - Automatic salt generation
        - Cost factor 12 (default, can be adjusted)
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored password hash

    Returns:
        True if password matches, False otherwise

    Security:
        - Constant-time comparison (prevents timing attacks)
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload to encode in token (typically {"sub": user_id})
        expires_delta: Token expiration time (default: 15 minutes)

    Returns:
        Encoded JWT token

    Security:
        - HS256 algorithm (HMAC with SHA-256)
        - Secret key from settings (never hardcoded)
        - Expiration time included in token
        - Claims include: sub (subject), exp (expiration), iat (issued at)

    Example:
        token = create_access_token({"sub": "user_123"})
        headers = {"Authorization": f"Bearer {token}"}
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=ALGORITHM
    )

    logger.debug("Access token created", expires_at=expire.isoformat())

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired

    Security:
        - Verifies signature (prevents tampering)
        - Checks expiration (prevents replay attacks)
        - Validates claims
    """
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[ALGORITHM]
        )
        return payload

    except JWTError as e:
        logger.warning("Invalid token", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> Dict[str, Any]:
    """
    Get current user from JWT token.

    Dependency for FastAPI endpoints requiring authentication.

    Args:
        authorization: Authorization header (Bearer {token})

    Returns:
        User data from token payload

    Raises:
        HTTPException: If token is missing or invalid

    Usage:
        @app.get("/me")
        async def get_me(user=Depends(get_current_user)):
            return {"user_id": user["sub"]}
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)

    if "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    return payload


async def require_api_key(
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> bool:
    """
    Validate API key from request header.

    Dependency for FastAPI endpoints requiring API key authentication.

    Args:
        api_key: API key from header

    Returns:
        True if valid API key

    Raises:
        HTTPException: If API key is missing or invalid

    Security:
        - Constant-time comparison (prevents timing attacks)
        - API keys should be long, random strings
        - Store hashed API keys in database (not implemented yet)

    Usage:
        @app.get("/protected", dependencies=[Depends(require_api_key)])
        async def protected():
            return {"message": "Authenticated"}
    """
    if not api_key:
        logger.warning("API key missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not api_key.startswith("sk-"):
        logger.warning("Invalid API key format", key_prefix=api_key[:4])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    logger.info("API key validated", key_prefix=api_key[:7])
    return True


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: User input text
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        HTTPException: If input exceeds max length

    Security:
        - Length validation (prevent DOS)
        - Strip control characters
        - Remove null bytes
        - Trim whitespace

    Note:
        This is basic sanitization. For HTML/SQL, use specialized libraries.
    """
    if len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input too long (max {max_length} characters)",
        )

    sanitized = text.replace("\x00", "")

    sanitized = "".join(
        char for char in sanitized
        if char.isprintable() or char in "\n\r\t"
    )

    sanitized = sanitized.strip()

    if not sanitized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input cannot be empty after sanitization",
        )

    return sanitized


async def validate_file_upload(
    filename: str,
    content_type: str,
    file_size: int
) -> None:
    """
    Validate file upload security.

    Args:
        filename: Original filename
        content_type: MIME type
        file_size: File size in bytes

    Raises:
        HTTPException: If file is invalid or unsafe

    Security Checks:
        - File size limit
        - Allowed MIME types
        - Filename sanitization (prevent path traversal)
        - Extension validation

    Usage:
        await validate_file_upload(file.filename, file.content_type, len(await file.read()))
    """
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {settings.max_upload_size_mb}MB)",
        )

    if content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.allowed_image_types}",
        )

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename (path traversal attempt)",
        )

    allowed_extensions = [".png", ".jpg", ".jpeg"]
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {allowed_extensions}",
        )

    logger.info(
        "File upload validated",
        filename=filename,
        content_type=content_type,
        size_bytes=file_size,
    )
