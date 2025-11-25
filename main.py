"""
Multi-Tenant AI Assistant API

A production-ready FastAPI application providing document Q&A capabilities
with multi-tenant support and vector database integration.

Author: Your Name
Version: 1.0.0
"""

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Depends,
    status,
    BackgroundTasks,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

import database
import qa_manager

# ============================================================================
# Configuration
# ============================================================================

class Settings:
    """Application configuration settings."""

    APP_NAME: str = "AsistenteIA API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: set = {".pdf"}
    TEMP_UPLOAD_DIR: Path = Path("temp_uploads")

    # CORS settings
    CORS_ORIGINS: list = ["*"]  # Override in production with specific domains
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]

    # Security
    TRUSTED_HOSTS: list = ["*"]  # Override in production

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class CompanyBase(BaseModel):
    """Base model for company data."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Company name",
        examples=["Acme Corporation"],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate and sanitize company name."""
        v = v.strip()
        if not v:
            raise ValueError("Company name cannot be empty")
        return v


class CompanyCreate(CompanyBase):
    """Model for company creation requests."""
    pass


class CompanyUpdate(BaseModel):
    """Model for updating company information."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("Company name cannot be empty")
        return v


class CompanyResponse(CompanyBase):
    """Model for company responses."""

    id: int = Field(..., description="Unique company identifier")

    model_config = ConfigDict(from_attributes=True)


class QuestionRequest(BaseModel):
    """Model for question requests."""

    company_id: int = Field(..., gt=0, description="Company identifier")
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question to ask",
        examples=["What is the company policy on remote work?"],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate and sanitize question."""
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v


class AnswerResponse(BaseModel):
    """Model for answer responses."""

    status: str = Field(..., description="Response status")
    answer: str = Field(..., description="Answer to the question")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class UploadResponse(BaseModel):
    """Model for file upload responses."""

    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    file_name: Optional[str] = Field(None, description="Uploaded file name")
    document_count: Optional[int] = Field(None, description="Number of documents processed")


class ErrorResponse(BaseModel):
    """Model for error responses."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    status_code: int = Field(..., description="HTTP status code")


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle events.

    Handles startup and shutdown operations including database connections
    and resource cleanup.
    """
    # Startup
    logger.info("Starting up application...")
    try:
        database.connect_to_db()
        settings.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down application...")
    try:
        # Cleanup temporary files
        if settings.TEMP_UPLOAD_DIR.exists():
            shutil.rmtree(settings.TEMP_UPLOAD_DIR, ignore_errors=True)
        # allow database to close if implemented
        try:
            database.close_db()
        except Exception:
            pass
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A production-ready multi-tenant AI assistant API with document Q&A capabilities",
    lifespan=lifespan,
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
)


# ============================================================================
# Middleware Configuration
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.TRUSTED_HOSTS,
)

# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            detail=str(exc.detail),
            status_code=exc.status_code,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            detail="An unexpected error occurred. Please try again later.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ).model_dump(),
    )


# ============================================================================
# Utility Functions
# ============================================================================

def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file against security and size constraints.

    Args:
        file: The uploaded file to validate

    Raises:
        HTTPException: If validation fails
    """
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}",
        )

    # Validate file size (if available)
    # Note: UploadFile does not always provide size; if your client sends Content-Length header you can use it.
    if hasattr(file, "size") and file.size and file.size > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {max_mb}MB",
        )


def cleanup_file(file_path: Path, background_tasks: BackgroundTasks) -> None:
    """
    Schedule file cleanup in background.

    Args:
        file_path: Path to file to cleanup
        background_tasks: FastAPI background tasks manager
    """
    def remove_file():
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

    background_tasks.add_task(remove_file)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    f"{settings.API_PREFIX}/",
    tags=["Health"],
    summary="API Health Check",
    response_model=dict,
)
async def root():
    """
    Health check endpoint.

    Returns basic API information and status.
    """
    return {
        "message": "Welcome to the Multi-Tenant AI Assistant API",
        "version": settings.APP_VERSION,
        "status": "healthy",
    }


@app.get(
    f"{settings.API_PREFIX}/health",
    tags=["Health"],
    summary="Detailed Health Check",
    response_model=dict,
)
async def health_check():
    """
    Detailed health check including database connectivity.

    Returns comprehensive health status of all system components.
    """
    health_status = {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "components": {
            "api": "operational",
            "database": "unknown",
        },
    }

    try:
        # Test database connection
        # database.get_db is expected to be a generator function returning Session
        db_gen = database.get_db()
        db = next(db_gen)
        # try a simple lightweight statement
        try:
            db.execute("SELECT 1")
            health_status["components"]["database"] = "operational"
        finally:
            # if the generator yields connection, make sure to close it if DB generator supports it
            try:
                db.close()
            except Exception:
                pass
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["status"] = "degraded"
        health_status["components"]["database"] = "unavailable"

    return health_status


# --- Company Management Endpoints ---

@app.get(
    f"{settings.API_PREFIX}/companies",
    tags=["Companies"],
    summary="List Companies",
    response_model=List[CompanyResponse],
    status_code=status.HTTP_200_OK,
)
async def list_companies(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(database.get_db),
):
    """
    Retrieve a list of all companies.

    Args:
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of companies
    """
    try:
        companies = db.query(database.Company).offset(skip).limit(limit).all()
        logger.info(f"Retrieved {len(companies)} companies")
        return companies
    except SQLAlchemyError as e:
        logger.error(f"Database error listing companies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve companies",
        )


@app.get(
    f"{settings.API_PREFIX}/companies/{{company_id}}",
    tags=["Companies"],
    summary="Get Company",
    response_model=CompanyResponse,
    status_code=status.HTTP_200_OK,
)
async def get_company(
    company_id: int,
    db: Session = Depends(database.get_db),
):
    """
    Retrieve a specific company by ID.

    Args:
        company_id: Company identifier
        db: Database session

    Returns:
        Company details

    Raises:
        HTTPException: If company not found
    """
    try:
        company = db.query(database.Company).filter(
            database.Company.id == company_id
        ).first()

        if not company:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Company with ID {company_id} not found",
            )

        logger.info(f"Retrieved company: {company.name}")
        return company
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving company {company_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve company",
        )


@app.post(
    f"{settings.API_PREFIX}/companies",
    tags=["Companies"],
    summary="Create Company",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_company(
    company: CompanyCreate,
    db: Session = Depends(database.get_db),
):
    """
    Create a new company.

    Args:
        company: Company creation data
        db: Database session

    Returns:
        Created company details

    Raises:
        HTTPException: If company name already exists
    """
    try:
        # Check for existing company
        existing = db.query(database.Company).filter(
            database.Company.name == company.name
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Company '{company.name}' already exists",
            )

        # Create new company
        new_company = database.Company(name=company.name)
        db.add(new_company)
        db.commit()
        db.refresh(new_company)

        # Initialize company collection in vector database
        try:
            qa_manager.get_company_collection(new_company.id)
        except Exception as e:
            logger.error(f"Failed to initialize collection for company {new_company.id}: {e}")
            # Rollback company creation if collection initialization fails
            try:
                db.delete(new_company)
                db.commit()
            except Exception:
                db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize company resources",
            )

        logger.info(f"Created company: {new_company.name} (ID: {new_company.id})")
        return new_company

    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error creating company: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create company",
        )


@app.patch(
    f"{settings.API_PREFIX}/companies/{{company_id}}",
    tags=["Companies"],
    summary="Update Company",
    response_model=CompanyResponse,
    status_code=status.HTTP_200_OK,
)
async def update_company(
    company_id: int,
    payload: CompanyUpdate,
    db: Session = Depends(database.get_db),
):
    """
    Update company fields (currently only name).

    Args:
        company_id: Company identifier
        payload: Fields to update
        db: Database session
    """
    try:
        company = db.query(database.Company).filter(database.Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

        if payload.name:
            # Ensure uniqueness
            existing = db.query(database.Company).filter(
                database.Company.name == payload.name,
                database.Company.id != company_id
            ).first()
            if existing:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Company name already in use")
            company.name = payload.name

        db.add(company)
        db.commit()
        db.refresh(company)
        return company
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Error updating company {company_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update company")


@app.delete(
    f"{settings.API_PREFIX}/companies/{{company_id}}",
    tags=["Companies"],
    summary="Delete Company",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_company(
    company_id: int,
    db: Session = Depends(database.get_db),
):
    """
    Delete a company and its associated resources.

    Args:
        company_id: Company identifier
        db: Database session
    """
    try:
        company = db.query(database.Company).filter(database.Company.id == company_id).first()
        if not company:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Company not found")

        # Try to remove vector DB collection first (best-effort)
        try:
            qa_manager.delete_company_collection(company_id)
        except Exception as e:
            logger.warning(f"Failed to delete vector collection for company {company_id}: {e}")

        db.delete(company)
        db.commit()
        logger.info(f"Deleted company {company_id}")
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Error deleting company {company_id}: {e}")
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete company")


# --- Document Q&A Endpoints ---

@app.post(
    f"{settings.API_PREFIX}/documents/upload",
    tags=["Documents"],
    summary="Upload PDF Document",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    company_id: int,
    file: UploadFile = File(..., description="PDF document to upload"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(database.get_db),
):
    """
    Upload and process a PDF document for a specific company.

    Args:
        company_id: Company identifier
        file: PDF file to upload
        background_tasks: Background task manager
        db: Database session

    Returns:
        Upload status and processing details

    Raises:
        HTTPException: If company not found or upload fails
    """
    # Validate company exists
    company = db.query(database.Company).filter(
        database.Company.id == company_id
    ).first()

    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with ID {company_id} not found",
        )

    # Validate file
    validate_file_upload(file)

    # Save file temporarily
    settings.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(file.filename).name  # simple sanitization
    file_path = settings.TEMP_UPLOAD_DIR / safe_filename

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Saved uploaded file: {file_path}")

        # Process PDF - expects qa_manager.process_pdf(path, company_id=...)
        result = qa_manager.process_pdf(str(file_path), company_id=company_id)

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to process PDF"),
            )

        logger.info(f"Processed PDF for company {company_id}: {file.filename}")

        # Schedule cleanup
        cleanup_file(file_path, background_tasks)

        return UploadResponse(
            status="success",
            message=result.get("message", "Document uploaded successfully"),
            file_name=safe_filename,
            document_count=result.get("document_count"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}",
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.post(
    f"{settings.API_PREFIX}/questions/ask",
    tags=["Questions"],
    summary="Ask Question",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(database.get_db),
):
    """
    Ask a question about company documents.

    Args:
        request: Question request containing company_id and question
        db: Database session

    Returns:
        Answer to the question with relevant context

    Raises:
        HTTPException: If company not found or query fails
    """
    # Validate company exists
    company = db.query(database.Company).filter(
        database.Company.id == request.company_id
    ).first()

    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with ID {request.company_id} not found",
        )

    try:
        # qa_manager.ask_question expected to return dict with keys: status, answer, metadata
        result = qa_manager.ask_question(
            pregunta=request.question,
            company_id=request.company_id,
        )

        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("answer", "Failed to process question"),
            )

        logger.info(f"Answered question for company {request.company_id}")

        return AnswerResponse(
            status=result.get("status", "success"),
            answer=result.get("answer", ""),
            metadata=result.get("metadata"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process question",
        )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )