import os
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def create_app() -> FastAPI:
    app = FastAPI(title="AI Study QnA", version="0.1.0")

    # CORS configuration - must be before other middleware
    # In production, allow all origins (for public deployment)
    # In development, use specific origins
    is_production = os.getenv("ENVIRONMENT", "").lower() == "production" or os.getenv("RENDER", "") == "true" or os.getenv("RAILWAY_ENVIRONMENT", "") != ""
    
    if is_production:
        # Production: Allow all origins for public access
        all_origins = ["*"]
    else:
        # Development: Allow specific origins from environment or default to localhost
        allowed_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
        default_origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
        # Combine and filter empty strings
        all_origins = [origin.strip() for origin in allowed_origins + default_origins if origin.strip()]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=all_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=3600,
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/hello")
    async def hello():
        return {"message": "Hello from FastAPI"}

    # Global exception handler for unhandled exceptions (not HTTPException)
    # This ensures CORS headers are always included even when unexpected errors occur
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # Let FastAPI handle HTTPException (it already includes CORS headers via middleware)
        if isinstance(exc, HTTPException):
            raise exc
        
        print(f"[Global Exception Handler] {type(exc).__name__}: {str(exc)}")
        import traceback
        print(f"[Global Exception Handler] Traceback: {traceback.format_exc()}")
        
        origin = request.headers.get("origin")
        cors_headers = {}
        # In production, allow all origins
        is_production = os.getenv("ENVIRONMENT", "").lower() == "production" or os.getenv("RENDER", "") == "true" or os.getenv("RAILWAY_ENVIRONMENT", "") != ""
        
        if is_production or (origin and origin):
            cors_headers = {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
                "Access-Control-Allow-Headers": "*",
            }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": f"Lỗi server: {str(exc)}"
            },
            headers=cors_headers
        )

    # Routers (Module 3+)
    from .routers import auth, documents, query, history, admin, calendar, quiz

    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(query.router, prefix="/query", tags=["query"])
    app.include_router(history.router, prefix="/history", tags=["history"])
    app.include_router(admin.router, prefix="/admin", tags=["admin"])
    app.include_router(calendar.router, prefix="/calendar", tags=["calendar"])
    app.include_router(quiz.router, prefix="/quiz", tags=["quiz"])

    return app


app = create_app()


