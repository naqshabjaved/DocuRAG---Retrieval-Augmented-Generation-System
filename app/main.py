from fastapi import FastAPI
from app.api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="DocuRAG - Retrieval Augmented Generation System",
        description="RAG-based document question answering using LLMs",
        version="0.1.0"
    )

    # Include API routes
    app.include_router(api_router)

    return app


app = create_app()
