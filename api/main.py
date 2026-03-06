from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes.omr import router as omr_router, init_services


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_services()
    print("✅ OMR Service e Redis inicializados")
    yield
    # Shutdown (cleanup se necessário)
    print("🛑 Encerrando API")


app = FastAPI(
    title="OMRChecker API",
    description="API para leitura automática de cartões de múltipla escolha",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(omr_router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
