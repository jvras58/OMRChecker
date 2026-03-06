import io
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import cv2

from api.services.omr_service import OMRService
from api.services.redis_service import RedisService
from api.settings.config import settings

router = APIRouter(prefix="/omr", tags=["OMR"])


def get_omr_service() -> OMRService:
    return _omr_service


def get_redis_service() -> RedisService:
    return _redis_service


# Instâncias únicas (inicializadas no startup da app)
_omr_service: OMRService | None = None
_redis_service: RedisService | None = None


def init_services():
    global _omr_service, _redis_service
    from pathlib import Path

    _omr_service = OMRService(sample_dir=Path(settings.sample_dir))
    _redis_service = RedisService(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        ttl=settings.redis_ttl,
    )


# ------------------------------------------------------------------
# POST /omr/process
# ------------------------------------------------------------------


@router.post(
    "/process",
    summary="Processa cartão OMR",
    response_description="Respostas detectadas + job_id para buscar imagem",
)
async def process_omr(
    file: UploadFile = File(..., description="Imagem do cartão OMR (jpg/png)"),
    omr_service: OMRService = Depends(get_omr_service),
    redis_service: RedisService = Depends(get_redis_service),
):
    """
    Envia a imagem do cartão de respostas e recebe:
    - **omr_response**: dicionário com as respostas detectadas
    - **multi_marked**: número de questões com múltipla marcação
    - **job_id**: use em `GET /omr/result-image/{job_id}` para obter a imagem anotada
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de arquivo não suportado: {file.content_type}. Use JPEG ou PNG.",
        )

    image_bytes = await file.read()

    try:
        result = omr_service.process_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro interno ao processar OMR: {str(e)}"
        )

    # Salva a imagem anotada no Redis
    job_id = redis_service.save_image(result["final_marked"])
    redis_service.save_json(
        job_id,
        {
            "omr_response": result["omr_response"],
            "multi_marked": result["multi_marked"],
        },
    )

    return JSONResponse(
        {
            "job_id": job_id,
            "omr_response": result["omr_response"],
            "multi_marked": result["multi_marked"],
            "image_url": f"/omr/result-image/{job_id}",
        }
    )


# ------------------------------------------------------------------
# GET /omr/result-image/{job_id}
# ------------------------------------------------------------------


@router.get(
    "/result-image/{job_id}",
    summary="Retorna imagem anotada do processamento",
    response_class=StreamingResponse,
)
async def get_result_image(
    job_id: str,
    redis_service: RedisService = Depends(get_redis_service),
):
    """
    Retorna a imagem JPEG com as bolhas marcadas/detectadas.
    O `job_id` é obtido do endpoint `POST /omr/process`.
    A imagem fica disponível por **1 hora** no Redis.
    """
    image = redis_service.get_image(job_id)

    if image is None:
        raise HTTPException(
            status_code=404,
            detail=f"Imagem não encontrada ou expirada para job_id='{job_id}'.",
        )

    # Codifica de volta para JPEG
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=result_{job_id}.jpg"},
    )


# ------------------------------------------------------------------
# GET /omr/result/{job_id}  (opcional — busca só o JSON)
# ------------------------------------------------------------------


@router.get(
    "/result/{job_id}",
    summary="Retorna resultado JSON de um processamento anterior",
)
async def get_result(
    job_id: str,
    redis_service: RedisService = Depends(get_redis_service),
):
    data = redis_service.get_json(job_id)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Resultado não encontrado ou expirado para job_id='{job_id}'.",
        )
    return data
