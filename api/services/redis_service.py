import uuid
import redis
import cv2
import numpy as np


class RedisService:
    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0, ttl: int = 3600
    ):
        self.client = redis.Redis(host=host, port=port, db=db)
        self.ttl = ttl  # tempo em segundos (1 hora padrão)

    def save_image(self, image: np.ndarray) -> str:
        """Salva imagem no Redis e retorna o job_id"""
        job_id = str(uuid.uuid4())
        _, buffer = cv2.imencode(".jpg", image)
        self.client.setex(f"img:{job_id}", self.ttl, buffer.tobytes())
        return job_id

    def get_image(self, job_id: str) -> np.ndarray | None:
        """Busca imagem no Redis pelo job_id"""
        data = self.client.get(f"img:{job_id}")
        if data is None:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def save_json(self, job_id: str, data: dict) -> None:
        """Salva resultado JSON vinculado ao mesmo job_id"""
        import json

        self.client.setex(f"result:{job_id}", self.ttl, json.dumps(data))

    def get_json(self, job_id: str) -> dict | None:
        import json

        data = self.client.get(f"result:{job_id}")
        if data is None:
            return None
        return json.loads(data)

    def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False
