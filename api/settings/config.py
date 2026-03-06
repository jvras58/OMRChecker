from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sample_dir: str = "samples/simureka"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_ttl: int = 3600  # 1 hora

    class Config:
        env_file = ".env"


settings = Settings()
