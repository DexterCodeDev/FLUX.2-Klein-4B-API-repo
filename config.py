from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    MODEL_ID: str = "black-forest-labs/FLUX.2-klein-4B"

    HF_HOME: str = "/tmp/huggingface"

    MAX_IMAGE_SIZE: int = 2048

    class Config:
        env_file = ".env"


settings = Settings()
