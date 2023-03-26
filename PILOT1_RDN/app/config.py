from typing import List

from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Load forecasting app"
    token_issuer: str = "token-issuer-url"
    client_id: str = "client_id"
    client_secret: str = "client_secret"
    admin_routes_roles: List[str] = ["inergy_admin"]
    basic_routes_roles: List[str] = ["inergy_admin", "data_scientist"]


settings = Settings()