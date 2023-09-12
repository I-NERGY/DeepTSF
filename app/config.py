from typing import List
from pydantic import BaseSettings
import os

# get environment variables
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    app_name: str = "Load forecasting app"
    token_issuer: str = os.environ.get("TOKEN_ISSUER_URL")
    client_id: str = os.environ.get("KEYCLOAK_ID")
    client_secret: str = os.environ.get("KEYCLOAK_SECRET")
    admin_routes_roles: List[str] = ["inergy_admin"]
    scientist_routes_roles: List[str] = ["inergy_admin", "data_scientist"]
    engineer_routes_roles: List[str] = ["inergy_admin", "energy_engineer"]
    common_routes_roles: List[str] = ["inergy_admin", "data_scientist", "energy_engineer"]



settings = Settings()