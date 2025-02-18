import os

from dotenv import load_dotenv, find_dotenv
from pydantic.v1 import BaseSettings, Field

load_dotenv(find_dotenv())


class ApplicationSetting(BaseSettings):
    num_cpus: int = Field(default=os.cpu_count())
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=10000)
    dashboard_port: int = Field(default=10001)
    open_api_key: str = Field(default="")
