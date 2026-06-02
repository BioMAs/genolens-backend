"""Pydantic schemas for the deployment management API."""
from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ServerConfig
# ---------------------------------------------------------------------------

class ServerConfigCreate(BaseModel):
    name: str
    host: str
    ssh_port: int = 22
    ssh_user: str = "ubuntu"
    ssh_key: Optional[str] = Field(None, description="PEM private key (will be encrypted at rest)")
    repo_path: str = "/opt/genolens"
    health_url: Optional[str] = None
    notes: Optional[str] = None


class ServerConfigUpdate(BaseModel):
    name: Optional[str] = None
    host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_user: Optional[str] = None
    ssh_key: Optional[str] = Field(None, description="Pass to replace the stored key; omit to keep current")
    repo_path: Optional[str] = None
    health_url: Optional[str] = None
    notes: Optional[str] = None


class ServerConfigResponse(BaseModel):
    id: UUID
    name: str
    host: str
    ssh_port: int
    ssh_user: str
    repo_path: str
    health_url: Optional[str]
    notes: Optional[str]
    has_ssh_key: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# DeploymentSecret
# ---------------------------------------------------------------------------

class SecretUpsert(BaseModel):
    """A single key/value pair for the .env.deploy file."""
    key: str
    value: str


class SecretKeyResponse(BaseModel):
    """Only expose the key name — never the encrypted value."""
    key: str
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# DeploymentJob
# ---------------------------------------------------------------------------

class DeploymentTrigger(BaseModel):
    server_id: UUID
    services: list[str] = Field(
        default=["backend"],
        description="Services to deploy: backend, license, ai"
    )
    skip_build: bool = False


class DeploymentJobResponse(BaseModel):
    id: UUID
    server_id: Optional[UUID]
    server_name: Optional[str] = None
    services: list[str]
    skip_build: bool
    status: str
    logs: Optional[str]
    triggered_by: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    created_at: datetime

    model_config = {"from_attributes": True}
