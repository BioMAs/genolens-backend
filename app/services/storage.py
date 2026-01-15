"""
Local Storage service for file operations.
Replaces Supabase Storage with local filesystem.
"""
import os
import aiofiles
from pathlib import Path
from typing import BinaryIO, Union
from uuid import UUID
from app.core.config import settings


class LocalStorageService:
    """Service for interacting with local filesystem storage."""

    def __init__(self):
        self.base_path = Path(settings.LOCAL_STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def upload_file(
        self,
        file_path: str,
        file_data: Union[BinaryIO, bytes],
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to local storage.

        Args:
            file_path: Path within the storage directory (e.g., "projects/{project_id}/raw/data.csv")
            file_data: File binary data or bytes
            content_type: MIME type of the file (unused for local storage but kept for compatibility)

        Returns:
            str: Relative path to the uploaded file
        """
        full_path = self.base_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(file_data, bytes):
            async with aiofiles.open(full_path, 'wb') as f:
                await f.write(file_data)
        else:
            # If it's an UploadFile or similar with async read
            if hasattr(file_data, 'read'):
                if hasattr(file_data, 'seek'):
                    await file_data.seek(0)
                content = await file_data.read()
                async with aiofiles.open(full_path, 'wb') as f:
                    await f.write(content)
            else:
                # Fallback for sync file objects
                content = file_data.read()
                async with aiofiles.open(full_path, 'wb') as f:
                    await f.write(content)

        return file_path

    async def download_file(self, file_path: str) -> bytes:
        """
        Download a file from local storage.

        Args:
            file_path: Path within the storage directory

        Returns:
            bytes: File content

        Raises:
            Exception: If file not found
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            raise Exception(f"File not found: {file_path}")

        async with aiofiles.open(full_path, 'rb') as f:
            return await f.read()

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from local storage.

        Args:
            file_path: Path within the storage directory

        Returns:
            bool: True if deleted, False if not found
        """
        full_path = self.base_path / file_path
        
        if full_path.exists():
            os.remove(full_path)
            return True
        return False

    def generate_file_path(
        self,
        project_id: UUID,
        dataset_id: UUID,
        filename: str,
        subfolder: str = "raw"
    ) -> str:
        """
        Generate a standardized file path for storage.

        Args:
            project_id: Project UUID
            dataset_id: Dataset UUID
            filename: Original filename
            subfolder: Subfolder (raw, processed, etc.)

        Returns:
            str: Standardized file path
        """
        return f"projects/{project_id}/{subfolder}/{dataset_id}/{filename}"


# Singleton instance
storage_service = LocalStorageService()
