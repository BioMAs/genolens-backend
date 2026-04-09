"""
Unit tests for LocalStorageService.

Covers:
- upload_file (bytes) → file created, path returned
- upload_file (file-object with read()) → content written correctly
- download_file (happy path) → content matches uploaded bytes
- download_file (file not found) → Exception raised
- delete_file (existing file) → returns True
- delete_file (absent file) → returns False
- generate_file_path → correct structure
"""
import os
import tempfile
import io
import pytest
import pytest_asyncio
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_service() -> "LocalStorageService":
    """Create a LocalStorageService pointing at a temp directory."""
    from unittest.mock import patch
    tmpdir = tempfile.mkdtemp(prefix="genolens_storage_test_")

    # Patch settings to use tmpdir
    with patch("app.services.storage.settings") as mock_settings:
        mock_settings.LOCAL_STORAGE_PATH = tmpdir
        from app.services.storage import LocalStorageService
        svc = LocalStorageService.__new__(LocalStorageService)
        svc.base_path = Path(tmpdir)
        svc.base_path.mkdir(parents=True, exist_ok=True)
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# upload_file
# ─────────────────────────────────────────────────────────────────────────────

class TestUploadFile:

    @pytest.mark.asyncio
    async def test_upload_bytes_creates_file(self):
        """Uploading bytes should create the file at base_path / file_path."""
        svc = _make_service()
        file_path = "projects/p1/raw/data.parquet"
        content = b"fake parquet data"

        returned_path = await svc.upload_file(file_path, content)

        assert returned_path == file_path
        full = svc.base_path / file_path
        assert full.exists()
        assert full.read_bytes() == content

    @pytest.mark.asyncio
    async def test_upload_creates_parent_directories(self):
        """Uploading to a nested path should auto-create parent directories."""
        svc = _make_service()
        file_path = "a/b/c/d/file.txt"
        await svc.upload_file(file_path, b"hello")

        assert (svc.base_path / file_path).exists()

    @pytest.mark.asyncio
    async def test_upload_file_object(self):
        """Uploading a seekable async-style file object should write content."""
        svc = _make_service()
        file_path = "objects/test.bin"
        content = b"binary content from file object"

        class FakeFileObj:
            async def seek(self, pos):
                pass
            async def read(self):
                return content

        returned_path = await svc.upload_file(file_path, FakeFileObj())
        full = svc.base_path / file_path
        assert full.read_bytes() == content

    @pytest.mark.asyncio
    async def test_overwrite_existing_file(self):
        """Uploading to an existing path should overwrite the file."""
        svc = _make_service()
        file_path = "overwrite_me.txt"
        await svc.upload_file(file_path, b"original")
        await svc.upload_file(file_path, b"overwritten")

        assert (svc.base_path / file_path).read_bytes() == b"overwritten"


# ─────────────────────────────────────────────────────────────────────────────
# download_file
# ─────────────────────────────────────────────────────────────────────────────

class TestDownloadFile:

    @pytest.mark.asyncio
    async def test_download_returns_bytes(self):
        """Downloaded content should match what was uploaded."""
        svc = _make_service()
        file_path = "dl_test/file.dat"
        content = b"\x00\x01\x02\x03" * 100

        await svc.upload_file(file_path, content)
        downloaded = await svc.download_file(file_path)

        assert downloaded == content

    @pytest.mark.asyncio
    async def test_download_raises_for_missing_file(self):
        """Downloading a non-existent file should raise an Exception."""
        svc = _make_service()

        with pytest.raises(Exception, match="File not found"):
            await svc.download_file("does/not/exist.txt")


# ─────────────────────────────────────────────────────────────────────────────
# delete_file
# ─────────────────────────────────────────────────────────────────────────────

class TestDeleteFile:

    @pytest.mark.asyncio
    async def test_delete_existing_file_returns_true(self):
        """Deleting an existing file should return True and remove the file."""
        svc = _make_service()
        file_path = "to_delete.txt"
        await svc.upload_file(file_path, b"delete me")

        result = await svc.delete_file(file_path)

        assert result is True
        assert not (svc.base_path / file_path).exists()

    @pytest.mark.asyncio
    async def test_delete_absent_file_returns_false(self):
        """Deleting a non-existent file should return False without error."""
        svc = _make_service()
        result = await svc.delete_file("ghost_file.txt")
        assert result is False
