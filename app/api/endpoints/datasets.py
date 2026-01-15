"""
Dataset API endpoints.
"""
import logging
import json
import hashlib
import numpy as np
from uuid import UUID
from pathlib import Path
from typing import Annotated, Optional
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Query,
    Body
)

from app.api.deps import get_current_user, get_db
from app.api.deps.subscription import (
    get_or_create_user,
    require_ai_access,
    check_ai_quota,
    increment_ai_usage
)
# from app.db.session import get_db  <-- Removed
from app.core.supabase_auth import SupabaseUser
from app.core.config import settings
from app.models.models import Project, Dataset, DatasetStatus, DatasetType, GeneSetDatabase, DegGene, EnrichmentPathway, ProjectMember, AIConversation, AIInterpretation, User
from sqlalchemy import select, func, delete, text, or_, desc, asc, and_
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.dataset import (
    DatasetResponse,
    DatasetUploadResponse,
    DatasetQueryParams,
    DatasetQueryResponse,
    DatasetUpdate
)
from app.services.storage import storage_service
from app.services.data_processor import data_processor
from app.worker.tasks import process_dataset_upload
from app.services.gsea_processor import GSEAProcessor, prepare_ranked_gene_list, GeneSetsLoader
from app.services.gene_set_loader import GeneSetLoader
from app.models.models import GeneSetDatabase
from app.services.ai_interpreter import LocalAIInterpreter
from app.services.clustering_service import ClusteringService
from sqlalchemy import text
from datetime import datetime

clustering_service = ClusteringService()

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    project_id: UUID = Form(...),
    name: str = Form(...),
    dataset_type: DatasetType = Form(...),
    description: str = Form(None),
    comparison_name: str = Form(None),
    is_normalized: bool = Form(False),
    contains_all_genes: bool = Form(True),
    file: UploadFile = File(...)
) -> dict:
    """
    Upload a dataset file.
    """
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension {file_extension} not allowed. "
                   f"Allowed: {', '.join(settings.ALLOWED_FILE_EXTENSIONS)}"
        )

    # Check project ownership
    query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Create dataset entry
    metadata = {
        "original_filename": file.filename,
        "file_size": file.size,
        "is_normalized": is_normalized,
        "contains_all_genes": contains_all_genes
    }
    
    if comparison_name:
        metadata["comparison_name"] = comparison_name

    # Upload raw file to Local Storage
    file_path = f"projects/{project_id}/raw/{file.filename}"
    uploaded_path = await storage_service.upload_file(file_path, file)

    # Create dataset record
    dataset = Dataset(
        project_id=project_id,
        name=name,
        type=dataset_type,
        description=description,
        status=DatasetStatus.PENDING,
        raw_file_path=uploaded_path,
        dataset_metadata=metadata,
        column_mapping={}
    )
    
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    # Trigger Celery task
    process_dataset_upload.delay(str(dataset.id), uploaded_path)

    return {
        "dataset_id": dataset.id,
        "message": f"Dataset '{name}' uploaded successfully and processing has started",
        "status": dataset.status
    }



@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Get dataset metadata by ID.
    """
    # Get dataset with project join to verify ownership
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    return dataset


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: UUID,
    dataset_in: DatasetUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Update dataset metadata.
    """
    # Check dataset exists and user has access
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    update_data = dataset_in.model_dump(exclude_unset=True)

    if "dataset_metadata" in update_data:
        # Merge existing metadata with new metadata
        current_metadata = dict(dataset.dataset_metadata or {})
        current_metadata.update(update_data["dataset_metadata"])
        update_data["dataset_metadata"] = current_metadata

    if not update_data:
        return dataset

    # Update dataset
    for key, value in update_data.items():
        setattr(dataset, key, value)
    
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)

    return dataset


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> None:
    """
    Delete a dataset and all associated data.

    This will:
    1. Delete files from Supabase Storage (raw and parquet files)
    2. Delete associated DEG genes from database
    3. Delete associated enrichment pathways from database
    4. Delete the dataset entry itself

    Requires project ownership.
    """
    # Get dataset with ownership check
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Delete files from storage
    files_deleted = []
    files_failed = []

    if dataset.raw_file_path:
        try:
            await storage_service.delete_file(dataset.raw_file_path)
            files_deleted.append(dataset.raw_file_path)
        except Exception as e:
            files_failed.append(f"raw_file: {str(e)}")

    if dataset.parquet_file_path:
        try:
            await storage_service.delete_file(dataset.parquet_file_path)
            files_deleted.append(dataset.parquet_file_path)
        except Exception as e:
            files_failed.append(f"parquet_file: {str(e)}")

    # Delete the dataset itself (Cascade delete handles related tables)
    await db.delete(dataset)
    await db.commit()

    # Log the deletion
    print(f"[DELETE] Dataset {dataset_id} deleted by user {current_user.user_id}")
    print(f"[DELETE] Files deleted: {files_deleted}")
    if files_failed:
        print(f"[DELETE] File deletion warnings: {files_failed}")


@router.post("/{dataset_id}/reprocess")
async def reprocess_dataset(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Reprocess an existing dataset (recalculate metadata, PCA, etc.).
    Useful when a dataset file exists but needs to be reanalyzed.
    """
    # Get dataset with ownership check
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    if not dataset.raw_file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No raw file found for this dataset"
        )

    # Reset status to PENDING and clear error
    dataset.status = DatasetStatus.PENDING
    dataset.error_message = None
    db.add(dataset)
    await db.commit()

    # Trigger reprocessing with is_reprocess=True flag
    process_dataset_upload.delay(
        dataset_id=str(dataset_id),
        raw_file_path=dataset.raw_file_path,
        is_reprocess=True
    )

    return {
        "dataset_id": dataset_id,
        "message": "Dataset reprocessing started.",
        "status": DatasetStatus.PENDING
    }


@router.post("/{dataset_id}/query", response_model=DatasetQueryResponse)
async def query_dataset(
    dataset_id: UUID,
    query_params: DatasetQueryParams,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Query a dataset by reading its Parquet file.

    This endpoint demonstrates the "lazy loading" architecture:
    - Metadata is stored in PostgreSQL
    - Actual data is in Parquet files
    - Data is loaded on-demand with filters

    - **gene_ids**: Optional list of gene IDs to filter
    - **sample_ids**: Optional list of sample column names to include
    - **limit**: Maximum rows to return (default: 100, max: 100000)
    - **offset**: Number of rows to skip (default: 0)
    """
    # Get dataset metadata with ownership check
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Check if dataset is ready
    if dataset.status != DatasetStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dataset is not ready. Current status: {dataset.status}"
        )

    if not dataset.parquet_file_path:
        # Try to infer from raw_file_path if available
        if dataset.raw_file_path:
            raw_path = dataset.raw_file_path
            # Assuming standard naming convention: /raw/filename.ext -> /processed/filename.parquet
            if "/raw/" in raw_path:
                inferred_path = raw_path.replace("/raw/", "/processed/")
                p = Path(inferred_path)
                dataset.parquet_file_path = str(p.with_suffix('.parquet'))
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Parquet file path not found and could not be inferred"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Parquet file path not found"
            )

    try:
        # Download Parquet file
        parquet_data = await storage_service.download_file(dataset.parquet_file_path)

        # Query the data
        query_result = await data_processor.query_parquet(
            parquet_data=parquet_data,
            gene_ids=query_params.gene_ids,
            sample_ids=query_params.sample_ids,

            limit=query_params.limit,
            offset=query_params.offset
        )

        return query_result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query dataset: {str(e)}"
        )


@router.get("/project/{project_id}", response_model=list[DatasetResponse])
async def list_project_datasets(
    project_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    dataset_type: Optional[DatasetType] = Query(None, description="Filter by dataset type"),
    status: Optional[DatasetStatus] = Query(None, description="Filter by status"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=100, ge=1, le=500, description="Items per page")
) -> list[Dataset]:
    """
    List datasets for a specific project with optional filtering and pagination.

    - **dataset_type**: Filter by MATRIX, DEG, ENRICHMENT, etc.
    - **status**: Filter by PENDING, PROCESSING, READY, FAILED
    - **page**: Page number (starts at 1)
    - **page_size**: Number of items per page (max 500)
    """
    # Check project ownership
    query = select(Project).where(
        Project.id == project_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Query datasets
    query = select(Dataset).where(Dataset.project_id == project_id)
    
    if dataset_type:
        query = query.where(Dataset.type == dataset_type)
    
    if status:
        query = query.where(Dataset.status == status)
    
    offset = (page - 1) * page_size
    query = query.order_by(Dataset.created_at.desc()).offset(offset).limit(page_size)
    
    result = await db.execute(query)
    datasets = result.scalars().all()

    return datasets



@router.get("/{dataset_id}/pca", response_model=dict)
async def get_dataset_pca(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    n_components: int = Query(2, ge=2, le=3)
) -> dict:
    """
    Calculate PCA for an expression matrix dataset.
    """
    # Fetch dataset with ownership check
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Dataset.status == DatasetStatus.READY,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or not ready"
        )

    if dataset.type != DatasetType.MATRIX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PCA can only be calculated on expression matrices"
        )

    # Check if PCA is already stored in metadata
    pca_key = f"pca_{n_components}d"
    dataset_metadata = dataset.dataset_metadata or {}
    if pca_key in dataset_metadata:
        return dataset_metadata[pca_key]

    # If not cached, calculate on-demand
    # Download parquet file
    parquet_file_path = dataset.parquet_file_path
    if not parquet_file_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No parquet file found for this dataset"
        )

    try:
        parquet_data = await storage_service.download_file(parquet_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset file: {str(e)}"
        )

    # Calculate PCA
    try:
        pca_result = await data_processor.calculate_pca(parquet_data, n_components=n_components)

        # Store in metadata for future use
        updated_metadata = dict(dataset_metadata)
        updated_metadata[pca_key] = pca_result
        
        dataset.dataset_metadata = updated_metadata
        db.add(dataset)
        await db.commit()

        return pca_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate PCA: {str(e)}"
        )


@router.get("/{dataset_id}/umap", response_model=dict)
async def get_dataset_umap(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    n_components: int = Query(2, ge=2, le=3),
    n_neighbors: int = Query(15, ge=2, le=200),
    min_dist: float = Query(0.1, ge=0.0, le=0.99)
) -> dict:
    """
    Calculate UMAP for an expression matrix dataset.
    """
    # Fetch dataset with ownership check
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Dataset.status == DatasetStatus.READY,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or not ready"
        )

    if dataset.type != DatasetType.MATRIX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="UMAP can only be calculated on expression matrices"
        )

    # Check if UMAP is already stored in metadata
    umap_key = f"umap_{n_components}d_n{n_neighbors}_d{min_dist}"
    dataset_metadata = dataset.dataset_metadata or {}
    if umap_key in dataset_metadata:
        return dataset_metadata[umap_key]

    # If not cached, calculate on-demand
    # Download parquet file
    parquet_file_path = dataset.parquet_file_path
    if not parquet_file_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No parquet file found for this dataset"
        )

    try:
        parquet_data = await storage_service.download_file(parquet_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset file: {str(e)}"
        )

    # Calculate UMAP
    try:
        umap_result = await data_processor.calculate_umap(
            parquet_data, 
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )

        # Store in metadata for future use
        updated_metadata = dict(dataset_metadata)
        updated_metadata[umap_key] = umap_result
        
        dataset.dataset_metadata = updated_metadata
        db.add(dataset)
        await db.commit()

        return umap_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate UMAP: {str(e)}"
        )


@router.get("/{dataset_id}/library_size", response_model=list[dict])
async def get_dataset_library_size(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> list[dict]:
    """
    Calculate library size for an expression matrix dataset.
    """
    # Fetch dataset with ownership check
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Dataset.status == DatasetStatus.READY,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found or not ready"
        )

    if dataset.type != DatasetType.MATRIX:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Library size can only be calculated on expression matrices"
        )

    # Download parquet file
    parquet_file_path = dataset.parquet_file_path
    if not parquet_file_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No parquet file found for this dataset"
        )

    try:
        parquet_data = await storage_service.download_file(parquet_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset file: {str(e)}"
        )


    # Calculate Library Size
    try:
        lib_size_result = await data_processor.calculate_library_size(parquet_data)
        return lib_size_result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate library size: {str(e)}"
        )


@router.get("/{dataset_id}/comparisons")
async def list_dataset_comparisons(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Debug endpoint: List all comparisons available in a dataset's metadata.
    """
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    metadata = dataset.dataset_metadata or {}
    columns_info = metadata.get('columns_info', {})
    comparisons = columns_info.get('comparisons') or metadata.get('comparisons', {})

    return {
        'dataset_id': str(dataset_id),
        'dataset_name': dataset.name,
        'comparisons': list(comparisons.keys()),
        'comparison_details': comparisons
    }


@router.get("/{dataset_id}/comparisons/stats")
async def get_dataset_comparisons_stats(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Get statistics (UP/DOWN counts) for all comparisons in the dataset.
    """
    # Check dataset ownership
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Check if stats are available in metadata
    metadata = dataset.dataset_metadata or {}
    comparisons_meta = metadata.get('comparisons', {})
    
    # If metadata has stats, use them (much faster)
    if comparisons_meta and any('deg_up' in comp for comp in comparisons_meta.values()):
        stats = {}
        for comp_name, comp_data in comparisons_meta.items():
            stats[comp_name] = {
                "up": comp_data.get('deg_up', 0),
                "down": comp_data.get('deg_down', 0),
                "total": comp_data.get('deg_total', 0)
            }
        
        return {
            "dataset_id": str(dataset_id),
            "stats": stats,
            "source": "metadata"
        }

    # Fallback: Aggregate counts by comparison and regulation from DB
    stmt = select(
        DegGene.comparison_name,
        DegGene.regulation,
        func.count(DegGene.id)
    ).where(
        DegGene.dataset_id == dataset_id
    ).group_by(
        DegGene.comparison_name,
        DegGene.regulation
    )

    result = await db.execute(stmt)
    rows = result.all()

    stats = {}
    for comparison_name, regulation, count in rows:
        if comparison_name not in stats:
            stats[comparison_name] = {"up": 0, "down": 0, "total": 0}
        
        if regulation == "UP":
            stats[comparison_name]["up"] = count
        elif regulation == "DOWN":
            stats[comparison_name]["down"] = count
        
        stats[comparison_name]["total"] += count

    # Save the calculated stats back to metadata for future requests
    if stats and comparisons_meta:
        for comp_name, comp_stats in stats.items():
            if comp_name in comparisons_meta:
                comparisons_meta[comp_name]['deg_up'] = comp_stats['up']
                comparisons_meta[comp_name]['deg_down'] = comp_stats['down']
                comparisons_meta[comp_name]['deg_total'] = comp_stats['total']
        
        metadata['comparisons'] = comparisons_meta
        dataset.dataset_metadata = metadata
        db.add(dataset)
        await db.commit()

    return {
        "dataset_id": str(dataset_id),
        "stats": stats,
        "source": "database"
    }


@router.get("/{dataset_id}/diagnose-deg/{comparison_name}")
async def diagnose_deg_filtering(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Diagnostic endpoint to analyze DEG filtering for a specific comparison.
    Returns detailed information about p-value distributions, logFC distributions,
    and step-by-step filtering results.
    """
    import pandas as pd
    import io

    # Get dataset with ownership check
    print(f"DEBUG: Looking for dataset {dataset_id} for user {current_user.user_id}")
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        print(f"DEBUG: Dataset not found for user {current_user.user_id}")
        # Try to find dataset without user filter to see if it exists
        check_query = select(Dataset).where(Dataset.id == dataset_id)
        check_result = await db.execute(check_query)
        check_dataset = check_result.scalar_one_or_none()
        
        if check_dataset:
            print(f"DEBUG: Dataset exists but belongs to different project")
        else:
            print(f"DEBUG: Dataset does not exist at all")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found or not accessible"
        )

    # Get column information from metadata
    metadata = dataset.dataset_metadata or {}
    columns_info = metadata.get('columns_info', {})
    comparisons = columns_info.get('comparisons') or metadata.get('comparisons', {})

    # Find comparison: exact match or with prefix pattern [prefix]:comparison_name
    matched_comparison = None
    comp_info = None

    if comparison_name in comparisons:
        # Exact match
        matched_comparison = comparison_name
        comp_info = comparisons[comparison_name]
    else:
        # Try to find with prefix pattern (prefix:name, prefix_name, or prefix.name)
        for key in comparisons.keys():
            # Check if key ends with separator + comparison_name
            if (key.endswith(f":{comparison_name}") or
                key.endswith(f"_{comparison_name}") or
                key.endswith(f".{comparison_name}")):
                matched_comparison = key
                comp_info = comparisons[key]
                break
            # Also check if the comparison_name appears anywhere in the key (case-insensitive)
            if comparison_name.lower() in key.lower():
                matched_comparison = key
                comp_info = comparisons[key]
                break

    if not comp_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison '{comparison_name}' not found. Available: {list(comparisons.keys())}"
        )
    logfc_col = comp_info.get('logFC')
    padj_col = comp_info.get('padj')
    contrast_col = comp_info.get('contrast')

    if not logfc_col or not padj_col:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing logFC or padj column information"
        )

    # Download and load data
    storage_path = dataset.parquet_file_path
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No storage path found for this dataset"
        )

    try:
        parquet_data = await storage_service.download_file(storage_path)
        df = pd.read_parquet(io.BytesIO(parquet_data))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load dataset: {str(e)}"
        )

    # Check if columns exist
    missing_cols = []
    if logfc_col not in df.columns:
        missing_cols.append(logfc_col)
    if padj_col not in df.columns:
        missing_cols.append(padj_col)

    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing columns: {missing_cols}"
        )

    # Analyze contrast column if present
    contrast_info = {}
    df_filtered = df.copy()

    if contrast_col and contrast_col in df.columns:
        contrast_values = df[contrast_col].value_counts(dropna=False).to_dict()
        contrast_info = {
            "column": contrast_col,
            "values": {str(k): int(v) for k, v in contrast_values.items()},
            "genes_with_contrast": int((df[contrast_col].notna() & (df[contrast_col] != '')).sum())
        }
        # Note: Not filtering by contrast here to see all data

    # P-value analysis
    padj_stats = {
        'total': len(df_filtered),
        'null_or_na': int(df_filtered[padj_col].isna().sum()),
        'equal_to_0': int((df_filtered[padj_col] == 0).sum()),
        '0_to_1e-300': int(((df_filtered[padj_col] > 0) & (df_filtered[padj_col] < 1e-300)).sum()),
        '1e-300_to_0.01': int(((df_filtered[padj_col] >= 1e-300) & (df_filtered[padj_col] < 0.01)).sum()),
        '0.01_to_0.05': int(((df_filtered[padj_col] >= 0.01) & (df_filtered[padj_col] < 0.05)).sum()),
        '0.05_to_0.1': int(((df_filtered[padj_col] >= 0.05) & (df_filtered[padj_col] < 0.1)).sum()),
        'gte_0.1': int((df_filtered[padj_col] >= 0.1).sum()),
    }

    # LogFC analysis
    logfc_stats = {
        'total': len(df_filtered),
        'null_or_na': int(df_filtered[logfc_col].isna().sum()),
        'equal_to_0': int((df_filtered[logfc_col] == 0).sum()),
        'positive_0_to_0.5': int(((df_filtered[logfc_col] > 0) & (df_filtered[logfc_col] <= 0.5)).sum()),
        'positive_0.5_to_1': int(((df_filtered[logfc_col] > 0.5) & (df_filtered[logfc_col] <= 1)).sum()),
        'positive_gt_1': int((df_filtered[logfc_col] > 1).sum()),
        'negative_0_to_-0.5': int(((df_filtered[logfc_col] < 0) & (df_filtered[logfc_col] >= -0.5)).sum()),
        'negative_-0.5_to_-1': int(((df_filtered[logfc_col] < -0.5) & (df_filtered[logfc_col] >= -1)).sum()),
        'negative_lt_-1': int((df_filtered[logfc_col] < -1).sum()),
    }

    # Progressive filtering
    step1 = df_filtered[df_filtered[padj_col].notna() & df_filtered[logfc_col].notna()]
    step2 = step1[step1[padj_col] > 0]
    step3 = step2[step2[padj_col] < 0.05]
    step4 = step3[step3[logfc_col].abs() > 1]

    up_genes = step4[step4[logfc_col] > 0]
    down_genes = step4[step4[logfc_col] < 0]

    progressive_filtering = {
        'start': len(df_filtered),
        'after_remove_nan': len(step1),
        'after_padj_gt_0': len(step2),
        'after_padj_lt_0.05': len(step3),
        'after_logfc_abs_gt_1': len(step4),
        'up_regulated': len(up_genes),
        'down_regulated': len(down_genes)
    }

    # Sample genes
    gene_col = 'gene_id' if 'gene_id' in df.columns else df.columns[0]

    sample_up = []
    sample_down = []

    if len(up_genes) > 0:
        top_up = up_genes.nlargest(10, logfc_col)
        sample_up = [
            {
                'gene': row[gene_col],
                'logFC': float(row[logfc_col]),
                'padj': float(row[padj_col])
            }
            for _, row in top_up.iterrows()
        ]

    if len(down_genes) > 0:
        top_down = down_genes.nsmallest(10, logfc_col)
        sample_down = [
            {
                'gene': row[gene_col],
                'logFC': float(row[logfc_col]),
                'padj': float(row[padj_col])
            }
            for _, row in top_down.iterrows()
        ]

    return {
        'dataset_id': str(dataset_id),
        'dataset_name': dataset.get("name"),
        'comparison_name': matched_comparison,
        'columns': {
            'logFC': logfc_col,
            'padj': padj_col,
            'contrast': contrast_col
        },
        'contrast_analysis': contrast_info,
        'padj_distribution': padj_stats,
        'logfc_distribution': logfc_stats,
        'progressive_filtering': progressive_filtering,
        'sample_genes': {
            'up_regulated': sample_up,
            'down_regulated': sample_down
        }
    }


@router.get("/{dataset_id}/deg-genes/{comparison_name}")
async def get_deg_genes(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    regulation: Optional[str] = Query(None, description="Filter by regulation: UP or DOWN"),
    padj_max: Optional[float] = Query(None, description="Maximum p-adjusted value"),
    logfc_min: Optional[float] = Query(None, description="Minimum absolute log fold change"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    sort_by: str = Query("padj", description="Sort by: padj, log_fc, gene_id"),
    sort_order: str = Query("asc", description="Sort order: asc or desc")
) -> dict:
    """
    Query DEG genes from database for a specific comparison.
    This is much faster than loading the entire Parquet file.

    - **regulation**: Filter by UP or DOWN regulation
    - **padj_max**: Maximum p-adjusted value (e.g., 0.05)
    - **logfc_min**: Minimum absolute log fold change (e.g., 0.58)
    - **page**: Page number (starts at 1)
    - **page_size**: Number of items per page (max 1000)
    - **sort_by**: Column to sort by (padj, log_fc, gene_id)
    - **sort_order**: Sort direction (asc or desc)
    """
    # Check dataset ownership
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    # Build query for DegGene
    stmt = select(DegGene).where(
        DegGene.dataset_id == dataset_id,
        DegGene.comparison_name == comparison_name
    )

    if regulation:
        stmt = stmt.where(DegGene.regulation == regulation.upper())

    if padj_max is not None:
        stmt = stmt.where(DegGene.padj <= padj_max)

    if logfc_min is not None:
        # ABS(log_fc) >= min  <=>  log_fc >= min OR log_fc <= -min
        stmt = stmt.where(or_(DegGene.log_fc >= logfc_min, DegGene.log_fc <= -logfc_min))

    # Count total matching genes
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar()

    # Count UP and DOWN regulated genes (without filters except dataset/comparison)
    base_stmt = select(func.count()).where(
        DegGene.dataset_id == dataset_id,
        DegGene.comparison_name == comparison_name
    )
    
    total_up_result = await db.execute(base_stmt.where(DegGene.regulation == "UP"))
    total_up = total_up_result.scalar()
    
    total_down_result = await db.execute(base_stmt.where(DegGene.regulation == "DOWN"))
    total_down = total_down_result.scalar()

    # Sorting
    valid_sort_columns = {
        "padj": DegGene.padj,
        "log_fc": DegGene.log_fc,
        "gene_id": DegGene.gene_id
    }
    sort_col = valid_sort_columns.get(sort_by, DegGene.padj)
    if sort_order.lower() == "desc":
        stmt = stmt.order_by(desc(sort_col))
    else:
        stmt = stmt.order_by(asc(sort_col))

    # Pagination
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    genes = [
        {
            "gene_id": row.gene_id,
            "log_fc": row.log_fc,
            "padj": row.padj,
            "regulation": row.regulation,
            "pvalue": row.pvalue,
            "base_mean": row.base_mean,
            "gene_name": row.gene_name
        }
        for row in rows
    ]

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return {
        "dataset_id": str(dataset_id),
        "comparison_name": comparison_name,
        "genes": genes,
        "total_up": total_up,
        "total_down": total_down,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages
        },
        "filters": {
            "regulation": regulation,
            "padj_max": padj_max,
            "logfc_min": logfc_min
        }
    }


@router.get("/{dataset_id}/volcano-plot/{comparison_name}")
async def get_volcano_plot_data(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    max_points: int = Query(5000, ge=100, le=20000, description="Maximum number of points to return"),
    force_recalculate: bool = Query(False, description="Force recalculation from Parquet file")
) -> dict:
    """
    Get volcano plot data for a comparison.
    Returns cached pre-calculated data when available, or computes on-demand from Parquet.
    
    - **max_points**: Maximum number of points (significant genes always included)
    - **force_recalculate**: Bypass cache and recalculate from source data
    """
    # Check dataset ownership
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )
    
    # Try to use cached volcano plot data first (unless force_recalculate is True)
    if not force_recalculate:
        metadata = dataset.dataset_metadata or {}
        volcano_plots = metadata.get("volcano_plots", {})
        
        if comparison_name in volcano_plots:
            cached_data = volcano_plots[comparison_name]
            
            # Convert from cached format to API response format
            points = []
            sig_count = 0
            
            for point_data in cached_data:
                is_sig = point_data.get('padj', 1.0) < 0.05 and abs(point_data.get('logFC', 0)) > 0.58
                if is_sig:
                    sig_count += 1
                    
                points.append({
                    "x": point_data.get("logFC", 0),
                    "y": point_data.get("negLogPadj", 0),
                    "gene": point_data.get("gene_id", "Unknown"),
                    "padj": point_data.get("padj", 1.0),
                    "is_significant": is_sig
                })
            
            return {
                "dataset_id": str(dataset_id),
                "comparison_name": comparison_name,
                "points": points,
                "total_genes": len(points),
                "significant_genes": sig_count,
                "downsampled": True,  # Pre-calculated data is always downsampled
                "cached": True
            }
    
    # If no cache or force_recalculate, compute from Parquet file
    # Check if we have parquet file path in metadata
    import math
    from pathlib import Path
    
    # For volcano plot, we MUST load from Parquet to get ALL genes (not just DEGs)
    if not dataset.parquet_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Parquet file required for volcano plot (contains all genes, not just DEGs)"
        )
    
    # Build full path using storage service
    from app.services.storage import LocalStorageService
    storage = LocalStorageService()
    full_parquet_path = storage.base_path / dataset.parquet_file_path
    
    try:
        import pyarrow.parquet as pq
        
        # Read parquet file using the full path
        table = pq.read_table(str(full_parquet_path))
        df = table.to_pandas()
        
        # Get column info from metadata
        metadata = dataset.dataset_metadata or {}
        comp_meta = metadata.get("comparisons", {}).get(comparison_name, {})
        
        logfc_col = comp_meta.get("logFC")
        padj_col = comp_meta.get("padj")
        
        if not logfc_col or not padj_col:
            # Try to auto-detect
            cols = df.columns.tolist()
            logfc_col = next((c for c in cols if "log2foldchange" in c.lower() or "logfc" in c.lower()), None)
            padj_col = next((c for c in cols if "padj" in c.lower() or "fdr" in c.lower()), None)
        
        if not logfc_col or not padj_col:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not find logFC or padj columns. Available: {df.columns.tolist()}"
            )
        
        gene_col = next((c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower() or c.lower() == "id"), None)
        
        # Filter out padj = 0 or NaN
        valid_mask = (df[padj_col] > 0) & (df[padj_col].notna()) & (df[logfc_col].notna())
        df_valid = df[valid_mask].copy()
        
        # Calculate -log10(padj)
        df_valid["y"] = -df_valid[padj_col].apply(lambda x: math.log10(x if x > 0 else 1e-300))
        df_valid["x"] = df_valid[logfc_col]
        df_valid["is_significant"] = (df_valid[padj_col] < 0.05) & (df_valid[logfc_col].abs() > 0.58)
        
        # Separate significant and non-significant
        sig_df = df_valid[df_valid["is_significant"]]
        non_sig_df = df_valid[~df_valid["is_significant"]]
        
        # Downsample non-significant genes
        remaining_slots = max_points - len(sig_df)
        if len(non_sig_df) > remaining_slots and remaining_slots > 0:
            # Sample evenly
            step = len(non_sig_df) / remaining_slots
            indices = [int(i * step) for i in range(remaining_slots)]
            non_sig_sampled = non_sig_df.iloc[indices]
        else:
            non_sig_sampled = non_sig_df
        
        # Combine
        import pandas as pd
        final_df = pd.concat([sig_df, non_sig_sampled])
        
        # Build response
        points = []
        for _, row in final_df.iterrows():
            points.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "gene": str(row[gene_col]) if gene_col else "Unknown",
                "padj": float(row[padj_col]),
                "is_significant": bool(row["is_significant"])
            })
        
        return {
            "dataset_id": str(dataset_id),
            "comparison_name": comparison_name,
            "points": points,
            "total_genes": len(df_valid),
            "significant_genes": len(sig_df),
            "downsampled": len(df_valid) > max_points,
            "cached": False
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load volcano plot data: {str(e)}"
        )


@router.get("/{dataset_id}/enrichment-pathways/{comparison_name}")
async def get_enrichment_pathways(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    regulation: Optional[str] = Query(None, description="Filter by regulation: ALL, UP, DOWN"),
    category: Optional[str] = Query(None, description="Filter by category: GO:BP, GO:MF, GO:CC, KEGG, etc."),
    padj_max: Optional[float] = Query(None, description="Maximum adjusted p-value"),
    min_gene_count: Optional[int] = Query(None, description="Minimum gene count"),
    search_term: Optional[str] = Query(None, description="Search in pathway name or description"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    sort_by: str = Query("padj", description="Sort by: padj, pvalue, gene_count, pathway_name"),
    sort_order: str = Query("asc", description="Sort order: asc or desc")
) -> dict:
    """
    Query enrichment pathways from database for a specific comparison.
    This is much faster than loading the entire Parquet file.

    Performance: <100ms (vs 2-5 seconds loading Parquet)
    """
    # Verify dataset exists and user has access
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Build query for EnrichmentPathway
    stmt = select(EnrichmentPathway).where(
        EnrichmentPathway.dataset_id == dataset_id,
        EnrichmentPathway.comparison_name == comparison_name
    )

    if regulation:
        stmt = stmt.where(EnrichmentPathway.regulation == regulation)

    if category:
        stmt = stmt.where(EnrichmentPathway.category == category)

    if padj_max is not None:
        stmt = stmt.where(EnrichmentPathway.padj <= padj_max)

    if min_gene_count is not None:
        stmt = stmt.where(EnrichmentPathway.gene_count >= min_gene_count)

    if search_term:
        search_pattern = f"%{search_term.lower()}%"
        stmt = stmt.where(or_(
            func.lower(EnrichmentPathway.pathway_name).like(search_pattern),
            func.lower(EnrichmentPathway.description).like(search_pattern)
        ))

    # Count total matching pathways
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total_count = total_result.scalar()

    # Sorting
    valid_sort_columns = {
        "padj": EnrichmentPathway.padj,
        "pvalue": EnrichmentPathway.pvalue,
        "gene_count": EnrichmentPathway.gene_count,
        "pathway_name": EnrichmentPathway.pathway_name
    }
    sort_col = valid_sort_columns.get(sort_by, EnrichmentPathway.padj)
    if sort_order.lower() == "desc":
        stmt = stmt.order_by(desc(sort_col))
    else:
        stmt = stmt.order_by(asc(sort_col))

    # Pagination
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    # Transform results
    pathways = []
    for row in rows:
        pathway_dict = {
            "pathway_id": row.pathway_id,
            "pathway_name": row.pathway_name,
            "gene_count": row.gene_count,
            "pvalue": row.pvalue,
            "padj": row.padj,
            "gene_ratio": row.gene_ratio,
            "bg_ratio": row.bg_ratio,
            "genes": row.genes,
            "category": row.category,
            "description": row.description,
            "regulation": row.regulation
        }
        pathways.append(pathway_dict)

    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0

    return {
        "dataset_id": str(dataset_id),
        "comparison_name": comparison_name,
        "pathways": pathways,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        "filters": {
            "category": category,
            "padj_max": padj_max,
            "min_gene_count": min_gene_count,
            "search_term": search_term
        },
        "sort": {
            "sort_by": sort_by,
            "sort_order": sort_order
        }
    }


@router.post("/{dataset_id}/enrichment-pathways/{comparison_name}/ai-select")
async def ai_select_enrichment_terms(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
    user_prompt: str = Body(..., embed=True, description="User instructions for term selection (e.g., 'use hepatocyte related terms')"),
    max_terms: int = Body(10, embed=True, ge=5, le=15, description="Maximum number of terms to select")
) -> dict:
    """
    Use AI to intelligently select enrichment terms based on user instructions.
    
    Requires PREMIUM or ADVANCED subscription.
    
    Args:
        user_prompt: Instructions for AI (e.g., "focus on liver metabolism", "immune response pathways")
        max_terms: Maximum number of terms to select (5-15)
        
    Returns:
        List of selected pathway IDs and terms
    """
    # Verify dataset exists and user has access
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Get top 50 enrichment pathways
    stmt = select(EnrichmentPathway).where(
        EnrichmentPathway.dataset_id == dataset_id,
        EnrichmentPathway.comparison_name == comparison_name
    ).order_by(asc(EnrichmentPathway.padj)).limit(50)

    pathway_result = await db.execute(stmt)
    pathways = pathway_result.scalars().all()

    if not pathways:
        raise HTTPException(status_code=404, detail="No enrichment data found for this comparison")

    # Prepare data for AI
    pathway_summaries = []
    for p in pathways:
        pathway_summaries.append({
            "id": p.pathway_id,
            "name": p.pathway_name,
            "description": p.description or "",
            "category": p.category,
            "p_value": p.pvalue,
            "gene_count": p.gene_count
        })

    # Build AI prompt
    ai_prompt = f"""You are a bioinformatics expert. The user has enrichment analysis results with {len(pathway_summaries)} significant pathways.

User's request: "{user_prompt}"

Available pathways:
{chr(10).join([f"- {p['name']} ({p['category']}, p={p['p_value']:.2e}, {p['gene_count']} genes): {p['description'][:100]}" for p in pathway_summaries])}

Task: Select exactly {max_terms} pathways that best match the user's request. Prioritize biological relevance to their query while maintaining statistical significance.

Return ONLY a JSON array of pathway IDs (the exact IDs from the list), nothing else. Example format:
["GO:0006096", "GO:0019319", "hsa00010"]"""

    # Call AI
    interpreter = LocalAIInterpreter()
    
    try:
        ai_response = await interpreter._call_ollama_raw(ai_prompt, max_tokens=500)
        
        # Parse AI response
        import json
        import re
        
        # Extract JSON array from response
        json_match = re.search(r'\[.*?\]', ai_response, re.DOTALL)
        if not json_match:
            # Fallback: return top N diverse pathways
            selected_ids = [p["id"] for p in pathway_summaries[:max_terms]]
        else:
            try:
                selected_ids = json.loads(json_match.group())
                # Validate that IDs exist
                valid_ids = [p["id"] for p in pathway_summaries]
                selected_ids = [id for id in selected_ids if id in valid_ids][:max_terms]
                
                # If AI didn't return enough, fill with top pathways
                if len(selected_ids) < max_terms:
                    remaining = [p["id"] for p in pathway_summaries if p["id"] not in selected_ids]
                    selected_ids.extend(remaining[:max_terms - len(selected_ids)])
            except json.JSONDecodeError:
                selected_ids = [p["id"] for p in pathway_summaries[:max_terms]]
        
        # Get full details of selected pathways
        selected_pathways = []
        for pathway_id in selected_ids:
            for p in pathways:
                if p.pathway_id == pathway_id:
                    selected_pathways.append({
                        "pathway_id": p.pathway_id,
                        "pathway_name": p.pathway_name,
                        "category": p.category,
                        "pvalue": p.pvalue,
                        "padj": p.padj,
                        "gene_count": p.gene_count,
                        "gene_ratio": p.gene_ratio,
                        "description": p.description
                    })
                    break
        
        return {
            "selected_terms": selected_pathways,
            "user_prompt": user_prompt,
            "total_available": len(pathway_summaries),
            "ai_model": interpreter.model
        }
        
    except Exception as e:
        logger.error(f"AI term selection failed: {e}")
        # Fallback: return top N diverse pathways
        categories_seen = set()
        selected_pathways = []
        
        for p in pathways:
            if len(selected_pathways) >= max_terms:
                break
            # Prioritize diversity
            if p.category not in categories_seen or len(categories_seen) >= 3:
                selected_pathways.append({
                    "pathway_id": p.pathway_id,
                    "pathway_name": p.pathway_name,
                    "category": p.category,
                    "pvalue": p.pvalue,
                    "padj": p.padj,
                    "gene_count": p.gene_count,
                    "gene_ratio": p.gene_ratio,
                    "description": p.description
                })
                categories_seen.add(p.category)
        
        return {
            "selected_terms": selected_pathways,
            "user_prompt": user_prompt,
            "total_available": len(pathway_summaries),
            "error": "AI selection failed, using fallback diverse selection"
        }


@router.post("/{dataset_id}/comparisons/{comparison_name}/interpret")
async def interpret_comparison(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
    quota_check: Annotated[User, Depends(check_ai_quota)],
    force_regenerate: bool = Query(False, description="Force regeneration even if cached"),
    language: str = Query("fr", description="Output language: fr or en")
) -> dict:
    """
    Generate AI interpretation of a comparison using local Ollama.
    
    **Privacy**: All processing is done locally - no data exported to external services.
    
    **Prerequisites**:
    1. Ollama installed and running (ollama serve)
    2. BioMistral model downloaded (ollama pull biomistral)
    
    The interpretation is cached in dataset metadata to avoid regeneration.
    Use force_regenerate=true to update the interpretation.
    
    Returns:
        {
            "interpretation": str,  # AI-generated text
            "cached": bool,
            "generated_at": str,
            "model": str,
            "comparison_name": str,
            "summary": {
                "deg_up": int,
                "deg_down": int,
                "top_pathways_count": int,
                "top_genes_count": int
            }
        }
    """
    # Verify dataset exists and user has access
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check if interpretation already exists in database
    existing_interpretation = await db.scalar(
        select(AIInterpretation)
        .where(AIInterpretation.dataset_id == dataset_id)
        .where(AIInterpretation.comparison_name == comparison_name)
    )
    
    if existing_interpretation and not force_regenerate:
        logger.info(f"Returning existing interpretation for {comparison_name}")
        return {
            "interpretation": existing_interpretation.interpretation,
            "cached": True,
            "generated_at": existing_interpretation.created_at.isoformat(),
            "model": existing_interpretation.model,
            "comparison_name": comparison_name,
            "summary": {
                "deg_up": existing_interpretation.deg_up,
                "deg_down": existing_interpretation.deg_down,
                "top_pathways_count": existing_interpretation.pathways_count,
                "top_genes_count": existing_interpretation.genes_count
            }
        }
    
    # If forcing regeneration, delete existing one
    if existing_interpretation and force_regenerate:
        logger.info(f"Deleting existing interpretation for {comparison_name} to regenerate")
        await db.delete(existing_interpretation)
        await db.commit()
    
    logger.info(f"Generating new interpretation for {comparison_name}")
    
    # Build context for AI
    try:
        # 1. Get DEG summary from database
        deg_up_query = select(func.count()).where(
            DegGene.dataset_id == dataset_id,
            DegGene.comparison_name == comparison_name,
            DegGene.regulation == "UP"
        )
        deg_down_query = select(func.count()).where(
            DegGene.dataset_id == dataset_id,
            DegGene.comparison_name == comparison_name,
            DegGene.regulation == "DOWN"
        )
        
        deg_up_result = await db.execute(deg_up_query)
        deg_down_result = await db.execute(deg_down_query)
        
        deg_up_count = deg_up_result.scalar() or 0
        deg_down_count = deg_down_result.scalar() or 0
        
        deg_summary = {
            "up_count": deg_up_count,
            "down_count": deg_down_count,
            "total": deg_up_count + deg_down_count
        }
        
        # 2. Get top enriched pathways (top 15 by padj)
        pathways_query = select(EnrichmentPathway).where(
            EnrichmentPathway.dataset_id == dataset_id,
            EnrichmentPathway.comparison_name == comparison_name
        ).order_by(EnrichmentPathway.padj.asc()).limit(15)
        
        pathways_result = await db.execute(pathways_query)
        pathways_rows = pathways_result.scalars().all()
        
        top_pathways = [
            {
                "pathway_name": p.pathway_name,
                "category": p.category,
                "padj": p.padj,
                "gene_count": p.gene_count,
                "genes": p.genes or []
            }
            for p in pathways_rows
        ]
        
        # 3. Get top DEGs (top 20 by absolute logFC)
        top_genes_query = select(DegGene).where(
            DegGene.dataset_id == dataset_id,
            DegGene.comparison_name == comparison_name
        ).order_by(
            func.abs(DegGene.log_fc).desc()
        ).limit(20)
        
        top_genes_result = await db.execute(top_genes_query)
        top_genes_rows = top_genes_result.scalars().all()
        
        top_genes = [
            {
                "gene_id": g.gene_id,
                "gene_name": g.gene_name or g.gene_id,
                "log_fc": g.log_fc,
                "padj": g.padj,
                "regulation": g.regulation
            }
            for g in top_genes_rows
        ]
        
        # Check if we have enough data
        if deg_summary["total"] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No DEG data found for comparison '{comparison_name}'. "
                       "The dataset may need reprocessing."
            )
        
        if len(top_pathways) == 0:
            logger.warning(f"No enrichment pathways found for {comparison_name}")
            # Continue anyway - can still interpret DEGs
        
        # 4. Call local AI interpreter
        interpreter = LocalAIInterpreter()
        
        # Check if Ollama is available
        availability = await interpreter.check_availability()
        if not availability["available"]:
            raise HTTPException(
                status_code=503,
                detail="Ollama n'est pas disponible. Vérifiez qu'Ollama est démarré (ollama serve) "
                       "et que le modèle est téléchargé (ollama pull biomistral)."
            )
        
        if not availability["model_available"]:
            raise HTTPException(
                status_code=503,
                detail=f"Le modèle '{interpreter.model}' n'est pas installé. "
                       f"Téléchargez-le avec: ollama pull {interpreter.model}"
            )
        
        interpretation = await interpreter.interpret_comparison(
            comparison_name=comparison_name,
            deg_summary=deg_summary,
            top_pathways=top_pathways,
            top_genes=top_genes,
            language=language
        )
        
        # Save to database with error handling for race conditions
        try:
            ai_interpretation = AIInterpretation(
                dataset_id=dataset_id,
                comparison_name=comparison_name,
                interpretation=interpretation,
                model=interpreter.model,
                deg_up=deg_up_count,
                deg_down=deg_down_count,
                pathways_count=len(top_pathways),
                genes_count=len(top_genes)
            )
            db.add(ai_interpretation)
            await db.commit()
            await db.refresh(ai_interpretation)
            logger.info(f"Saved interpretation for {comparison_name} to database")
        
        except Exception as e:
            # Check for unique violation (race condition)
            await db.rollback()
            if "duplicate key value violates unique constraint" in str(e) or "UniqueViolationError" in str(e):
                logger.warning(f"Race condition detected: Interpretation for {comparison_name} was created by another request. Fetching it.")
                existing = await db.scalar(
                    select(AIInterpretation)
                    .where(AIInterpretation.dataset_id == dataset_id)
                    .where(AIInterpretation.comparison_name == comparison_name)
                )
                if existing:
                    return {
                        "interpretation": existing.interpretation,
                        "cached": True,
                        "generated_at": existing.created_at.isoformat(),
                        "model": existing.model,
                        "comparison_name": comparison_name,
                        "summary": {
                            "deg_up": existing.deg_up,
                            "deg_down": existing.deg_down,
                            "top_pathways_count": existing.pathways_count,
                            "top_genes_count": existing.genes_count
                        }
                    }
            # Re-raise if it's not the unique violation or we couldn't recover
            raise e

        # Consommer le quota IA après génération
        await increment_ai_usage(
            user=user,
            db=db,
            action_type="interpretation",
            dataset_id=dataset_id,
            comparison_name=comparison_name,
            model_used=interpreter.model
        )

        return {
            "interpretation": interpretation,
            "cached": False,
            "generated_at": ai_interpretation.created_at.isoformat(),
            "model": interpreter.model,
            "comparison_name": comparison_name,
            "summary": {
                "deg_up": deg_up_count,
                "deg_down": deg_down_count,
                "top_pathways_count": len(top_pathways),
                "top_genes_count": len(top_genes)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate AI interpretation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de l'interprétation: {str(e)}"
        )


@router.post("/{dataset_id}/comparisons/{comparison_name}/ask")
async def ask_ai_question(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    user: Annotated[User, Depends(require_ai_access)],
    quota_check: Annotated[User, Depends(check_ai_quota)],
    question: str = Body(..., embed=True),
    context: str = Body("", embed=True)
) -> dict:
    """
    Ask a specific question to the AI about a comparison analysis.
    Saves the conversation to database for persistence.
    
    Args:
        dataset_id: UUID of the DEG dataset
        comparison_name: Name of the comparison
        question: User's question
        context: Optional context (previous interpretation)
    
    Returns:
        {
            "answer": str,
            "question": str,
            "model": str,
            "conversation_id": UUID,
            "created_at": datetime
        }
    """
    # Verify dataset exists and user has access
    query = select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = dataset.project
    is_owner = project.owner_id == current_user.user_id

    if not is_owner:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    try:
        interpreter = LocalAIInterpreter()
        
        # Build prompt with context
        prompt = f"""You are an expert in bioinformatics and transcriptomic analysis.

User question: {question}

"""
        
        if context:
            prompt += f"""Context from previous analysis:
{context[:1000]}  

"""

        # Get basic comparison data for context
        deg_up = await db.scalar(
            select(func.count(DegGene.id))
            .where(DegGene.dataset_id == dataset_id)
            .where(DegGene.comparison_name == comparison_name)
            .where(DegGene.regulation == "UP")
        )
        
        deg_down = await db.scalar(
            select(func.count(DegGene.id))
            .where(DegGene.dataset_id == dataset_id)
            .where(DegGene.comparison_name == comparison_name)
            .where(DegGene.regulation == "DOWN")
        )
        
        # Get top genes
        top_genes_query = (
            select(DegGene)
            .where(DegGene.dataset_id == dataset_id)
            .where(DegGene.comparison_name == comparison_name)
            .order_by(func.abs(DegGene.log_fc).desc())
            .limit(10)
        )
        top_genes_result = await db.execute(top_genes_query)
        top_genes = top_genes_result.scalars().all()
        
        prompt += f"""Comparison data for '{comparison_name}':
- {deg_up} upregulated genes
- {deg_down} downregulated genes
- Top 10 genes (by |log2FC|): {', '.join([g.gene_name or g.gene_id for g in top_genes[:10]])}

Answer concisely and precisely in English (maximum 200 words).

IMPORTANT: Write in plain text only. Do NOT use Markdown formatting (no # for headings, no ** for bold, no * for italic). Use simple line breaks to separate ideas."""

        # Generate answer
        answer = await interpreter.generate_simple_answer(prompt)
        
        # Save conversation to database
        conversation = AIConversation(
            dataset_id=dataset_id,
            comparison_name=comparison_name,
            question=question,
            answer=answer,
            model=interpreter.model,
            user_id=current_user.user_id
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)
        
        # Consommer le quota IA après génération
        await increment_ai_usage(
            user=user,
            db=db,
            action_type="question",
            dataset_id=dataset_id,
            comparison_name=comparison_name,
            model_used=interpreter.model
        )

        return {
            "answer": answer,
            "question": question,
            "model": interpreter.model,
            "conversation_id": conversation.id,
            "created_at": conversation.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error answering AI question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la réponse: {str(e)}"
        )


@router.get("/{dataset_id}/comparisons/{comparison_name}/conversations")
async def get_conversation_history(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Get conversation history for a specific comparison.
    
    Returns:
        {
            "conversations": [
                {
                    "id": UUID,
                    "question": str,
                    "answer": str,
                    "model": str,
                    "created_at": datetime
                }
            ]
        }
    """
    # Verify dataset exists and user has access
    query = select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = dataset.project
    is_owner = project.owner_id == current_user.user_id

    if not is_owner:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    # Get all conversations for this comparison
    conversations_query = (
        select(AIConversation)
        .where(AIConversation.dataset_id == dataset_id)
        .where(AIConversation.comparison_name == comparison_name)
        .order_by(AIConversation.created_at.asc())
    )
    conversations_result = await db.execute(conversations_query)
    conversations = conversations_result.scalars().all()
    
    return {
        "conversations": [
            {
                "id": str(conv.id),
                "question": conv.question,
                "answer": conv.answer,
                "model": conv.model,
                "created_at": conv.created_at.isoformat()
            }
            for conv in conversations
        ]
    }


@router.post("/{dataset_id}/comparisons/{comparison_name}/visualizations/pca")
async def calculate_custom_pca(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    body: dict = Body(...)
) -> dict:
    """
    Calculate PCA with custom gene selection and coloring options.
    Returns PCA coordinates for each sample with optional metadata for coloring.
    
    Request body:
    {
        "gene_list": ["GENE1", "GENE2", ...],  // Optional
        "n_components": 2,  // 2 or 3
        "color_by": "condition"  // Optional
    }
    """
    # Extract parameters from body
    gene_list = body.get("gene_list")
    n_components = body.get("n_components", 2)
    color_by = body.get("color_by")
    
    # Verify access
    query = select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = dataset.project
    is_owner = project.owner_id == current_user.user_id

    if not is_owner:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    # Load parquet file
    parquet_path = Path(settings.LOCAL_STORAGE_PATH) / dataset.parquet_file_path
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    # Filter by gene list if provided
    if gene_list and len(gene_list) > 0:
        gene_col = next((c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()), None)
        if gene_col:
            df = df[df[gene_col].isin(gene_list)]
            if df.empty:
                raise HTTPException(status_code=400, detail="No matching genes found")

    # Get comparison metadata to identify sample columns
    metadata = dataset.dataset_metadata or {}
    comp_meta = metadata.get("comparisons", {}).get(comparison_name, {})
    
    # Extract expression columns (numeric columns that are not stats columns)
    stat_columns = {"log2FoldChange", "logFC", "pvalue", "padj", "baseMean", "lfcSE", "stat"}
    gene_col = next((c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()), None)
    
    expression_cols = [
        c for c in df.columns 
        if df[c].dtype in ['int64', 'float64'] 
        and c not in stat_columns 
        and c != gene_col
    ]

    if not expression_cols:
        raise HTTPException(
            status_code=400,
            detail="No expression columns found for PCA calculation"
        )

    # Create expression matrix (genes x samples)
    expr_df = df[expression_cols].fillna(0)
    
    # Transpose for PCA (samples x genes)
    X = expr_df.T.values
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Build response
    results = []
    for i, sample_name in enumerate(expression_cols):
        point = {
            "sample": sample_name,
            "pc1": float(principal_components[i, 0]),
            "pc2": float(principal_components[i, 1]) if n_components > 1 else 0,
            "pc3": float(principal_components[i, 2]) if n_components > 2 else 0
        }
        
        # Add color metadata if requested
        if color_by:
            # Try to extract from sample name or metadata
            point["group"] = sample_name.split("_")[0] if "_" in sample_name else sample_name
        
        results.append(point)
    
    return {
        "data": results,
        "explained_variance": [float(x) for x in pca.explained_variance_ratio_],
        "total_variance": float(sum(pca.explained_variance_ratio_)),
        "n_genes_used": len(df),
        "n_samples": len(expression_cols)
    }


@router.post("/{dataset_id}/comparisons/{comparison_name}/visualizations/umap")
async def calculate_custom_umap(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    body: dict = Body(...)
) -> dict:
    """
    Calculate UMAP with custom gene selection and parameters.
    
    Request body:
    {
        "gene_list": ["GENE1", "GENE2", ...],  // Optional
        "n_components": 2,  // 2 or 3
        "n_neighbors": 15,  // 2-200
        "min_dist": 0.1,  // 0.0-0.99
        "color_by": "condition"  // Optional
    }
    """
    # Extract parameters from body
    gene_list = body.get("gene_list")
    n_components = body.get("n_components", 2)
    n_neighbors = body.get("n_neighbors", 15)
    min_dist = body.get("min_dist", 0.1)
    color_by = body.get("color_by")
    
    # Verify access
    query = select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = dataset.project
    is_owner = project.owner_id == current_user.user_id

    if not is_owner:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    # Load parquet file
    parquet_path = Path(settings.LOCAL_STORAGE_PATH) / dataset.parquet_file_path
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    # Filter by gene list if provided
    if gene_list and len(gene_list) > 0:
        gene_col = next((c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()), None)
        if gene_col:
            df = df[df[gene_col].isin(gene_list)]
            if df.empty:
                raise HTTPException(status_code=400, detail="No matching genes found")

    # Get expression columns
    stat_columns = {"log2FoldChange", "logFC", "pvalue", "padj", "baseMean", "lfcSE", "stat"}
    gene_col = next((c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()), None)
    
    expression_cols = [
        c for c in df.columns 
        if df[c].dtype in ['int64', 'float64'] 
        and c not in stat_columns 
        and c != gene_col
    ]

    if not expression_cols:
        raise HTTPException(
            status_code=400,
            detail="No expression columns found for UMAP calculation"
        )

    # Create expression matrix
    expr_df = df[expression_cols].fillna(0)
    X = expr_df.T.values
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run UMAP
    try:
        from umap import UMAP
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="UMAP not installed. Please install umap-learn package."
        )
    
    umap = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    embedding = umap.fit_transform(X_scaled)
    
    # Build response
    results = []
    for i, sample_name in enumerate(expression_cols):
        point = {
            "sample": sample_name,
            "umap1": float(embedding[i, 0]),
            "umap2": float(embedding[i, 1]) if n_components > 1 else 0,
            "umap3": float(embedding[i, 2]) if n_components > 2 else 0
        }
        
        if color_by:
            point["group"] = sample_name.split("_")[0] if "_" in sample_name else sample_name
        
        results.append(point)
    
    return {
        "data": results,
        "n_genes_used": len(df),
        "n_samples": len(expression_cols),
        "parameters": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist
        }
    }


@router.post("/{dataset_id}/comparisons/{comparison_name}/visualizations/boxplot")
async def get_custom_boxplot_data(
    dataset_id: UUID,
    comparison_name: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    body: dict = Body(...)
) -> dict:
    """
    Get expression data for custom box plots.
    Returns expression values for each gene across all samples.
    
    Request body should contain:
    {
        "gene_list": ["GENE1", "GENE2", ...]
    }
    """
    # Extract gene_list from body
    gene_list = body.get("gene_list", [])
    
    # Verify access
    query = select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = dataset.project
    is_owner = project.owner_id == current_user.user_id

    if not is_owner:
        raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    if not gene_list or len(gene_list) == 0:
        raise HTTPException(status_code=400, detail="At least one gene must be provided")

    # Load parquet file
    parquet_path = Path(settings.LOCAL_STORAGE_PATH) / dataset.parquet_file_path
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    import pyarrow.parquet as pq
    table = pq.read_table(str(parquet_path))
    df = table.to_pandas()

    # Find gene column
    gene_col = next((c for c in df.columns if "gene" in c.lower() or "symbol" in c.lower()), None)
    if not gene_col:
        raise HTTPException(status_code=400, detail="No gene column found in dataset")

    # Filter by genes
    df_genes = df[df[gene_col].isin(gene_list)]
    
    if df_genes.empty:
        raise HTTPException(status_code=404, detail="No matching genes found")

    # Get expression columns
    stat_columns = {"log2FoldChange", "logFC", "pvalue", "padj", "baseMean", "lfcSE", "stat"}
    expression_cols = [
        c for c in df.columns 
        if df[c].dtype in ['int64', 'float64'] 
        and c not in stat_columns 
        and c != gene_col
    ]

    if not expression_cols:
        raise HTTPException(
            status_code=400,
            detail="No expression columns found"
        )

    # Build response - one entry per gene
    gene_data = []
    for _, row in df_genes.iterrows():
        gene_name = row[gene_col]
        
        # Extract expression values across samples
        expression_values = []
        for col in expression_cols:
            value = row[col]
            if pd.notna(value):
                expression_values.append({
                    "sample": col,
                    "value": float(value),
                    "group": col.split("_")[0] if "_" in col else col  # Extract condition from sample name
                })
        
        gene_data.append({
            "gene": gene_name,
            "values": expression_values
        })

    return {
        "genes": gene_data,
        "n_genes": len(df_genes),
        "n_samples": len(expression_cols)
    }


@router.get("/ai/status")
async def check_ai_status(
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
) -> dict:
    """
    Check if local AI (Ollama) is available and which models are installed.
    
    Returns:
        {
            "available": bool,
            "models": List[str],
            "recommended_model": str,
            "current_model": str,
            "version": str
        }
    """
    interpreter = LocalAIInterpreter()
    status_info = await interpreter.check_availability()
    
    return {
        **status_info,
        "recommended_model": "biomistral",
        "install_instructions": {
            "macos": "brew install ollama && ollama serve && ollama pull biomistral",
            "linux": "curl -fsSL https://ollama.ai/install.sh | sh && ollama serve && ollama pull biomistral",
            "windows": "Download from https://ollama.ai/download"
        }
    }


@router.post("/{dataset_id}/venn-analysis")
async def venn_analysis(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    comparisons: list[str] = Body(..., description="List of comparison names to analyze"),
    padj_threshold: float = Body(0.05, description="Adjusted p-value threshold"),
    logfc_threshold: float = Body(0.58, description="Log fold change threshold")
) -> dict:
    """
    Perform Venn diagram analysis to find common and unique genes between comparisons.

    Supports 2-5 comparisons:
    - 2-3 comparisons: Returns data for Venn diagram
    - 4-5 comparisons: Returns data for UpSet plot

    Returns all intersections with their gene lists for export.
    """
    # Validate number of comparisons
    if len(comparisons) < 2:
        raise HTTPException(status_code=400, detail="At least 2 comparisons required")
    if len(comparisons) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 comparisons allowed")

    # Verify dataset exists and user has access
    query = select(Dataset).options(joinedload(Dataset.project)).where(Dataset.id == dataset_id)
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    project = dataset.project
    is_owner = project.owner_id == current_user.user_id

    if not is_owner:
        # Check if user is a project member
        member_query = select(ProjectMember).where(
            ProjectMember.project_id == project.id,
            ProjectMember.user_id == current_user.user_id
        )
        member_result = await db.execute(member_query)
        is_member = member_result.scalar_one_or_none() is not None

        if not is_member:
            raise HTTPException(status_code=403, detail="Access denied")

    # Fetch DEG genes for each comparison from database
    comparison_genes: dict[str, set[str]] = {}

    for comp_name in comparisons:
        # Query deg_genes table for this comparison
        stmt = select(DegGene.gene_id).where(
            DegGene.dataset_id == dataset_id,
            DegGene.comparison_name == comp_name,
            DegGene.padj <= padj_threshold,
            or_(DegGene.log_fc >= logfc_threshold, DegGene.log_fc <= -logfc_threshold)
        )

        result = await db.execute(stmt)
        genes = {row for row in result.scalars().all()}
        comparison_genes[comp_name] = genes

        print(f"[Venn] {comp_name}: {len(genes)} genes")

    # Calculate all possible intersections
    from itertools import combinations

    intersections = []

    # For each possible combination of sets (from 1 to n comparisons)
    for r in range(1, len(comparisons) + 1):
        for combo in combinations(comparisons, r):
            combo_list = list(combo)

            # Find genes in this intersection
            if len(combo_list) == 1:
                # Genes unique to this comparison
                genes_in_combo = comparison_genes[combo_list[0]].copy()
                # Remove genes that appear in any other comparison
                for other_comp in comparisons:
                    if other_comp not in combo_list:
                        genes_in_combo -= comparison_genes[other_comp]
            else:
                # Genes common to all comparisons in combo
                genes_in_combo = comparison_genes[combo_list[0]].copy()
                for comp in combo_list[1:]:
                    genes_in_combo &= comparison_genes[comp]

                # Remove genes that appear in comparisons NOT in this combo
                # (to get exclusive intersection)
                other_comps = set(comparisons) - set(combo_list)
                for other_comp in other_comps:
                    genes_in_combo -= comparison_genes[other_comp]

            if genes_in_combo:  # Only add non-empty intersections
                intersections.append({
                    "sets": combo_list,
                    "size": len(genes_in_combo),
                    "genes": sorted(list(genes_in_combo))
                })

    return {
        "dataset_id": str(dataset_id),
        "comparisons": comparisons,
        "sets": comparisons,
        "total_genes": {comp: len(genes) for comp, genes in comparison_genes.items()},
        "intersections": intersections,
        "thresholds": {
            "padj": padj_threshold,
            "logfc": logfc_threshold
        }
    }


@router.post("/{dataset_id}/advanced-filter")
async def apply_advanced_filter(
    dataset_id: UUID,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)],
    filter_data: dict = Body(...),
    comparison_name: str = Body(...),
    page: int = Body(1),
    page_size: int = Body(50)
) -> dict:
    """
    Apply advanced filters with AND/OR logic to DEG genes.

    Filter structure:
    {
        "groups": [
            {
                "operator": "AND" | "OR",
                "conditions": [
                    {
                        "field": "logFC" | "padj" | "gene_id" | "regulation" | "gene_name",
                        "operator": ">" | "<" | ">=" | "<=" | "=" | "!=" | "contains" | "not_contains" | "in_list",
                        "value": any
                    }
                ]
            }
        ],
        "groupOperator": "AND" | "OR"
    }
    """
    # Check ownership
    query = select(Dataset).join(Project).where(
        Dataset.id == dataset_id,
        Project.owner_id == current_user.user_id
    )
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found"
        )

    try:
        groups = filter_data.get("groups", [])
        group_operator = filter_data.get("groupOperator", "AND")

        if not groups:
            return {
                "genes": [],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0
                }
            }

        # Base query
        stmt = select(DegGene).where(
            DegGene.dataset_id == dataset_id,
            DegGene.comparison_name == comparison_name
        )

        # Build filter conditions
        group_conditions = []
        
        for group in groups:
            conditions = group.get("conditions", [])
            condition_operator = group.get("operator", "AND")
            
            if not conditions:
                continue
                
            current_group_exprs = []
            
            for condition in conditions:
                field = condition.get("field")
                operator = condition.get("operator")
                value = condition.get("value")
                
                # Map field to model column
                column = None
                if field == "logFC":
                    column = DegGene.log_fc
                elif field == "padj":
                    column = DegGene.padj
                elif field == "gene_id":
                    column = DegGene.gene_id
                elif field == "gene_name":
                    column = DegGene.gene_name
                elif field == "regulation":
                    column = DegGene.regulation
                
                if column is None:
                    continue
                    
                # Build expression
                expr = None
                if operator == ">":
                    expr = column > value
                elif operator == "<":
                    expr = column < value
                elif operator == ">=":
                    expr = column >= value
                elif operator == "<=":
                    expr = column <= value
                elif operator == "=":
                    expr = column == value
                elif operator == "!=":
                    expr = column != value
                elif operator == "contains":
                    expr = column.ilike(f"%{value}%")
                elif operator == "not_contains":
                    expr = ~column.ilike(f"%{value}%")
                elif operator == "in_list":
                    if isinstance(value, str):
                        gene_list = [g.strip() for g in value.replace(',', '\n').split('\n') if g.strip()]
                        if gene_list:
                            expr = column.in_(gene_list)
                
                if expr is not None:
                    current_group_exprs.append(expr)
            
            if current_group_exprs:
                if condition_operator == "AND":
                    group_conditions.append(and_(*current_group_exprs))
                else:
                    group_conditions.append(or_(*current_group_exprs))
        
        if group_conditions:
            if group_operator == "AND":
                stmt = stmt.where(and_(*group_conditions))
            else:
                stmt = stmt.where(or_(*group_conditions))
        else:
            # No valid conditions found
             return {
                "genes": [],
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0
                }
            }

        # Count query
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar() or 0

        # Data query with pagination
        stmt = stmt.order_by(asc(DegGene.padj), desc(func.abs(DegGene.log_fc)))
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(stmt)
        rows = result.scalars().all()

        genes = [
            {
                "gene_id": row.gene_id,
                "log_fc": row.log_fc,
                "padj": row.padj,
                "regulation": row.regulation,
                "gene_name": row.gene_name
            }
            for row in rows
        ]

        total_pages = (total + page_size - 1) // page_size if total > 0 else 0

        return {
            "genes": genes,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages
            },
            "filter_summary": {
                "groups": len(groups),
                "total_conditions": sum(len(g.get("conditions", [])) for g in groups),
                "group_operator": group_operator
            }
        }

    except Exception as e:
        import traceback
        print(f"Error in advanced filter: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Filter error: {str(e)}")


@router.post("/{dataset_id}/gsea")
async def run_gsea_analysis(
    dataset_id: UUID,
    comparison_name: str = Body(...),
    gene_set_database: str = Body("GO_BP"),  # GO_BP, GO_MF, GO_CC, KEGG, REACTOME, HALLMARK
    ranking_metric: str = Body("signed_pvalue"),  # log_fc, signed_pvalue, signal2noise
    min_size: int = Body(15),
    max_size: int = Body(500),
    n_permutations: int = Body(1000),
    fdr_threshold: float = Body(0.25),
    db: Annotated[AsyncSession, Depends(get_db)] = None
) -> dict:
    """
    Run Gene Set Enrichment Analysis (GSEA) on a comparison

    Args:
        dataset_id: Dataset UUID
        comparison_name: Comparison name
        gene_set_database: Gene set database to use
        ranking_metric: Method to rank genes (log_fc, signed_pvalue, signal2noise)
        min_size: Minimum gene set size
        max_size: Maximum gene set size
        n_permutations: Number of permutations for null distribution
        fdr_threshold: FDR q-value threshold for significance

    Returns:
        GSEA results with enrichment scores, p-values, and visualizations
    """
    import pandas as pd
    from sqlalchemy import text

    try:
        # Fetch all DEG genes for this comparison
        query = text("""
            SELECT gene_id, log_fc, padj, gene_name
            FROM deg_genes
            WHERE dataset_id = :dataset_id
            AND comparison_name = :comparison_name
            ORDER BY padj ASC
        """)

        result = await db.execute(query, {
            "dataset_id": str(dataset_id),
            "comparison_name": comparison_name
        })
        rows = result.fetchall()

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No DEG data found for comparison: {comparison_name}"
            )

        # Convert to DataFrame
        deg_data = pd.DataFrame(rows, columns=["gene_id", "log_fc", "padj", "gene_name"])

        # Prepare ranked gene list
        ranked_genes = prepare_ranked_gene_list(deg_data, ranking_metric=ranking_metric)

        # Load gene sets from database
        logger.info(f"Loading gene sets from database: {gene_set_database}")
        gene_set_loader = GeneSetLoader(db)

        try:
            # Convert string database name to enum
            database_enum = GeneSetDatabase(gene_set_database)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid gene set database: {gene_set_database}. "
                       f"Valid options: {[e.value for e in GeneSetDatabase]}"
            )

        # Retrieve gene sets from database
        gene_sets_db = await gene_set_loader.get_gene_sets(
            database=database_enum,
            organism="Homo sapiens",
            min_size=min_size,
            max_size=max_size
        )

        # Convert to dictionary format expected by GSEA processor
        if not gene_sets_db:
            # Fallback to placeholder if no gene sets in database
            logger.warning(f"No gene sets found for {gene_set_database}, using placeholders")
            gene_sets = GeneSetsLoader.get_default_gene_sets()
        else:
            gene_sets = {gs.name: gs.genes for gs in gene_sets_db}
            logger.info(f"Loaded {len(gene_sets)} gene sets from database")

        # Initialize GSEA processor
        gsea_processor = GSEAProcessor(
            min_size=min_size,
            max_size=max_size,
            power=1.0
        )

        # Run GSEA
        logger.info(f"Running GSEA with {len(ranked_genes)} genes and {len(gene_sets)} gene sets")
        results = gsea_processor.run_gsea(
            ranked_genes=ranked_genes,
            gene_sets=gene_sets,
            metric_column="metric",
            n_permutations=n_permutations
        )

        # Filter by FDR threshold
        significant_results = [r for r in results if r.fdr_q_value <= fdr_threshold]

        # Convert to dict for JSON response
        results_dict = [r.to_dict() for r in significant_results]

        # Calculate summary statistics
        n_enriched_positive = len([r for r in significant_results if r.normalized_enrichment_score > 0])
        n_enriched_negative = len([r for r in significant_results if r.normalized_enrichment_score < 0])

        return {
            "dataset_id": str(dataset_id),
            "comparison_name": comparison_name,
            "parameters": {
                "gene_set_database": gene_set_database,
                "ranking_metric": ranking_metric,
                "min_size": min_size,
                "max_size": max_size,
                "n_permutations": n_permutations,
                "fdr_threshold": fdr_threshold
            },
            "summary": {
                "total_genes": len(ranked_genes),
                "total_gene_sets_tested": len(gene_sets),
                "significant_gene_sets": len(significant_results),
                "enriched_in_phenotype_pos": n_enriched_positive,
                "enriched_in_phenotype_neg": n_enriched_negative
            },
            "results": results_dict
        }

    except Exception as e:
        import traceback
        logger.error(f"GSEA error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"GSEA analysis failed: {str(e)}")


@router.get("/{dataset_id}/gsea/{gene_set_name}/enrichment-plot")
async def get_enrichment_plot_data(
    dataset_id: UUID,
    gene_set_name: str,
    comparison_name: str = Query(...),
    ranking_metric: str = Query("signed_pvalue"),
    db: Annotated[AsyncSession, Depends(get_db)] = None
) -> dict:
    """
    Get data for GSEA enrichment plot for a specific gene set

    Returns running enrichment score and gene positions for visualization
    """
    import pandas as pd
    from sqlalchemy import text

    try:
        # Fetch DEG data
        query = text("""
            SELECT gene_id, log_fc, padj
            FROM deg_genes
            WHERE dataset_id = :dataset_id
            AND comparison_name = :comparison_name
            ORDER BY padj ASC
        """)

        result = await db.execute(query, {
            "dataset_id": str(dataset_id),
            "comparison_name": comparison_name
        })
        rows = result.fetchall()

        deg_data = pd.DataFrame(rows, columns=["gene_id", "log_fc", "padj"])
        ranked_genes = prepare_ranked_gene_list(deg_data, ranking_metric=ranking_metric)

        # Load gene set from database by name
        # Try to find the gene set across all databases
        gene_set_loader = GeneSetLoader(db)
        gene_set_obj = None

        # Try each database until we find the gene set
        for database in GeneSetDatabase:
            gene_set_obj = await gene_set_loader.get_gene_set_by_name(
                name=gene_set_name,
                database=database,
                organism="Homo sapiens"
            )
            if gene_set_obj:
                break

        if not gene_set_obj:
            # Fallback to placeholder gene sets
            gene_sets = GeneSetsLoader.get_default_gene_sets()
            if gene_set_name not in gene_sets:
                raise HTTPException(status_code=404, detail=f"Gene set not found: {gene_set_name}")
            gene_set = gene_sets[gene_set_name]
        else:
            gene_set = gene_set_obj.genes

        # Calculate enrichment score
        gsea_processor = GSEAProcessor()
        gene_list = ranked_genes.index.tolist()
        metrics = ranked_genes["metric"].values

        es, running_es, positions = gsea_processor._calculate_enrichment_score(
            gene_list, gene_set, metrics
        )

        # Prepare data for plotting
        return {
            "gene_set_name": gene_set_name,
            "enrichment_score": es,
            "running_enrichment_scores": running_es.tolist(),
            "gene_positions": positions,
            "ranked_genes": gene_list,
            "metrics": metrics.tolist(),
            "gene_set_size": len(gene_set)
        }

    except Exception as e:
        import traceback
        logger.error(f"Enrichment plot error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to generate plot data: {str(e)}")


from pydantic import BaseModel
import numpy as np
import hashlib
import json

from typing import List

class ClusteringRequest(BaseModel):
    """Parameters for hierarchical clustering."""
    top_n_genes: int = 2000
    gene_ids: Optional[List[str]] = None
    cluster_rows: bool = True
    cluster_cols: bool = True
    method: str = "ward" # ward, average, complete, single
    metric: str = "euclidean" # euclidean, correlation, cosine

@router.post("/{dataset_id}/cluster")
async def cluster_dataset(
    dataset_id: UUID,
    params: ClusteringRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[SupabaseUser, Depends(get_current_user)]
):
    """
    Perform hierarchical clustering on a dataset (matrix type).
    """
    # 1. Check permissions and get dataset
    query = select(Dataset).join(Project).filter(Dataset.id == dataset_id)
    
    result = await db.execute(query)
    dataset = result.scalar_one_or_none()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")
        
    # Check cache
    try:
        # Create a stable cache key based on params
        cache_key_dict = params.model_dump()
        cache_key_str = json.dumps(cache_key_dict, sort_keys=True)
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
        cache_path = f"projects/{dataset.project_id}/clustering/{dataset_id}_{cache_hash}.json"
        
        # Check if cache exists
        try:
            cached_data = await storage_service.download_file(cache_path)
            logger.info(f"Returning cached clustering result for {dataset_id}")
            return json.loads(cached_data)
        except:
            # Not found in cache
            pass
            
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")

    # Check if user is owner of project
    project_result = await db.execute(select(Project).where(Project.id == dataset.project_id))
    project = project_result.scalar_one()
    
    if str(project.owner_id) != str(current_user.user_id):
         # Check membership
         member_query = select(ProjectMember).where(
             ProjectMember.project_id == project.id,
             ProjectMember.user_id == current_user.user_id
         )
         member = (await db.execute(member_query)).scalar_one_or_none()
         if not member:
             raise HTTPException(status_code=403, detail="Not authorized to access this dataset")

    # 2. Check if dataset is MATRIX type
    if dataset.type != DatasetType.MATRIX:
        raise HTTPException(status_code=400, detail="Clustering only supported for count/expression matrices")

    if dataset.status != DatasetStatus.READY:
        raise HTTPException(status_code=400, detail="Dataset is not ready for analysis")

    # 3. Fetch data from Storage
    parquet_path = dataset.parquet_file_path or dataset.raw_file_path
    if not parquet_path:
         raise HTTPException(status_code=500, detail="Dataset file path missing")

    try:
        file_data = await storage_service.download_file(parquet_path)
        df = await data_processor.get_dataframe(file_data)
        
        # Determine gene ID column if present and set as index
        if "gene_id" in df.columns:
            df = df.set_index("gene_id")
        elif "Unnamed: 0" in df.columns:
             df = df.set_index("Unnamed: 0")
        
        # Ensure all data is numeric for clustering
        # Filter out non-numeric columns just in case
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
             raise HTTPException(status_code=400, detail="No numeric data found in dataset for clustering")
        
        # 4. Perform Clustering
        try:
            result = clustering_service.perform_clustering(
                numeric_df,
                top_n_genes=params.top_n_genes,
                gene_ids=params.gene_ids,
                cluster_rows=params.cluster_rows,
                cluster_cols=params.cluster_cols,
                method=params.method,
                metric=params.metric
            )

            # Save to cache
            try:
                cache_key_dict = params.model_dump()
                cache_key_str = json.dumps(cache_key_dict, sort_keys=True)
                cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()
                cache_path = f"projects/{dataset.project_id}/clustering/{dataset_id}_{cache_hash}.json"
                
                await storage_service.upload_file(
                    cache_path,
                    json.dumps(result).encode('utf-8'),
                    content_type="application/json"
                )
            except Exception as e:
                logger.warning(f"Failed to save clustering cache: {e}")

            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error accessing dataset data: {str(e)}")
