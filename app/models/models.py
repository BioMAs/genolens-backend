"""
SQLAlchemy models for GenoLens Next.
Following the asset-based architecture - only metadata stored in PostgreSQL.
"""
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import String, ForeignKey, JSON, Enum as SQLEnum, Text, Index, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.models.base import Base, TimestampMixin


class UserRole(str, enum.Enum):
    """User roles in the system."""
    ADMIN = "ADMIN"  # Full access: manage users, projects, subscriptions, upload analyses
    USER = "USER"  # Standard user with subscription-based access


class SubscriptionPlan(str, enum.Enum):
    """Subscription plans with different feature access."""
    BASIC = "BASIC"  # Access own projects for visualization and sharing only
    PREMIUM = "PREMIUM"  # + AI interpretation (15 free, then purchase tokens)
    ADVANCED = "ADVANCED"  # + Launch analyses + unlimited AI (coming soon)


class DatasetType(str, enum.Enum):
    """Types of datasets that can be stored."""
    MATRIX = "MATRIX"  # Count/Expression matrices
    DEG = "DEG"  # Differential Expression Gene results
    ENRICHMENT = "ENRICHMENT"  # Pathway/GO enrichment results
    METADATA = "METADATA"  # Generic metadata (deprecated)
    METADATA_SAMPLE = "METADATA_SAMPLE"  # Sample description
    METADATA_CONTRAST = "METADATA_CONTRAST"  # Contrast metadata


class DatasetStatus(str, enum.Enum):
    """Processing status of datasets."""
    PENDING = "PENDING"  # Upload initiated
    PROCESSING = "PROCESSING"  # Being converted to Parquet
    READY = "READY"  # Ready for queries
    FAILED = "FAILED"  # Processing failed
    ARCHIVED = "ARCHIVED"  # Soft deleted


class ProjectMember(Base, TimestampMixin):
    """
    ProjectMember: Junction table for sharing projects between users.
    Enables collaboration and role-based access to projects.
    user_id references Supabase Auth user UUID (no FK to local table).
    """
    __tablename__ = "project_members"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    # Supabase Auth user ID - no foreign key as users are in Supabase
    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="Supabase Auth user UUID"
    )
    access_level: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole, name="user_role_enum"),
        nullable=False,
        default=UserRole.USER,
        comment="Role of this user for this specific project"
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="members")

    # Indexes and constraints
    __table_args__ = (
        Index("ix_project_members_project_user", "project_id", "user_id", unique=True),
        Index("ix_project_members_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<ProjectMember(project_id={self.project_id}, user_id={self.user_id}, access={self.access_level})>"


class Project(Base, TimestampMixin):
    """
    Project: Top-level container for organizing related analyses.
    A project groups samples and datasets together.
    owner_id references Supabase Auth user UUID (no FK to local table).
    """
    __tablename__ = "projects"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Supabase Auth user ID - no foreign key as users are in Supabase
    owner_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="Supabase Auth user UUID of project owner"
    )

    # Relationships
    members: Mapped[list["ProjectMember"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )
    samples: Mapped[list["Sample"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )
    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_projects_owner_created", "owner_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name}, owner_id={self.owner_id})>"


class Sample(Base, TimestampMixin):
    """
    Sample: Represents a biological sample within a project.
    Stores metadata about the sample (condition, timepoint, replicate, etc.)
    """
    __tablename__ = "samples"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # Flexible metadata storage (condition, tissue, treatment, etc.)
    sample_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="samples")

    # Indexes
    __table_args__ = (
        Index("ix_samples_project_name", "project_id", "name"),
    )

    def __repr__(self) -> str:
        return f"<Sample(id={self.id}, name={self.name}, project_id={self.project_id})>"


class Dataset(Base, TimestampMixin):
    """
    Dataset: References a data file stored in Supabase Storage.
    This is the core of the asset-based model - the actual data is in Parquet files,
    only metadata and file paths are stored in PostgreSQL.
    """
    __tablename__ = "datasets"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Dataset type (MATRIX, DEG, ENRICHMENT, etc.)
    type: Mapped[DatasetType] = mapped_column(
        SQLEnum(DatasetType, name="dataset_type"),
        nullable=False,
        index=True
    )

    # Processing status
    status: Mapped[DatasetStatus] = mapped_column(
        SQLEnum(DatasetStatus, name="dataset_status"),
        nullable=False,
        default=DatasetStatus.PENDING,
        index=True
    )

    # File storage paths
    raw_file_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="Path to original uploaded file in Supabase Storage"
    )
    parquet_file_path: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
        comment="Path to processed Parquet file in Supabase Storage"
    )

    # Column mapping: Maps user's column names to standard schema
    # Example: {"gene_id": "Gene", "sample_A": "Sample_A_TPM"}
    column_mapping: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Mapping of standard column names to actual file columns"
    )

    # Additional metadata about the dataset
    dataset_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional dataset metadata (row count, file size, etc.)"
    )

    # Error information if processing failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="datasets")

    # Indexes
    __table_args__ = (
        Index("ix_datasets_project_type", "project_id", "type"),
        Index("ix_datasets_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name={self.name}, type={self.type}, status={self.status})>"


class GeneSetDatabase(str, enum.Enum):
    """Gene set databases available for GSEA."""
    GO_BP = "GO_BP"  # Gene Ontology Biological Process
    GO_MF = "GO_MF"  # Gene Ontology Molecular Function
    GO_CC = "GO_CC"  # Gene Ontology Cellular Component
    KEGG = "KEGG"  # KEGG Pathways
    REACTOME = "REACTOME"  # Reactome Pathways
    HALLMARK = "HALLMARK"  # MSigDB Hallmark gene sets
    C2_CURATED = "C2_CURATED"  # MSigDB C2 curated gene sets
    C5_ONTOLOGY = "C5_ONTOLOGY"  # MSigDB C5 ontology gene sets
    C6_ONCOGENIC = "C6_ONCOGENIC"  # MSigDB C6 oncogenic signatures
    C7_IMMUNOLOGIC = "C7_IMMUNOLOGIC"  # MSigDB C7 immunologic signatures
    CUSTOM = "CUSTOM"  # User-defined custom gene sets


class GeneSet(Base, TimestampMixin):
    """
    GeneSet: Stores gene sets for enrichment analysis (GSEA, ORA).
    Gene sets are collections of genes grouped by biological function, pathway, etc.
    Each gene set belongs to a database (GO, KEGG, MSigDB, etc.)
    """
    __tablename__ = "gene_sets"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    # Gene set identification
    name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="Gene set name (e.g., GO:0006955, HALLMARK_TNFA_SIGNALING_VIA_NFKB)"
    )

    database: Mapped[GeneSetDatabase] = mapped_column(
        SQLEnum(GeneSetDatabase, name="gene_set_database"),
        nullable=False,
        index=True,
        comment="Source database"
    )

    # Gene set description
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description of the gene set"
    )

    # Genes in this set (stored as array for efficient queries)
    genes: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        comment="List of gene symbols in this gene set"
    )

    # Gene set size (denormalized for quick filtering)
    size: Mapped[int] = mapped_column(
        nullable=False,
        index=True,
        comment="Number of genes in the set"
    )

    # Additional metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    gene_set_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata (GO term hierarchy, pathway URL, etc.)"
    )

    # Version tracking for gene set updates
    version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Version of the gene set database (e.g., GO 2024-01, MSigDB 2024.1)"
    )

    # Species/organism
    organism: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="Homo sapiens",
        index=True,
        comment="Organism for this gene set"
    )

    # Indexes for efficient queries
    __table_args__ = (
        Index("ix_gene_sets_database_organism", "database", "organism"),
        Index("ix_gene_sets_name_database", "name", "database", unique=True),
        Index("ix_gene_sets_size", "size"),
    )

    def __repr__(self) -> str:
        return f"<GeneSet(name={self.name}, database={self.database}, size={self.size})>"


class DegGene(Base, TimestampMixin):
    """
    DegGene: Stores Differential Expression Analysis results at the gene level.
    Used for fast querying and filtering without loading large Parquet files.
    """
    __tablename__ = "deg_genes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    comparison_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    gene_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    gene_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    base_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    log_fc: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    pvalue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    padj: Mapped[Optional[float]] = mapped_column(Float, nullable=True, index=True)
    
    regulation: Mapped[str] = mapped_column(String(50), nullable=True, index=True)  # UP, DOWN, NS

    # Relationships
    dataset: Mapped["Dataset"] = relationship()

    # Indexes
    __table_args__ = (
        Index("ix_deg_genes_dataset_comparison", "dataset_id", "comparison_name"),
        Index("ix_deg_genes_dataset_comparison_regulation", "dataset_id", "comparison_name", "regulation"),
    )

    def __repr__(self) -> str:
        return f"<DegGene(dataset_id={self.dataset_id}, gene={self.gene_id}, logFC={self.log_fc})>"


class EnrichmentPathway(Base, TimestampMixin):
    """
    EnrichmentPathway: Stores Pathway Enrichment Analysis results.
    Used for fast querying and filtering.
    """
    __tablename__ = "enrichment_pathways"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    comparison_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    pathway_id: Mapped[str] = mapped_column(String(255), nullable=False)
    pathway_name: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)  # GO:BP, KEGG, etc.
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    gene_count: Mapped[int] = mapped_column(Integer, nullable=False)
    pvalue: Mapped[float] = mapped_column(Float, nullable=False)
    padj: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    
    gene_ratio: Mapped[Optional[str]] = mapped_column(String(50), nullable=True) # stored as string "k/n" usually
    bg_ratio: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)   # stored as string "M/N" usually
    
    genes: Mapped[list] = mapped_column(JSON, nullable=True) # List of gene IDs/Symbols
    regulation: Mapped[str] = mapped_column(String(10), nullable=False, default="ALL", index=True) # ALL, UP, DOWN

    # Relationships
    dataset: Mapped["Dataset"] = relationship()

    # Indexes
    __table_args__ = (
        Index("ix_enrichment_pathways_dataset_comparison", "dataset_id", "comparison_name"),
    )

    def __repr__(self) -> str:
        return f"<EnrichmentPathway(dataset_id={self.dataset_id}, pathway={self.pathway_name}, padj={self.padj})>"


class AIConversation(Base, TimestampMixin):
    """
    AIConversation: Stores Q&A conversations with AI for each comparison.
    Enables persistence of chat history across page refreshes.
    """
    __tablename__ = "ai_conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    comparison_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # User who asked the question (Supabase Auth user UUID)
    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="Supabase Auth user UUID"
    )

    # Relationships
    dataset: Mapped["Dataset"] = relationship()

    # Indexes
    __table_args__ = (
        Index("ix_ai_conversations_dataset_comparison", "dataset_id", "comparison_name"),
        Index("ix_ai_conversations_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<AIConversation(dataset_id={self.dataset_id}, comparison={self.comparison_name}, user={self.user_id})>"


class AIInterpretation(Base, TimestampMixin):
    """
    AIInterpretation: Stores the main AI interpretation for each comparison.
    One interpretation per comparison - cannot be regenerated.
    """
    __tablename__ = "ai_interpretations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    comparison_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    
    interpretation: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Summary statistics at the time of generation
    deg_up: Mapped[int] = mapped_column(Integer, nullable=False)
    deg_down: Mapped[int] = mapped_column(Integer, nullable=False)
    pathways_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    genes_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationships
    dataset: Mapped["Dataset"] = relationship()

    # Indexes - ensure one interpretation per comparison
    __table_args__ = (
        Index("ix_ai_interpretations_dataset_comparison", "dataset_id", "comparison_name", unique=True),
    )

    def __repr__(self) -> str:
        return f"<AIInterpretation(dataset_id={self.dataset_id}, comparison={self.comparison_name})>"


class User(Base, TimestampMixin):
    """
    User: Extended user profile linked to Supabase Auth.
    Stores subscription info and quotas.
    """
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, comment="Supabase Auth user UUID")
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Role and subscription
    role: Mapped[UserRole] = mapped_column(
        SQLEnum(UserRole, name="user_role_enum"),
        nullable=False,
        default=UserRole.USER
    )
    subscription_plan: Mapped[SubscriptionPlan] = mapped_column(
        SQLEnum(SubscriptionPlan, name="subscription_plan_enum"),
        nullable=False,
        default=SubscriptionPlan.BASIC
    )
    
    # AI interpretation quotas (for PREMIUM users)
    ai_interpretations_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_tokens_purchased: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    # Subscription management
    subscription_starts_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    subscription_ends_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    
    # Stripe integration (for future)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role}, plan={self.subscription_plan})>"
    
    @property
    def can_use_ai(self) -> bool:
        """Check if user can use AI features."""
        return self.role == UserRole.ADMIN or self.subscription_plan in [SubscriptionPlan.PREMIUM, SubscriptionPlan.ADVANCED]
    
    @property
    def ai_interpretations_remaining(self) -> int:
        """Calculate remaining AI interpretations for PREMIUM users."""
        if self.subscription_plan != SubscriptionPlan.PREMIUM:
            return -1  # Unlimited or not applicable
        
        free_limit = 15
        purchased = self.ai_tokens_purchased - self.ai_tokens_used
        return max(0, (free_limit - self.ai_interpretations_used) + purchased)
    
    @property
    def can_launch_analyses(self) -> bool:
        """Check if user can launch new analyses."""
        return self.role == UserRole.ADMIN or self.subscription_plan == SubscriptionPlan.ADVANCED


class AIUsageLog(Base, TimestampMixin):
    """
    AIUsageLog: Track AI usage for billing and analytics.
    """
    __tablename__ = "ai_usage_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"),
        nullable=True
    )
    
    action_type: Mapped[str] = mapped_column(
        String(50), 
        nullable=False,
        comment="interpretation or question"
    )
    comparison_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    was_free: Mapped[bool] = mapped_column(
        nullable=False, 
        default=True,
        comment="True if within free quota, False if purchased token"
    )

    # Relationships
    user: Mapped["User"] = relationship()

    __table_args__ = (
        Index("ix_ai_usage_logs_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AIUsageLog(user_id={self.user_id}, action={self.action_type}, tokens={self.tokens_used})>"

