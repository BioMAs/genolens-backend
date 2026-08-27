"""
SQLAlchemy models for GenoLens Next.
Following the asset-based architecture - only metadata stored in PostgreSQL.
"""
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime
from sqlalchemy import String, ForeignKey, JSON, Enum as SQLEnum, Text, Index, Float, Integer, DateTime
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import text as sa_text
import enum

from app.models.base import Base, TimestampMixin


class UserRole(str, enum.Enum):
    """User roles in the system."""
    ADMIN = "ADMIN"  # Full access: manage users, projects, subscriptions, upload analyses
    SCILICIUM_ADMIN = "SCILICIUM_ADMIN"  # Scilicium internal admin (same access as ADMIN + license management)
    USER = "USER"  # Standard user with subscription-based access
    VIEWER = "VIEWER"  # Read-only access
    ANALYST = "ANALYST"  # Can run analyses but not manage users/billing
    SUBSCRIBER = "SUBSCRIBER"  # Subscription-based access (legacy/external auth)


class UserStatus(str, enum.Enum):
    """User account status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"


class SubscriptionPlan(str, enum.Enum):
    """Subscription plans with different feature access."""
    STARTER = "STARTER"     # 3 users, 15 projects, 30 comparisons/month
    TEAM = "TEAM"           # 10 users, unlimited projects, 150 comparisons/month + AI
    ON_PREMISE = "ON_PREMISE"  # Self-hosted, unlimited everything


class SelfServiceAnalysisStatus(str, enum.Enum):
    """Status of a self-service analysis job."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class OmicsDataType(str, enum.Enum):
    """Type of omics data for a self-service analysis."""
    TRANSCRIPTOMICS = "transcriptomics"
    PROTEOMICS = "proteomics"
    LIPIDOMICS = "lipidomics"


class ClaimDirection(str, enum.Enum):
    """Direction of a pathway signature that supports a cosmetic claim.

    Mirrors the `Updated_direction` column of the SciLicium claim referential.
    """
    UP = "UP"            # increased pathway signature supports the claim
    DOWN = "DOWN"        # decreased pathway signature supports the claim
    BOTH = "BOTH"        # direction depends on context
    UNKNOWN = "UNKNOWN"  # insufficient basis
    AVOID = "AVOID"      # not appropriate for consumer-benefit inference (pathology/oncogenic)


class EvidenceLevel(str, enum.Enum):
    """Strength of the literature evidence backing a claim mapping."""
    HIGH = "HIGH"        # broad consensus / hallmark link
    MODERATE = "MODERATE"  # plausible / common but context-dependent
    LOW = "LOW"          # weak or indirect evidence


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
    
    # Pre-calculated statistics for DEG datasets (performance optimization)
    deg_up_count: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Pre-calculated count of up-regulated genes (padj < 0.05, logFC > 0)"
    )
    deg_down_count: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Pre-calculated count of down-regulated genes (padj < 0.05, logFC < 0)"
    )
    deg_significant_count: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Pre-calculated count of significant genes (padj < 0.05)"
    )
    total_genes: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Total number of genes in dataset"
    )

    # Error information if processing failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="datasets")

    # Indexes
    __table_args__ = (
        Index("ix_datasets_project_type", "project_id", "type"),
        Index("ix_datasets_status", "status"),
        Index("ix_datasets_metadata_gin", "dataset_metadata", postgresql_using="gin"),
        Index("ix_datasets_deg_counts", "type", "deg_significant_count"),
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

    # Scoping for user-defined (CUSTOM) gene sets. NULL for built-in databases
    # (GO/KEGG/... are global). Custom sets are scoped to a project (and owner).
    project_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Owning project for custom gene sets (NULL = built-in/global)"
    )
    user_id: Mapped[Optional[UUID]] = mapped_column(
        nullable=True,
        comment="Creator of a custom gene set (NULL = built-in)"
    )

    # Indexes for efficient queries. Uniqueness is split so built-in sets stay
    # globally unique on (name, database) while custom sets are unique per project
    # (a nullable project_id would otherwise make NULL rows non-comparable).
    __table_args__ = (
        Index("ix_gene_sets_database_organism", "database", "organism"),
        Index("ix_gene_sets_size", "size"),
        Index(
            "ix_gene_sets_name_db_builtin", "name", "database",
            unique=True, postgresql_where=sa_text("project_id IS NULL"),
        ),
        Index(
            "ix_gene_sets_name_db_project", "name", "database", "project_id",
            unique=True, postgresql_where=sa_text("project_id IS NOT NULL"),
        ),
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


class AgentSession(Base, TimestampMixin):
    """
    AgentSession: a multi-turn conversation with the agentic "chat mode".
    Bound to an explicitly selected context (project + DEG dataset + comparison).
    Distinct from the legacy per-comparison AIConversation Q&A history.
    """
    __tablename__ = "agent_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        nullable=False, index=True, comment="Supabase Auth user UUID"
    )
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    comparison_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AgentMessage.sequence",
    )

    __table_args__ = (
        Index("ix_agent_sessions_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<AgentSession(id={self.id}, dataset_id={self.dataset_id}, comparison={self.comparison_name})>"


class AgentMessage(Base, TimestampMixin):
    """
    AgentMessage: one turn in an AgentSession.

    A single assistant turn can carry narrative text, the tool calls it made, the
    compact tool results fed to the model, and the full figure payloads used to
    re-render plots on reload.
    """
    __tablename__ = "agent_messages"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # user | assistant
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    tool_calls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    tool_results: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    figures: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    session: Mapped["AgentSession"] = relationship(back_populates="messages")

    __table_args__ = (
        Index("ix_agent_messages_session_sequence", "session_id", "sequence"),
    )

    def __repr__(self) -> str:
        return f"<AgentMessage(session_id={self.session_id}, role={self.role}, seq={self.sequence})>"


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
        default=SubscriptionPlan.STARTER
    )

    # AI interpretation quotas (for TEAM users)
    ai_interpretations_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_tokens_purchased: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ai_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Subscription management
    subscription_starts_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    subscription_ends_at: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    status: Mapped[UserStatus] = mapped_column(
        SQLEnum(UserStatus, name="user_status_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=UserStatus.ACTIVE,
        index=True,
    )

    @property
    def is_active(self) -> bool:
        """Backward-compat property — True when status is ACTIVE."""
        return self.status == UserStatus.ACTIVE

    # Stripe integration (for future)
    stripe_customer_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, unique=True)
    stripe_subscription_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Comparison quota (resets monthly)
    comparisons_used_this_month: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Counter reset on the 1st of each month"
    )
    quota_reset_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Timestamp of last monthly quota reset"
    )

    # Optional add-on modules unlocked individually by an admin (independent of plan)
    cosmetics_module_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=sa_text("false"),
        comment="Whether the Cosmetics (skin-claims) module is unlocked for this user",
    )
    report_customization_module_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=sa_text("false"),
        comment="Whether the report customization module (branding, logo, conclusion, M&M) is unlocked for this user",
    )
    scientific_module_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=sa_text("false"),
        comment="Whether the Scientific tools module (GSEA, contrast scatter, signature scoring, custom gene sets, DEG patterns) is unlocked for this user",
    )
    drug_discovery_module_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default=sa_text("false"),
        comment="Whether the Drug Discovery module (target ranking, indications, reports) is unlocked for this user",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role}, plan={self.subscription_plan})>"

    @property
    def can_use_ai(self) -> bool:
        """TEAM and ON_PREMISE only. ADMIN/SCILICIUM_ADMIN always yes."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.subscription_plan in (
            SubscriptionPlan.TEAM, SubscriptionPlan.ON_PREMISE
        )

    @property
    def can_launch_analyses(self) -> bool:
        """All paid plans can launch analyses (subject to comparison quota)."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.subscription_plan in (
            SubscriptionPlan.STARTER, SubscriptionPlan.TEAM, SubscriptionPlan.ON_PREMISE
        )

    @property
    def can_use_multi_comparison(self) -> bool:
        """Multi-comparison reserved for TEAM and ON_PREMISE."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.subscription_plan in (
            SubscriptionPlan.TEAM, SubscriptionPlan.ON_PREMISE
        )

    @property
    def can_export_advanced(self) -> bool:
        """PDF/Excel export reserved for TEAM and ON_PREMISE."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.subscription_plan in (
            SubscriptionPlan.TEAM, SubscriptionPlan.ON_PREMISE
        )

    @property
    def comparisons_quota(self) -> Optional[int]:
        """Monthly comparison quota. None = unlimited."""
        if self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
            return None
        return {
            SubscriptionPlan.STARTER: 30,
            SubscriptionPlan.TEAM: 150,
            SubscriptionPlan.ON_PREMISE: None,
        }.get(self.subscription_plan, 30)

    @property
    def comparisons_remaining(self) -> Optional[int]:
        """Remaining comparisons this month. None = unlimited."""
        quota = self.comparisons_quota
        if quota is None:
            return None
        return max(0, quota - self.comparisons_used_this_month)

    @property
    def max_projects(self) -> Optional[int]:
        """Max number of projects. None = unlimited."""
        if self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
            return None
        return {
            SubscriptionPlan.STARTER: 15,
            SubscriptionPlan.TEAM: None,
            SubscriptionPlan.ON_PREMISE: None,
        }.get(self.subscription_plan, 15)

    @property
    def max_datasets_per_project(self) -> Optional[int]:
        """Max datasets per project. None = unlimited."""
        if self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
            return None
        return {
            SubscriptionPlan.STARTER: 5,
            SubscriptionPlan.TEAM: None,
            SubscriptionPlan.ON_PREMISE: None,
        }.get(self.subscription_plan, 5)

    @property
    def ai_interpretations_remaining(self) -> Optional[int]:
        """TEAM/ON_PREMISE: None (AI access governed at endpoint). STARTER: -1 (no access)."""
        if self.subscription_plan in (SubscriptionPlan.TEAM, SubscriptionPlan.ON_PREMISE) or self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN):
            return None
        return -1  # STARTER: no access

    @property
    def has_cosmetics_module(self) -> bool:
        """Cosmetics add-on: explicitly unlocked per-user by an admin. Admins always have it."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.cosmetics_module_enabled

    @property
    def has_report_customization(self) -> bool:
        """Report customization add-on: explicitly unlocked per-user by an admin. Admins always have it."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.report_customization_module_enabled

    @property
    def has_scientific_module(self) -> bool:
        """Scientific tools add-on: explicitly unlocked per-user by an admin. Admins always have it."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.scientific_module_enabled

    @property
    def has_drug_discovery_module(self) -> bool:
        """Drug Discovery add-on: explicitly unlocked per-user by an admin (independent of
        plan). Admins always have it."""
        return self.role in (UserRole.ADMIN, UserRole.SCILICIUM_ADMIN) or self.drug_discovery_module_enabled


class UserReportSettings(Base, TimestampMixin):
    """
    Persistent per-user report branding settings (report customization module).

    Logo and colours are reused across every report the user generates; the
    default M&M / conclusion text pre-fill the per-report editor. Gated behind
    User.has_report_customization.
    """
    __tablename__ = "user_report_settings"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # Branding (override the hardcoded SciLicium defaults in the LaTeX templates)
    logo_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    institute_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    institute_address: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    primary_color: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    secondary_color: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # Defaults that pre-fill the per-report editor
    default_materials_methods: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    default_conclusion: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Page-layout defaults: first_page_type in {detailed, simple, cover};
    # last_page_type in {color, contact}.
    first_page_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="detailed", server_default="detailed"
    )
    last_page_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="color", server_default="color"
    )
    # Default cover/project information (project name, client, sponsor, test
    # facility/site, signatures, contact) — see schemas.report.CoverInfo.
    cover_info: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


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


class ModalMonthlyCost(Base, TimestampMixin):
    """
    Modal LLM spend for a given calendar month, entered by an admin.

    Drives per-user AI cost estimation: rate = spend_eur / total_tokens(month),
    cost_user = tokens_user(month) * rate. One row per (year, month).
    """
    __tablename__ = "modal_monthly_costs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False, comment="1-12")
    spend_eur: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ux_modal_monthly_costs_year_month", "year", "month", unique=True),
    )

    def __repr__(self) -> str:
        return f"<ModalMonthlyCost({self.year}-{self.month:02d}: {self.spend_eur}€)>"


class CachedComputation(Base, TimestampMixin):
    """
    CachedComputation: Persistent cache for expensive computations.
    
    Stores results of clustering, volcano plots, statistics, etc. to avoid
    re-computing on every request. Provides database-backed caching with
    expiration and hit tracking for monitoring.
    """
    __tablename__ = "cached_computations"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    computation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type: clustering, volcano, stats, pca, etc."
    )
    
    params_hash: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="MD5 hash of computation parameters for cache key"
    )
    
    result_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Cached computation result as JSON"
    )
    
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Optional expiration timestamp (NULL = never expires)"
    )
    
    hit_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of times this cache entry was accessed"
    )
    
    last_accessed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        comment="Last time this cache entry was accessed"
    )
    
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    
    # Indexes
    __table_args__ = (
        # Composite unique index for fast cache lookups
        Index(
            "ix_cached_computations_lookup",
            "dataset_id", "computation_type", "params_hash",
            unique=True
        ),
        # Index for cleaning up expired entries
        Index(
            "ix_cached_computations_expires",
            "expires_at",
            postgresql_where=sa_text("expires_at IS NOT NULL")
        ),
        # GIN index on result_data for potential JSONB queries
        Index(
            "ix_cached_computations_result_gin",
            "result_data",
            postgresql_using="gin"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<CachedComputation(dataset_id={self.dataset_id}, type={self.computation_type}, hits={self.hit_count})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        if self.expires_at is None:
            return False
        from datetime import datetime, timezone
        return datetime.now(timezone.utc) > self.expires_at


class SampleCorrelation(Base, TimestampMixin):
    """
    SampleCorrelation: Stores pre-computed sample-to-sample correlations and distances.
    Used for instant heatmap rendering without recalculating clustering each time.
    """
    __tablename__ = "sample_correlations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Sample identifiers
    sample_a: Mapped[str] = mapped_column(String(255), nullable=False)
    sample_b: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Computed values
    correlation: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Correlation coefficient between samples"
    )
    distance: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Distance metric value between samples"
    )
    
    # Computation parameters (used for cache invalidation)
    method: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Clustering method: ward, average, complete, single, kmeans"
    )
    metric: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Distance metric: euclidean, manhattan, correlation, cosine"
    )
    top_n_genes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=2000,
        comment="Number of genes used for computation"
    )
    
    # Relationships
    dataset: Mapped["Dataset"] = relationship()
    
    # Indexes for fast lookups
    __table_args__ = (
        Index(
            "ix_sample_correlations_dataset_method_metric",
            "dataset_id", "method", "metric", "top_n_genes"
        ),
        Index(
            "ix_sample_correlations_samples",
            "dataset_id", "sample_a", "sample_b"
        ),
    )
    
    def __repr__(self) -> str:
        return f"<SampleCorrelation(dataset={self.dataset_id}, {self.sample_a}<->{self.sample_b}, method={self.method})>"


class GOTerm(Base, TimestampMixin):
    """
    GOTerm: Stores Gene Ontology terms with their hierarchical relationships.
    Enables GO enrichment analysis and visualization of term hierarchies.
    """
    __tablename__ = "go_terms"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    
    # GO term identification
    go_id: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        unique=True,
        index=True,
        comment="GO identifier (e.g., GO:0006955)"
    )
    
    # Term information
    name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="GO term name"
    )
    
    namespace: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="GO namespace: biological_process, molecular_function, cellular_component"
    )
    
    definition: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Textual definition of the GO term"
    )
    
    # Hierarchical relationships (stored as JSON arrays)
    is_a: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of parent GO IDs (is_a relationships)"
    )
    
    part_of: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of parent GO IDs (part_of relationships)"
    )
    
    regulates: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of GO IDs this term regulates"
    )
    
    # Additional metadata
    synonyms: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Alternative names for this term"
    )
    
    is_obsolete: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        index=True,
        comment="Whether this term is obsolete"
    )
    
    replaced_by: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="GO ID that replaces this obsolete term"
    )
    
    # Denormalized for performance
    level: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="Depth level in GO hierarchy (0=root)"
    )
    
    gene_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        index=True,
        comment="Number of genes annotated to this term (including descendants)"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_go_terms_namespace_level", "namespace", "level"),
        Index("ix_go_terms_name_search", "name", postgresql_using="gin", postgresql_ops={"name": "gin_trgm_ops"}),
    )
    
    def __repr__(self) -> str:
        return f"<GOTerm(go_id={self.go_id}, name={self.name}, namespace={self.namespace})>"


class GOAnnotation(Base, TimestampMixin):
    """
    GOAnnotation: Links genes to GO terms with evidence codes.
    Supports filtering by evidence type and enables GO enrichment analysis.
    """
    __tablename__ = "go_annotations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    
    # Gene identification
    gene_symbol: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Gene symbol (e.g., TP53)"
    )
    
    gene_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Entrez Gene ID or Ensembl ID"
    )
    
    # GO term
    go_id: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="GO identifier"
    )
    
    # Evidence
    evidence_code: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        index=True,
        comment="Evidence code (IEA, IDA, IMP, etc.)"
    )
    
    # Annotation source and quality
    source_db: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="UniProt",
        comment="Source database (UniProt, MGI, etc.)"
    )
    
    qualifier: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Qualifier (NOT, contributes_to, etc.)"
    )
    
    # Organism
    organism: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="Homo sapiens",
        index=True,
        comment="Organism"
    )
    
    # Additional metadata
    annotation_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional annotation details (references, date, etc.)"
    )
    
    # Relationships
    go_term: Mapped["GOTerm"] = relationship(
        foreign_keys=[go_id],
        primaryjoin="GOAnnotation.go_id == foreign(GOTerm.go_id)",
        viewonly=True
    )
    
    # Indexes for fast queries
    __table_args__ = (
        Index("ix_go_annotations_gene_organism", "gene_symbol", "organism"),
        Index("ix_go_annotations_go_organism", "go_id", "organism"),
        Index("ix_go_annotations_gene_go", "gene_symbol", "go_id", unique=True),
        Index("ix_go_annotations_evidence", "evidence_code"),
    )
    
    def __repr__(self) -> str:
        return f"<GOAnnotation(gene={self.gene_symbol}, go_id={self.go_id}, evidence={self.evidence_code})>"


class GeneBookmark(Base, TimestampMixin):
    """
    GeneBookmark: User's bookmarked/favorited genes with notes and tags.
    Allows users to mark genes of interest within a project context.
    """
    __tablename__ = "gene_bookmarks"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    
    # User and project context
    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="User who created bookmark (Supabase Auth ID)"
    )
    
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Project context for bookmark"
    )
    
    # Gene identification
    gene_symbol: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Gene symbol (e.g., TP53)"
    )
    
    gene_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Gene ID (Entrez, Ensembl)"
    )
    
    # User annotations
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="User notes about this gene"
    )
    
    tags: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Custom tags for categorization"
    )
    
    color: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Custom color for visualization (hex)"
    )
    
    is_favorite: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        index=True,
        comment="Quick favorite flag"
    )
    
    extra_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata"
    )
    
    # Relationships
    project: Mapped["Project"] = relationship()
    
    # Indexes
    __table_args__ = (
        Index("ix_gene_bookmarks_user_project", "user_id", "project_id"),
        Index("ix_gene_bookmarks_project_gene", "project_id", "gene_symbol", unique=True),
    )
    
    def __repr__(self) -> str:
        return f"<GeneBookmark(gene={self.gene_symbol}, project={self.project_id}, user={self.user_id})>"


class GeneList(Base, TimestampMixin):
    """
    GeneList: Custom lists of genes created by users.
    Allows grouping genes thematically (e.g., "Cancer markers", "Immune genes").
    """
    __tablename__ = "gene_lists"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    
    # List identification
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="List name"
    )
    
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="List description"
    )
    
    # Ownership
    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="Owner user ID (Supabase Auth)"
    )
    
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Associated project"
    )
    
    # Gene data
    genes: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Array of gene symbols"
    )
    
    gene_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        index=True,
        comment="Number of genes in list"
    )
    
    # Display properties
    color: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Display color (hex)"
    )
    
    is_public: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        index=True,
        comment="Visible to project members"
    )
    
    tags: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Tags for categorization"
    )
    
    extra_data: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata (source, date, etc.)"
    )
    
    # Relationships
    project: Mapped["Project"] = relationship()
    
    # Indexes
    __table_args__ = (
        Index("ix_gene_lists_user_project", "user_id", "project_id"),
    )
    
    def __repr__(self) -> str:
        return f"<GeneList(name={self.name}, genes={self.gene_count}, project={self.project_id})>"


class CommentType(str, enum.Enum):
    """Types of comments in the system."""
    GENERAL = "GENERAL"  # General project comment
    GENE = "GENE"  # Comment on a specific gene
    COMPARISON = "COMPARISON"  # Comment on a comparison
    PATHWAY = "PATHWAY"  # Comment on a pathway


class ProjectComment(Base, TimestampMixin):
    """
    ProjectComment: Comments and annotations on projects, genes, comparisons, etc.
    Enables collaboration and knowledge sharing.
    user_id references Supabase Auth user UUID (no FK to local table).
    """
    __tablename__ = "project_comments"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Project this comment belongs to"
    )
    # Supabase Auth user ID - no foreign key as users are in Supabase
    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="Supabase Auth user UUID who created the comment"
    )
    comment_type: Mapped[CommentType] = mapped_column(
        SQLEnum(CommentType, name="comment_type_enum"),
        nullable=False,
        default=CommentType.GENERAL,
        comment="Type of comment (general, gene, comparison, pathway)"
    )
    target_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="ID of the target entity (gene_symbol, comparison_name, pathway_id)"
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Comment content in markdown format"
    )
    parent_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("project_comments.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Parent comment ID for threaded discussions"
    )
    is_resolved: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        comment="Whether this comment thread is resolved"
    )
    extra_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional metadata (mentions, tags, etc.)"
    )

    # Relationships
    project: Mapped["Project"] = relationship()
    parent: Mapped[Optional["ProjectComment"]] = relationship(
        remote_side=[id],
        back_populates="replies"
    )
    replies: Mapped[list["ProjectComment"]] = relationship(
        back_populates="parent",
        cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("ix_project_comments_project_type", "project_id", "comment_type"),
        Index("ix_project_comments_target", "project_id", "target_id"),
        Index("ix_project_comments_user", "user_id"),
        Index("ix_project_comments_parent", "parent_id"),
    )

    def __repr__(self) -> str:
        return f"<ProjectComment(project={self.project_id}, type={self.comment_type}, user={self.user_id})>"


# ============================================================================
# Project Activity Log
# ============================================================================

class ActivityEventType(str, enum.Enum):
    """Types of activity events tracked in the audit log."""
    DATASET_UPLOADED = "dataset_uploaded"
    DATASET_DELETED = "dataset_deleted"
    COMPARISON_CREATED = "comparison_created"
    ENRICHMENT_RUN = "enrichment_run"
    CLUSTERING_RUN = "clustering_run"
    GSEA_RUN = "gsea_run"
    GO_ENRICHMENT_RUN = "go_enrichment_run"
    BOOKMARK_CREATED = "bookmark_created"
    BOOKMARK_BATCH_CREATED = "bookmark_batch_created"
    BOOKMARK_DELETED = "bookmark_deleted"
    GENE_LIST_CREATED = "gene_list_created"
    COMMENT_ADDED = "comment_added"
    PROJECT_SHARED = "project_shared"


class ProjectActivityLog(Base):
    """
    ProjectActivityLog: Immutable audit trail of actions performed within a project.
    Records who did what and when, enabling a complete history/timeline view.
    user_id references Supabase Auth user UUID (no FK to local table).
    """
    __tablename__ = "project_activity_log"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Project this event belongs to"
    )

    user_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
        comment="Supabase Auth user UUID who triggered the event"
    )

    event_type: Mapped[ActivityEventType] = mapped_column(
        SQLEnum(ActivityEventType, name="activity_event_type_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True,
        comment="Type of activity event"
    )

    entity_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Type of affected entity (dataset, comparison, bookmark, etc.)"
    )

    entity_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="ID of the affected entity"
    )

    entity_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Human-readable name of the affected entity"
    )

    extra_metadata: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional context for the event (gene count, method, etc.)"
    )

    # No updated_at — activity log entries are immutable
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="When the event occurred"
    )

    # Relationships
    project: Mapped["Project"] = relationship()

    # Indexes
    __table_args__ = (
        Index("ix_activity_log_project_created", "project_id", "created_at"),
        Index("ix_activity_log_project_event", "project_id", "event_type"),
    )

    def __repr__(self) -> str:
        return f"<ProjectActivityLog(project={self.project_id}, event={self.event_type}, user={self.user_id})>"


class LicenseRecord(Base, TimestampMixin):
    """
    Issued application licenses for on-premise GenoLens deployments.
    Each record holds the signed license key and its metadata.
    """
    __tablename__ = "license_records"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    client_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    plan: Mapped[str] = mapped_column(String(50), nullable=False)
    product: Mapped[str] = mapped_column(String(100), nullable=False, default="genolens")
    expires_at: Mapped[int] = mapped_column(Integer, nullable=False)
    license_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_revoked: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    __table_args__ = (
        Index("ix_license_records_client_id", "client_id"),
        Index("ix_license_records_expires_at", "expires_at"),
    )

    @property
    def is_expired(self) -> bool:
        import time
        return time.time() > self.expires_at

    @property
    def days_until_expiry(self) -> Optional[int]:
        import time
        seconds_left = self.expires_at - time.time()
        if seconds_left < 0:
            return None
        return int(seconds_left // 86400)

    def __repr__(self) -> str:
        return f"<LicenseRecord(client={self.client_id}, plan={self.plan}, expires={self.expires_at})>"


class DeploymentStatus(str, enum.Enum):
    """Status values for a deployment job."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class ServerConfig(Base, TimestampMixin):
    """
    ServerConfig: Connection details for an on-premise deployment target.
    SSH private key is stored encrypted (Fernet) using DEPLOYMENT_ENCRYPTION_KEY.
    """
    __tablename__ = "server_configs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, comment="Human-readable label (e.g. 'Client Acme – prod')")
    host: Mapped[str] = mapped_column(String(255), nullable=False, comment="IP or hostname of the target server")
    ssh_port: Mapped[int] = mapped_column(Integer, nullable=False, default=22)
    ssh_user: Mapped[str] = mapped_column(String(100), nullable=False, default="ubuntu")
    ssh_key_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="Fernet-encrypted SSH private key PEM")
    repo_path: Mapped[str] = mapped_column(String(500), nullable=False, default="/opt/genolens", comment="Absolute path to the repo on the remote server")
    health_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True, comment="URL polled for health check after deployment")
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    secrets: Mapped[list["DeploymentSecret"]] = relationship(back_populates="server", cascade="all, delete-orphan")
    jobs: Mapped[list["DeploymentJob"]] = relationship(back_populates="server", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_server_configs_host", "host"),
    )

    def __repr__(self) -> str:
        return f"<ServerConfig(id={self.id}, name={self.name}, host={self.host})>"


class DeploymentSecret(Base, TimestampMixin):
    """
    DeploymentSecret: A single key/value pair for a server's .env.deploy file.
    The value is stored encrypted (Fernet). Keys are stored in plain text.
    """
    __tablename__ = "deployment_secrets"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    server_id: Mapped[UUID] = mapped_column(ForeignKey("server_configs.id", ondelete="CASCADE"), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False, comment="Environment variable name (e.g. POSTGRES_PASSWORD)")
    value_encrypted: Mapped[str] = mapped_column(Text, nullable=False, comment="Fernet-encrypted value")

    # Relationship
    server: Mapped["ServerConfig"] = relationship(back_populates="secrets")

    __table_args__ = (
        Index("ix_deployment_secrets_server_key", "server_id", "key", unique=True),
    )

    def __repr__(self) -> str:
        return f"<DeploymentSecret(server_id={self.server_id}, key={self.key})>"


class DeploymentJob(Base, TimestampMixin):
    """
    DeploymentJob: Tracks a single deployment run triggered from the admin UI.
    Logs are appended line-by-line; secrets are never written to logs.
    """
    __tablename__ = "deployment_jobs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    server_id: Mapped[UUID] = mapped_column(ForeignKey("server_configs.id", ondelete="SET NULL"), nullable=True, index=True)
    services: Mapped[list] = mapped_column(JSON, nullable=False, default=list, comment="List of service names deployed: backend, license, ai")
    skip_build: Mapped[bool] = mapped_column(nullable=False, default=False)
    status: Mapped[DeploymentStatus] = mapped_column(
        SQLEnum(DeploymentStatus, name="deployment_status_enum", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DeploymentStatus.PENDING,
        index=True,
    )
    logs: Mapped[Optional[str]] = mapped_column(Text, nullable=True, comment="Append-only deployment log (no secrets)")
    triggered_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, comment="User ID who triggered the deployment")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationship
    server: Mapped[Optional["ServerConfig"]] = relationship(back_populates="jobs")

    __table_args__ = (
        Index("ix_deployment_jobs_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<DeploymentJob(id={self.id}, server_id={self.server_id}, status={self.status})>"


class UserLoginEvent(Base):
    """
    UserLoginEvent: Deduplicated login event (one per 30-minute window per user).
    Used to track active users over time for admin analytics.
    user_id references Supabase Auth user UUID (no FK to local table).
    """
    __tablename__ = "user_login_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    user_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Supabase Auth user UUID who authenticated"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="When the login event was recorded"
    )

    __table_args__ = (
        Index("ix_user_login_events_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<UserLoginEvent(user={self.user_id}, created_at={self.created_at})>"


class SelfServiceAnalysis(Base):
    """
    SelfServiceAnalysis: A user-launched DESeq2/multi-method analysis job.
    Inputs are existing datasets (matrix + samples + comparisons).
    Outputs are DEG/enrichment datasets produced by the R pipeline.
    """
    __tablename__ = "self_service_analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    data_type: Mapped[OmicsDataType] = mapped_column(
        SQLEnum(OmicsDataType, name="omics_data_type", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=OmicsDataType.TRANSCRIPTOMICS,
        server_default=OmicsDataType.TRANSCRIPTOMICS.value,
    )
    status: Mapped[SelfServiceAnalysisStatus] = mapped_column(
        SQLEnum(SelfServiceAnalysisStatus, name="self_service_analysis_status"),
        nullable=False,
        default=SelfServiceAnalysisStatus.PENDING,
        index=True,
    )
    matrix_dataset_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    samples_dataset_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    comparisons_dataset_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result_dataset_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    intermediate_dataset_ids: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    current_step: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    progress_log: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[UUID] = mapped_column(nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_self_service_analyses_project_status", "project_id", "status"),
        Index("ix_self_service_analyses_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<SelfServiceAnalysis(id={self.id}, name={self.name}, status={self.status})>"


# ---------------------------------------------------------------------------
# Cosmetics module — claim referential (pathway -> skin claim associations)
# ---------------------------------------------------------------------------


class ClaimPathwayMapping(Base, TimestampMixin):
    """
    ClaimPathwayMapping: one row per curated pathway in the SciLicium claim
    referential. Maps a functional term (GO / Reactome / annoDB cluster) to one
    or more cosmetic claims, with the direction that supports the claim and the
    strength of the underlying evidence.

    Admin-managed (CRUD/import restricted to admins). Joined against
    EnrichmentPathway.pathway_id (via term_id_normalized) to score a comparison.
    """
    __tablename__ = "claim_pathway_mappings"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

    # Original term id as authored (e.g. "GO:0006084", "HSA-111447", "CL:9273")
    term_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    # Normalized join key (uppercased, "R-" prefix stripped, etc.) — see normalize_pathway_id()
    term_id_normalized: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    common_abbrev: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    suggested_acronym: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    original_claims: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_direction: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    updated_claim_framing: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    updated_direction: Mapped[ClaimDirection] = mapped_column(
        SQLEnum(ClaimDirection, name="claim_direction_enum"),
        nullable=False,
        default=ClaimDirection.UNKNOWN,
    )

    category: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    evidence_level: Mapped[EvidenceLevel] = mapped_column(
        SQLEnum(EvidenceLevel, name="evidence_level_enum"),
        nullable=False,
        default=EvidenceLevel.LOW,
    )

    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    caveats: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ref_cat: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Canonical claim slugs derived from original_claims / category (e.g. ["anti-ageing", "firming"])
    canonical_claims: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    is_active: Mapped[bool] = mapped_column(
        nullable=False, default=True, server_default=sa_text("true")
    )

    __table_args__ = (
        Index("ix_claim_pathway_mappings_norm_active", "term_id_normalized", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<ClaimPathwayMapping(term_id={self.term_id}, dir={self.updated_direction})>"


class ClaimReference(Base, TimestampMixin):
    """
    ClaimReference: category-level literature citation backing claim mappings.
    Sourced from the `References` sheet of the claim referential workbooks.
    """
    __tablename__ = "claim_references"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    ref_cat: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    def __repr__(self) -> str:
        return f"<ClaimReference(ref_cat={self.ref_cat})>"


class CosmeticClaim(Base, TimestampMixin):
    """
    CosmeticClaim: canonical, display-facing taxonomy of cosmetic claims
    (English labels). Each curated mapping is reduced to one or more of these
    slugs so the UI (radar, claim cards, skin schematic) shows a stable set.
    """
    __tablename__ = "cosmetic_claims"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    skin_zone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_active: Mapped[bool] = mapped_column(
        nullable=False, default=True, server_default=sa_text("true")
    )

    def __repr__(self) -> str:
        return f"<CosmeticClaim(slug={self.slug})>"


class CosmeticInterpretation(Base, TimestampMixin):
    """
    CosmeticInterpretation: cached AI cosmetic-focused interpretation for a
    comparison (kept separate from AIInterpretation so the transcriptomics flow
    is untouched). One row per (dataset, comparison).
    """
    __tablename__ = "cosmetic_interpretations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    dataset_id: Mapped[UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    comparison_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    interpretation: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    claims_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    dataset: Mapped["Dataset"] = relationship()

    __table_args__ = (
        Index(
            "ix_cosmetic_interpretations_dataset_comparison",
            "dataset_id",
            "comparison_name",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<CosmeticInterpretation(dataset_id={self.dataset_id}, comparison={self.comparison_name})>"
