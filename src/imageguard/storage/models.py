"""DB schema — SQLAlchemy 2.x declarative. 5 bảng + meta.schema_version.

Quyết định (docs/database.md): không Alembic — single-user desktop, schema
nhỏ; nâng cấp = bump SCHEMA_VERSION + recreate (ghi rõ, không im lặng).
DB chỉ lưu RELATIVE paths — absolute path cấm (di chuyển máy không vỡ).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    pass


class MetaKV(Base):
    __tablename__ = "meta"
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class ImageAsset(Base):
    __tablename__ = "image_assets"
    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    file_name: Mapped[str] = mapped_column(String)  # basename only
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    format: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="image")


class Analysis(Base):
    __tablename__ = "analyses"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    image_id: Mapped[int] = mapped_column(ForeignKey("image_assets.id"))
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))
    tool_version: Mapped[str] = mapped_column(String(16))
    overall_verdict: Mapped[str] = mapped_column(String(32))
    config_hash: Mapped[str] = mapped_column(String(16))
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    limitations_json: Mapped[str] = mapped_column(Text, default="[]")

    image: Mapped[ImageAsset] = relationship(back_populates="analyses")
    method_results: Mapped[list["MethodResult"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan")
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan")


class MethodResult(Base):
    __tablename__ = "method_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"))
    method: Mapped[str] = mapped_column(String(48))
    verdict: Mapped[str] = mapped_column(String(32))
    score: Mapped[float | None] = mapped_column(nullable=True)
    score_definition: Mapped[str] = mapped_column(Text, default="")
    evidence_json: Mapped[str] = mapped_column(Text, default="{}")
    limitations_json: Mapped[str] = mapped_column(Text, default="[]")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    analysis: Mapped[Analysis] = relationship(back_populates="method_results")


class Artifact(Base):
    __tablename__ = "artifacts"
    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"))
    kind: Mapped[str] = mapped_column(String(32))  # mask|heatmap|ela|pairs|report
    method: Mapped[str] = mapped_column(String(48), default="")
    rel_path: Mapped[str] = mapped_column(String)  # relative to artifacts root
    analysis: Mapped[Analysis] = relationship(back_populates="artifacts")


class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(64), index=True)
    dataset: Mapped[str] = mapped_column(String(32))
    split: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))
    provenance_json: Mapped[str] = mapped_column(Text, default="{}")
    results: Mapped[list["ExperimentResult"]] = relationship(
        cascade="all, delete-orphan")


class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    method: Mapped[str] = mapped_column(String(48))
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
