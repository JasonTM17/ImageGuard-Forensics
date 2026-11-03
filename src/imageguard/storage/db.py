"""SQLite session factory — WAL mode, create_all + schema_version check."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from imageguard.storage.models import SCHEMA_VERSION, Base, MetaKV

from imageguard.core.paths import data_root

DEFAULT_DB = data_root() / "imageguard.db"


def make_session_factory(db_path: Path | str = DEFAULT_DB) -> sessionmaker[Session]:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(f"sqlite:///{p}", connect_args={"timeout": 10})
    with eng.connect() as c:
        c.exec_driver_sql("PRAGMA journal_mode=WAL")
        c.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(eng)
    factory = sessionmaker(eng, expire_on_commit=False)
    with factory() as s:
        row = s.get(MetaKV, "schema_version")
        if row is None:
            s.add(MetaKV(key="schema_version", value=str(SCHEMA_VERSION)))
            s.commit()
        elif int(row.value) != SCHEMA_VERSION:
            raise RuntimeError(
                f"schema_version {row.value} != {SCHEMA_VERSION} — xem "
                "docs/database.md (recreate data/imageguard.db hoặc migrate)"
            )
    return factory
