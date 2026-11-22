"""SQLite session factory — WAL mode, create_all + schema_version check."""

from __future__ import annotations

import weakref
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from imageguard.core.paths import data_root
from imageguard.storage.models import SCHEMA_VERSION, Base, MetaKV

DEFAULT_DB = data_root() / "imageguard.db"


def make_session_factory(db_path: Path | str | None = None) -> sessionmaker[Session]:
    p = Path(db_path) if db_path is not None else data_root() / "imageguard.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(URL.create("sqlite", database=str(p)), connect_args={"timeout": 10})

    # PRAGMA foreign_keys chỉ có hiệu lực PER CONNECTION — gắn listener để
    # MỌI connection từ pool đều bật FK, không chỉ connection khởi tạo.
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    try:
        with eng.connect() as c:
            c.exec_driver_sql("PRAGMA journal_mode=WAL")
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
    except Exception:
        eng.dispose()
        raise
    weakref.finalize(factory, eng.dispose)
    return factory
