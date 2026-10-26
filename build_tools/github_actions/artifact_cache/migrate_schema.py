#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Explicit revision-3/4/5 -> revision-6 migration: add bundle membership.

Existing component keys get an empty member list. Legacy namespaces are removed.

Stop all cache commands first. A required new SQLite backup precedes all changes.
Keys with identical (epoch, artifact_id, fingerprint) are merged onto the smallest
existing key ID. Productions and observations are relinked; entry/run/artifact IDs,
file inventories, state, timestamps and payload locations are preserved. Version-3
keys acquire fixed epoch 1. CacheDB never migrates a database implicitly.
"""

import argparse
import json
import os
from pathlib import Path
import sqlite3

from .cache_db import SchemaError, _connect, _validate_schema

LEGACY_EPOCH = 1
SCHEMA_VERSION = 6
SCHEMA = Path(__file__).resolve().parents[2] / "therock_fingerprint_cache.sql"


def _quote_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def _check_integrity(db):
    if db.execute("PRAGMA foreign_key_check").fetchall():
        raise SchemaError("foreign-key check failed; database was not migrated")
    if [r[0] for r in db.execute("PRAGMA integrity_check")] != ["ok"]:
        raise SchemaError("integrity check failed; database was not migrated")


def _backup(database, destination):
    # BEGIN IMMEDIATE on the migration connection excludes other writers while
    # this separate read connection captures the last committed snapshot.
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=5) as source:
        with sqlite3.connect(destination) as target:
            source.backup(target)
            _check_integrity(target)


def migrate(database: Path, backup: Path) -> dict:
    database = Path(database).expanduser().resolve(strict=True)
    requested_backup = Path(backup).expanduser().absolute()
    if os.path.lexists(requested_backup):
        raise FileExistsError(f"backup must be a new file: {requested_backup}")
    backup = requested_backup.resolve()
    if backup == database:
        raise ValueError("backup must differ from the database")
    with sqlite3.connect(":memory:") as reference:
        reference.executescript(SCHEMA.read_text(encoding="utf-8"))
        key_ddl = reference.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='artifact_keys'"
        ).fetchone()[0]

    db = _connect(database, 5)
    try:
        # Referenced-table reconstruction and coordinated relinking require FK
        # checking at the end, inside the same transaction, before commit.
        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("BEGIN IMMEDIATE")
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 3, 4, 5, SCHEMA_VERSION):
            raise SchemaError(f"unsupported database schema version: {version}")
        columns = {r["name"] for r in db.execute('PRAGMA table_info("artifact_keys")')}
        if "components_json" in columns:
            _validate_schema(db)
            _check_integrity(db)
            if version != SCHEMA_VERSION:
                raise SchemaError("unrecognized schema version for namespace-free keys")
            db.rollback()
            return {"operation": "bundle-membership-migration", "status": "already_current",
                    "schema_version": SCHEMA_VERSION, "database": str(database), "backup": None}
        legacy = "epoch" not in columns
        namespaced = "key_namespace" in columns
        component_only = not namespaced
        expected = {"artifact_key_id", "artifact_id", "fingerprint"}
        if namespaced:
            expected.add("key_namespace")
        if not legacy:
            expected.add("epoch")
        if columns != expected or version not in ((0, 3) if legacy else (0, 4) if namespaced else (0, 5)):
            raise SchemaError("unrecognized artifact_keys shape/version; migration refused")
        _validate_schema(db, legacy=legacy, namespaced=namespaced and not legacy,
                         component_only=component_only)
        _check_integrity(db)
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='artifact_keys_v6_new'").fetchone():
            raise SchemaError("artifact_keys_v6_new already exists; migration refused")
        expected_unique = (("key_namespace", "artifact_id", "fingerprint") if legacy else
                           ("key_namespace", "epoch", "artifact_id", "fingerprint") if namespaced else
                           ("epoch", "artifact_id", "fingerprint"))
        for index in db.execute('PRAGMA index_list("artifact_keys")').fetchall():
            names = tuple(r["name"] for r in db.execute(
                "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (index["name"],)
            ))
            if index["unique"] and (index["origin"] != "u" or index["partial"] or names not in {
                expected_unique, ("artifact_key_id", "artifact_id"),
            }):
                raise SchemaError("unrecognized unique artifact_keys index; migration refused")
        # Save and restore all dependent SQL. Unknown namespace-based extensions
        # need an explicit redesign; never silently discard or rewrite them.
        dependents = db.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE type IN ('trigger', 'view') OR "
            "(type='index' AND tbl_name='artifact_keys' AND sql IS NOT NULL) ORDER BY rowid"
        ).fetchall()
        for obj in dependents:
            if "key_namespace" in obj["sql"].lower():
                raise SchemaError(f"custom object {obj['name']} refers to key_namespace; migration refused")
        counts = {table: db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                  for table in ("artifacts", "prototype_runs", "artifact_observations", "cache_entries", "artifact_files")}
        before_keys = db.execute("SELECT count(*) FROM artifact_keys").fetchone()[0]
        _backup(database, backup)
        for obj in dependents:
            db.execute(f"DROP {obj['type'].upper()} {_quote_identifier(obj['name'])}")
        db.execute("CREATE TEMP TABLE key_remap (old_id INTEGER PRIMARY KEY, new_id INTEGER NOT NULL)")
        partition = "artifact_id, fingerprint" if legacy else "epoch, artifact_id, fingerprint"
        db.execute(
            "INSERT INTO key_remap SELECT artifact_key_id, "
            f"min(artifact_key_id) OVER (PARTITION BY {partition}) FROM artifact_keys"
        )
        db.execute(key_ddl.replace("CREATE TABLE artifact_keys", "CREATE TABLE artifact_keys_v6_new", 1))
        epoch = str(LEGACY_EPOCH) if legacy else "k.epoch"
        db.execute(
            "INSERT INTO artifact_keys_v6_new (artifact_key_id, artifact_id, epoch, fingerprint) "
            f"SELECT k.artifact_key_id, k.artifact_id, {epoch}, k.fingerprint "
            "FROM artifact_keys k JOIN key_remap m ON m.old_id = k.artifact_key_id "
            "WHERE m.old_id = m.new_id"
        )
        for table in ("cache_entries", "artifact_observations"):
            db.execute(
                f"UPDATE {table} SET artifact_key_id = (SELECT new_id FROM key_remap "
                f"WHERE old_id = {table}.artifact_key_id) WHERE artifact_key_id IN "
                "(SELECT old_id FROM key_remap WHERE old_id <> new_id)"
            )
        db.execute("DROP TABLE artifact_keys")
        db.execute("ALTER TABLE artifact_keys_v6_new RENAME TO artifact_keys")
        for obj in dependents:
            db.execute(obj["sql"])
        for obj in dependents:
            if obj["type"] == "view":
                db.execute(f'SELECT * FROM {_quote_identifier(obj["name"])} LIMIT 0').fetchall()
        for table, count in counts.items():
            if db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0] != count:
                raise SchemaError(f"{table} row count changed during migration")
        after_keys = db.execute("SELECT count(*) FROM artifact_keys").fetchone()[0]
        if after_keys != db.execute("SELECT count(DISTINCT new_id) FROM key_remap").fetchone()[0]:
            raise SchemaError("unexpected key count after merging")
        db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        _validate_schema(db)
        _check_integrity(db)
        db.execute("DROP TABLE key_remap")
        db.commit()
        return {"operation": "bundle-membership-migration", "status": "migrated",
                "database": str(database), "backup": str(backup),
                "schema_version": SCHEMA_VERSION,
                "legacy_epoch": LEGACY_EPOCH if legacy else None,
                "keys_before": before_keys, "keys_after": after_keys,
                "merged_keys": before_keys - after_keys,
                "preserved_rows": counts}
    except BaseException:
        db.rollback()
        raise
    finally:
        db.execute("PRAGMA foreign_keys = ON")
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(migrate(args.db, args.backup), indent=2))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
