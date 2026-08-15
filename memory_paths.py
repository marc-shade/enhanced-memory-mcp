#!/usr/bin/env python3
"""Single source of truth for where the memory database lives.

Several modules used to carry ~/.claude/enhanced_memories/memory.db inline, which
silently ignored every configuration override. That is harmless right up until
a test run or a second instance writes to the operator's real database instead
of its own -- so the resolution order lives here once and is imported rather
than retyped.

Order (highest precedence first):
    ENHANCED_MEMORY_DB_PATH / MEMORY_DB_PATH   full path to the database file
    ENHANCED_MEMORY_DIR     / MEMORY_DIR       directory holding memory.db
    ~/.claude/enhanced_memories/memory.db      default
"""

import os
from pathlib import Path

DEFAULT_MEMORY_DIR = Path.home() / ".claude" / "enhanced_memories"


def expand_config_path(value: str) -> Path:
    """Expand both $VARS and ~ in a configured path."""
    return Path(os.path.expandvars(os.path.expanduser(value)))


def get_memory_paths() -> tuple[Path, Path]:
    """Return (memory_dir, db_path) honouring the environment overrides."""
    db_override = os.environ.get("ENHANCED_MEMORY_DB_PATH") or os.environ.get(
        "MEMORY_DB_PATH"
    )
    if db_override:
        db_path = expand_config_path(db_override)
        return db_path.parent, db_path

    dir_override = os.environ.get("ENHANCED_MEMORY_DIR") or os.environ.get("MEMORY_DIR")
    memory_dir = (
        expand_config_path(dir_override) if dir_override else DEFAULT_MEMORY_DIR
    )
    return memory_dir, memory_dir / "memory.db"


def get_db_path() -> Path:
    """Return just the resolved database path."""
    return get_memory_paths()[1]
