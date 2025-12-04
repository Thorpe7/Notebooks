"""XNAT I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import os

import xnat  # type: ignore


def get_xnat_connection(host: str | None = None, username: str | None = None, password: str | None = None):
    """Create and return an XNAT connection."""
    host = host or os.environ.get("XNAT_HOST")
    username = username or os.environ.get("XNAT_USER")
    password = password or os.environ.get("XNAT_PASS")

    if host is None or username is None or password is None:
        raise ValueError("XNAT_HOST, XNAT_USER, XNAT_PASS must be set or passed explicitly.")

    return xnat.connect(host=host, user=username, password=password)


def fetch_project_metadata(connection, project_id: str) -> Dict[str, Any]:
    """Fetch basic metadata for a given XNAT project."""
    if project_id not in connection.projects:
        raise KeyError(f"Project '{project_id}' not found on XNAT.")

    project = connection.projects[project_id]

    metadata: Dict[str, Any] = {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "num_experiments": len(project.experiments),
    }

    return metadata
