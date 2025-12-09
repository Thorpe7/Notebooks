"""
Upload structured DICOM archives to XNAT.

This module provides a single main entry point:

    from src.utils.upload import upload_archive_to_xnat

    result = upload_archive_to_xnat(
        archive_path=archive_zip,
        project_id="NewUserTrainingProject",
        subject_label="Test_Subject_999",
        destination="/archive",  # or "/prearchive"
    )

The XNAT connection defaults to the environment variables:

    XNAT_HOST
    XNAT_USER
    XNAT_PASS
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

import xnat


def get_xnat_session(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> xnat.XNATSession:
    """
    Establish and return an XNAT connection.

    If host/user/password are not provided, fall back to environment vars:
        - XNAT_HOST
        - XNAT_USER
        - XNAT_PASS
    """
    host = host or os.environ.get("XNAT_HOST")
    user = user or os.environ.get("XNAT_USER")
    password = password or os.environ.get("XNAT_PASS")

    if host is None or user is None or password is None:
        raise ValueError(
            "XNAT_HOST, XNAT_USER, XNAT_PASS must be set in the environment "
            "or host/user/password must be passed explicitly."
        )

    # xnat.connect returns an XNATSession
    return xnat.connect(host, user=user, password=password)


def upload_archive_to_xnat(
    archive_path: Union[str, Path],
    project_id: str,
    subject_label: str,
    destination: str = "/archive",
    overwrite: str = "delete",
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    session: Optional[xnat.XNATSession] = None,
    cleanup_zip: bool = True,
) -> dict:
    """
    Upload a structured archive directory or ZIP to XNAT.

    Parameters
    ----------
    archive_path : str or Path
        Path to the structured archive directory or ZIP file, typically
        produced by your `structuring.py` script, e.g.:

            data/Test_Subject_999_archive/
            or
            data/Test_Subject_999_archive.zip

    project_id : str
        XNAT project ID to upload into.
    subject_label : str
        Expected subject label (used for logging and as the `subject` parameter
        in the import call; should match the subject folder name in the archive).
    destination : str
        XNAT import destination path. Common values:
            "/archive"    -> send directly to archive
            "/prearchive" -> send to prearchive
    overwrite : str
        XNAT overwrite behavior ("delete", "append", "none").
        Default "delete" gives deterministic example behavior.
    host, user, password : str, optional
        XNAT connection settings. If omitted, falls back to env vars.
    session : xnat.XNATSession, optional
        Existing XNAT session to reuse. If provided, host/user/password are ignored.
    cleanup_zip : bool
        If True and a temporary ZIP is created from a directory, delete it after upload.

    Returns
    -------
    dict
        {
            "success": bool,
            "message": str,
            "import_result": <raw result from xnatpy or None>,
        }
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        return {
            "success": False,
            "message": f"Archive path not found: {archive_path}",
            "import_result": None,
        }

    # If archive_path is a directory (<subject>_archive/), zip it so the ZIP root
    # is that directory, as XNAT expects for pre-structured archives.
    temp_zip_created = False
    if archive_path.is_dir():
        zip_path = archive_path.with_suffix(".zip")
        print(f"Creating ZIP archive from directory: {archive_path}")
        # root_dir = parent of archive dir, base_dir = archive dir name
        shutil.make_archive(
            str(zip_path.with_suffix("")),
            "zip",
            root_dir=archive_path.parent,
            base_dir=archive_path.name,
        )
        upload_path = zip_path
        temp_zip_created = True
    else:
        upload_path = archive_path

    # Sanity check: XNAT import_ expects a ZIP here
    if not str(upload_path).lower().endswith(".zip"):
        return {
            "success": False,
            "message": f"Upload path must be a ZIP file or directory: {upload_path}",
            "import_result": None,
        }

    # Manage session lifecycle
    close_session = False
    if session is None:
        session = get_xnat_session(host, user, password)
        close_session = True

    try:
        # Verify project exists
        if project_id not in session.projects:
            available = list(session.projects.keys())
            return {
                "success": False,
                "message": f"Project '{project_id}' not found. Available: {available}",
                "import_result": None,
            }

        print("Uploading to XNAT...")
        print(f"  Project: {project_id}")
        print(f"  Subject: {subject_label}")
        print(f"  Destination: {destination}")
        print(f"  Archive: {upload_path}")

        # Core import call: pre-structured archive + explicit project/subject/destination
        import_result = session.services.import_(
            str(upload_path),
            project=project_id,
            subject=subject_label,
            destination=destination,
            overwrite=overwrite,
        )

        print(f"Import complete: {import_result}")

        return {
            "success": True,
            "message": "Upload completed successfully",
            "import_result": import_result,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Upload failed: {str(e)}",
            "import_result": None,
        }

    finally:
        if close_session:
            session.disconnect()

        if temp_zip_created and cleanup_zip and upload_path.exists():
            upload_path.unlink()
            print(f"Cleaned up temporary ZIP: {upload_path}")
