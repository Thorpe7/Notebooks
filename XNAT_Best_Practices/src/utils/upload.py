"""
Upload structured DICOM archives to XNAT.

This module provides functions to upload pre-structured archives (created by
structuring.py) to an XNAT instance using the xnatpy import service.

Usage:
    from src.utils.upload import upload_archive_to_xnat

    result = upload_archive_to_xnat(
        archive_path="data/Test_Subject_999_archive",
        project_id="NewUserTrainingProject",
        subject_label="Test_Subject_999",
        host="https://xnat.example.com",
        user="username",
        password="password",
    )
"""

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

    Defaults to environment variables if arguments are not supplied:
        - XNAT_HOST
        - XNAT_USER
        - XNAT_PASS

    Parameters
    ----------
    host : str, optional
        XNAT server URL.
    user : str, optional
        XNAT username.
    password : str, optional
        XNAT password.

    Returns
    -------
    xnat.XNATSession
        Active XNAT session.

    Raises
    ------
    ValueError
        If required credentials are not provided.
    """
    host = host or os.environ.get("XNAT_HOST")
    user = user or os.environ.get("XNAT_USER")
    password = password or os.environ.get("XNAT_PASS")

    if host is None or user is None or password is None:
        raise ValueError(
            "XNAT_HOST, XNAT_USER, XNAT_PASS must be set in the environment or "
            "passed explicitly."
        )

    session = xnat.connect(host, user=user, password=password)
    return session


def upload_archive_to_xnat(
    archive_path: Union[str, Path],
    project_id: str,
    subject_label: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    session: Optional[xnat.XNATSession] = None,
    destination: str = "/archive",
    overwrite: str = "none",
    cleanup_zip: bool = True,
) -> dict:
    """
    Upload a structured archive directory or ZIP to XNAT.

    Parameters
    ----------
    archive_path : str or Path
        Path to the structured archive directory or ZIP file.
        If a directory is provided, it will be zipped before upload.
    project_id : str
        XNAT project ID to upload to.
    subject_label : str
        Subject label for the upload.
    host : str, optional
        XNAT server URL. Falls back to XNAT_HOST env var.
    user : str, optional
        XNAT username. Falls back to XNAT_USER env var.
    password : str, optional
        XNAT password. Falls back to XNAT_PASS env var.
    session : xnat.XNATSession, optional
        Existing XNAT session to reuse. If provided, host/user/password are ignored.
    destination : str
        Import destination. Options:
        - "/archive" (default): Import directly to archive
        - "/prearchive": Import to prearchive for review first
    overwrite : str
        Overwrite behavior. Options:
        - "none" (default): Do not overwrite existing data
        - "append": Append to existing session
        - "delete": Delete and replace existing session
    cleanup_zip : bool
        If True and archive_path is a directory, delete the temporary ZIP after upload.

    Returns
    -------
    dict
        Result information including:
        - success: bool
        - message: str
        - import_result: The result from XNAT import service (if successful)

    Example
    -------
    >>> from src.utils.upload import upload_archive_to_xnat
    >>> result = upload_archive_to_xnat(
    ...     archive_path="data/Test_Subject_999_archive",
    ...     project_id="NewUserTrainingProject",
    ...     subject_label="Test_Subject_999",
    ... )
    >>> print(result["message"])
    """
    archive_path = Path(archive_path)

    if not archive_path.exists():
        return {
            "success": False,
            "message": f"Archive path not found: {archive_path}",
            "import_result": None,
        }

    # If directory, create a ZIP file
    temp_zip_created = False
    if archive_path.is_dir():
        zip_path = archive_path.with_suffix(".zip")
        print(f"Creating ZIP archive from directory: {archive_path}")
        shutil.make_archive(
            str(zip_path.with_suffix("")),
            "zip",
            archive_path.parent,
            archive_path.name,
        )
        upload_path = zip_path
        temp_zip_created = True
    else:
        upload_path = archive_path

    # Verify it's a ZIP file
    if not str(upload_path).endswith(".zip"):
        return {
            "success": False,
            "message": f"Upload path must be a ZIP file or directory: {upload_path}",
            "import_result": None,
        }

    # Handle session management
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

        print(f"Uploading to XNAT...")
        print(f"  Project: {project_id}")
        print(f"  Subject: {subject_label}")
        print(f"  Destination: {destination}")
        print(f"  Archive: {upload_path}")

        # Perform the import
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

        # Clean up temporary ZIP if requested
        if temp_zip_created and cleanup_zip and upload_path.exists():
            upload_path.unlink()
            print(f"Cleaned up temporary ZIP: {upload_path}")


def upload_to_prearchive(
    archive_path: Union[str, Path],
    project_id: str,
    subject_label: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    session: Optional[xnat.XNATSession] = None,
) -> dict:
    """
    Upload a structured archive to the XNAT prearchive for review.

    This is a convenience wrapper around upload_archive_to_xnat that sets
    destination="/prearchive". Use this when you want to review data before
    committing it to the archive.

    Parameters
    ----------
    archive_path : str or Path
        Path to the structured archive directory or ZIP file.
    project_id : str
        XNAT project ID.
    subject_label : str
        Subject label.
    host, user, password : str, optional
        XNAT credentials (falls back to environment variables).
    session : xnat.XNATSession, optional
        Existing XNAT session to reuse.

    Returns
    -------
    dict
        Result information from the upload.
    """
    return upload_archive_to_xnat(
        archive_path=archive_path,
        project_id=project_id,
        subject_label=subject_label,
        host=host,
        user=user,
        password=password,
        session=session,
        destination="/prearchive",
    )


def verify_upload(
    project_id: str,
    subject_label: str,
    session_label: Optional[str] = None,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    session: Optional[xnat.XNATSession] = None,
) -> dict:
    """
    Verify that data was successfully uploaded to XNAT.

    Parameters
    ----------
    project_id : str
        XNAT project ID.
    subject_label : str
        Subject label to verify.
    session_label : str, optional
        Specific session label to check. If None, returns all sessions.
    host, user, password : str, optional
        XNAT credentials.
    session : xnat.XNATSession, optional
        Existing XNAT session to reuse.

    Returns
    -------
    dict
        Verification results including:
        - found: bool
        - subject_exists: bool
        - sessions: list of session labels
        - scans: dict mapping session labels to scan info
    """
    close_session = False
    if session is None:
        session = get_xnat_session(host, user, password)
        close_session = True

    try:
        if project_id not in session.projects:
            return {
                "found": False,
                "subject_exists": False,
                "sessions": [],
                "scans": {},
                "message": f"Project '{project_id}' not found",
            }

        project = session.projects[project_id]

        if subject_label not in project.subjects:
            return {
                "found": False,
                "subject_exists": False,
                "sessions": [],
                "scans": {},
                "message": f"Subject '{subject_label}' not found in project",
            }

        subject = project.subjects[subject_label]
        sessions_info = []
        scans_info = {}

        for exp in subject.experiments.values():
            sessions_info.append(exp.label)
            scans_info[exp.label] = []

            for scan in exp.scans.values():
                scans_info[exp.label].append({
                    "id": scan.id,
                    "type": getattr(scan, "type", None),
                    "series_description": getattr(scan, "series_description", None),
                })

        # Filter to specific session if requested
        if session_label:
            if session_label in sessions_info:
                return {
                    "found": True,
                    "subject_exists": True,
                    "sessions": [session_label],
                    "scans": {session_label: scans_info.get(session_label, [])},
                    "message": "Session found",
                }
            else:
                return {
                    "found": False,
                    "subject_exists": True,
                    "sessions": sessions_info,
                    "scans": scans_info,
                    "message": f"Session '{session_label}' not found, but subject exists",
                }

        return {
            "found": len(sessions_info) > 0,
            "subject_exists": True,
            "sessions": sessions_info,
            "scans": scans_info,
            "message": f"Found {len(sessions_info)} session(s)",
        }

    finally:
        if close_session:
            session.disconnect()
