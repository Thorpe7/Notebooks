"""
Ingest restructured TCIA data into XNAT using xnatpy.

This script reads the XNAT-ready data structure from ready_for_ingest/
and uploads it to an XNAT server using the xnatpy library.

Structure expected:
    ready_for_ingest/
    PROJECT_ID/
        SUBJECT_ID/
            SESSION_UID/
                SCAN_UID/
                   DICOM/
                       *.dcm
                 session_metadata.json
"""

import json
import logging
import os
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional
import xnat
from xnat.mixin import ProjectData, SubjectData
from xnat.session import XNATSession



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def connect_to_xnat(
    server: str,
    user_env_var: str|None = 'ALIAS',
    password_env_var: str|None = 'SECRET'
):
    """
    Connect to XNAT server using credentials from environment variables.

    Args:
        server: XNAT server URL (e.g., 'https://xnat.example.com')
        user_env_var: Name of environment variable containing XNAT username (default: 'ALIAS')
        password_env_var: Name of environment variable containing XNAT password (default: 'SECRET')

    Returns:
        XNATSession object

    Raises:
        ValueError: If specified environment variables are not set
        Exception: If connection fails
    """
    logger.info(f"Connecting to XNAT server: {server}")

    # Get credentials from environment variables
    username = os.environ.get(user_env_var)
    password = os.environ.get(password_env_var)

    if not username or not password:
        raise ValueError(
            f"XNAT credentials not found. Set the {user_env_var} and {password_env_var} environment variables."
        )

    try:
        session = xnat.connect(server, user=username, password=password)
        logger.info("Successfully connected to XNAT")
        return session
    except Exception as e:
        logger.error(f"Failed to connect to XNAT: {e}")
        raise


def get_or_create_project(session: XNATSession, project_id: str) -> ProjectData:
    """
    Get existing project or create new one.

    Args:
        session: XNATSession object
        project_id: Project ID to create/retrieve

    Returns:
        XNAT Project object
    """
    if project_id in session.projects:
        project = session.projects[project_id]
        logger.info(f"Found existing project: {project_id}")
        return project
    else:
        raise ValueError(f"Project {project_id} not found on server. Please create the project first.")


def get_or_create_subject(
    session: XNATSession,
    project: ProjectData,
    subject_id: str
) -> SubjectData:
    """
    Get existing subject or create new one.

    Args:
        session: XNATSession object
        project: XNAT Project object
        subject_id: Subject ID to create/retrieve

    Returns:
        XNAT Subject object
    """
    if subject_id in project.subjects:
        subject = project.subjects[subject_id]
        logger.info(f"  Found existing subject: {subject_id}")
        return subject
    else:
        logger.info(f"  Creating new subject: {subject_id}")
        subject = session.classes.SubjectData(parent=project, label=subject_id)
        return subject


def upload_session_dicom(
    session: XNATSession,
    project: ProjectData,
    subject_label: str,
    experiment_label: str,
    dicom_files: list[Path],
    tmp_dir: str
) -> None:
    """
    Upload DICOM files to XNAT using services.import_().

    Args:
        session: XNATSession object
        project: XNAT Project object
        subject_label: Subject label/ID
        experiment_label: Experiment/session label
        dicom_files: List of paths to DICOM files
        tmp_dir: Temporary directory for creating zip files

    Returns:
        None
    """
    if not dicom_files:
        logger.warning(f"No DICOM files to upload for {subject_label}/{experiment_label}")
        return

    logger.info(f"Uploading {len(dicom_files)} DICOM files for {subject_label}/{experiment_label}")

    # Create zip file containing all DICOM files
    zip_name = f"{experiment_label}.zip"
    zip_path = os.path.join(tmp_dir, zip_name)

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for dcm_file in dicom_files:
                zf.write(str(dcm_file), arcname=dcm_file.name)

        # Use XNAT's import service - import directly to archive (no prearchive)
        session.services.import_(
            zip_path,
            project=project.id,
            subject=subject_label,
            experiment=experiment_label,
            overwrite="delete"
        )

        logger.info(f"Successfully uploaded {len(dicom_files)} files to {subject_label}/{experiment_label}")

    except Exception as e:
        logger.error(f"Failed to upload DICOM files: {e}")
        raise
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)


def ingest_subject(
    session: XNATSession,
    project: ProjectData,
    subject_dir: Path,
    tmp_dir: str
) -> Dict[str, int]:
    """
    Ingest a single subject's data into XNAT.

    Args:
        session: XNATSession object
        project: XNAT Project object
        subject_dir: Path to subject directory
        tmp_dir: Temporary directory for creating zip files

    Returns:
        Dictionary with statistics (sessions, scans, files)
    """
    subject_id = subject_dir.name
    stats = {'sessions': 0, 'scans': 0, 'files': 0}

    logger.info(f"Processing subject: {subject_id}")

    # Ensure subject exists
    get_or_create_subject(session, project, subject_id)

    # Iterate through sessions (Study UIDs)
    for session_dir in subject_dir.iterdir():
        if not session_dir.is_dir():
            continue

        session_uid = session_dir.name
        metadata_file = session_dir / 'session_metadata.json'

        if not metadata_file.exists():
            logger.warning(f"  Metadata file not found: {metadata_file}")
            continue

        # Load session metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        stats['sessions'] += 1
        logger.info(f"  Processing session: {session_uid[:50]}...")

        # Collect all DICOM files from all scans in this session
        all_dicom_files = []

        # Iterate through scans (Series UIDs)
        for scan_dir in session_dir.iterdir():
            if not scan_dir.is_dir():
                continue

            series_uid = scan_dir.name
            dicom_dir = scan_dir / 'DICOM'

            if not dicom_dir.exists():
                continue

            # Get scan metadata
            if series_uid in metadata.get('scans', {}):
                stats['scans'] += 1
                dicom_files = list(dicom_dir.glob('*.dcm'))
                all_dicom_files.extend(dicom_files)
                stats['files'] += len(dicom_files)
            else:
                logger.warning(f"Scan metadata not found for: {series_uid}")

        # Upload all DICOM files for this session
        if all_dicom_files:
            # Use a shortened session label for XNAT
            experiment_label = session_uid[:50] if len(session_uid) > 50 else session_uid
            upload_session_dicom(
                session, project, subject_id, experiment_label, all_dicom_files, tmp_dir
            )

    return stats


def ingest_tcia_to_xnat(
    data_dir: Path,
    xnat_server: str,
    project_id: Optional[str] = None,
    dry_run: bool = False,
    user_env_var: str = 'ALIAS',
    password_env_var: str = 'SECRET'
) -> Dict[str, int]:
    """
    Main function to ingest restructured TCIA data into XNAT.

    Args:
        data_dir: Path to ready_for_ingest directory
        xnat_server: XNAT server URL
        project_id: XNAT project ID (if None, uses directory name)
        dry_run: If True, uploads only the first subject then terminates (for testing)
        user_env_var: Name of environment variable containing XNAT username (default: 'ALIAS')
        password_env_var: Name of environment variable containing XNAT password (default: 'SECRET')

    Returns:
        Dictionary with statistics
    """
    logger.info("=" * 70)
    logger.info("Starting XNAT Ingestion")
    logger.info("=" * 70)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Statistics
    stats = {
        'subjects': 0,
        'sessions': 0,
        'scans': 0,
        'files': 0
    }

    # Find project directory (should be single directory under ready_for_ingest)
    project_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    if not project_dirs:
        raise ValueError(f"No project directories found in {data_dir}")

    if len(project_dirs) > 1:
        logger.warning(f"Multiple project directories found. Processing first: {project_dirs[0].name}")

    project_dir = project_dirs[0]
    proj_id = project_id or project_dir.name

    logger.info(f"Project: {proj_id}")
    logger.info(f"Data directory: {data_dir.absolute()}")
    logger.info(f"XNAT server: {xnat_server}")
    logger.info(f"Dry run: {dry_run}")
    if dry_run:
        logger.info("DRY RUN MODE - Will upload first subject only")
    logger.info("=" * 70)

    # Connect to XNAT
    session = connect_to_xnat(xnat_server, user_env_var, password_env_var)
    project = get_or_create_project(session, proj_id)

    # Create temporary directory for zip files
    tmp_dir = tempfile.mkdtemp(prefix="xnat_ingest_")

    try:
        # Process each subject
        for subject_dir in project_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            stats['subjects'] += 1
            subject_stats = ingest_subject(session, project, subject_dir, tmp_dir)

            stats['sessions'] += subject_stats['sessions']
            stats['scans'] += subject_stats['scans']
            stats['files'] += subject_stats.get('files', 0)

            # In dry-run mode, stop after first subject
            if dry_run:
                logger.info("")
                logger.info("=" * 70)
                logger.info("DRY RUN COMPLETE - Stopped after first subject")
                logger.info("=" * 70)
                break

        logger.info("=" * 70)
        logger.info("Ingestion Complete!")
        logger.info("=" * 70)
        logger.info(f"Subjects processed: {stats['subjects']}")
        logger.info(f"Sessions created:   {stats['sessions']}")
        logger.info(f"Scans uploaded:     {stats['scans']}")
        logger.info(f"Files uploaded:     {stats['files']}")
        logger.info("=" * 70)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        session.disconnect()
        logger.info("Disconnected from XNAT")

    return stats


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Ingest restructured TCIA data into XNAT. '
                    'Credentials are read from environment variables (configurable).'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/ready_for_ingest'),
        help='Path to ready_for_ingest directory'
    )
    parser.add_argument(
        '--xnat-server',
        type=str,
        required=True,
        help='XNAT server URL (e.g., https://xnat.example.com)'
    )
    parser.add_argument(
        '--project-id',
        type=str,
        help='XNAT project ID (defaults to directory name)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Upload only the first subject then terminate (for testing)'
    )
    parser.add_argument(
        '--user-env-var',
        type=str,
        default='ALIAS',
        help='Name of environment variable containing XNAT username (default: ALIAS)'
    )
    parser.add_argument(
        '--password-env-var',
        type=str,
        default='SECRET',
        help='Name of environment variable containing XNAT password (default: SECRET)'
    )

    args = parser.parse_args()

    try:
        stats = ingest_tcia_to_xnat(
            data_dir=args.data_dir,
            xnat_server=args.xnat_server,
            project_id=args.project_id,
            dry_run=args.dry_run,
            user_env_var=args.user_env_var,
            password_env_var=args.password_env_var
        )

        return 0

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
