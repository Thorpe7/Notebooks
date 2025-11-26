"""
Reorganize TCIA data into XNAT-ready structure using metadata.csv

This script reads the TCIA manifest metadata.csv and restructures the data
into a clean directory hierarchy suitable for XNAT ingestion.

Uses DICOM identifiers:
- Subject: Subject ID
- Session: Study Instance UID
- Scan: Series Instance UID

Output structure:
    ready_for_ingest/
    └── UPENN-GBM/
        └── UPENN-GBM-00001/  (Subject ID)
            └── 1.3.6.1.4.1.14519.5.2.1.30392.../  (Study UID)
                ├── 1.3.6.1.4.1.14519.5.2.1.33749.../  (Series UID)
                │   └── DICOM/
                │       └── *.dcm
                └── session_metadata.json
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_metadata_csv(metadata_path: Path) -> List[Dict[str, str]]:
    """
    Parse the TCIA metadata.csv file.

    Args:
        metadata_path: Path to metadata.csv file

    Returns:
        List of dictionaries containing series metadata
    """
    logger.info(f"Reading metadata from: {metadata_path}")

    series_list = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            series_list.append(row)

    logger.info(f"Loaded {len(series_list)} series from metadata")
    return series_list


def get_series_uid(series: Dict[str, str]) -> str:
    """
    Get Series Instance UID to use as scan identifier.

    Args:
        series: Series metadata dictionary

    Returns:
        Series Instance UID
    """
    return series['Series UID']


def get_study_uid(series: Dict[str, str]) -> str:
    """
    Get Study Instance UID to use as session identifier.

    Args:
        series: Series metadata dictionary

    Returns:
        Study Instance UID
    """
    return series['Study UID']


def group_series_by_hierarchy(series_list: List[Dict[str, str]]) -> Dict:
    """
    Group series data into XNAT hierarchy: Subject -> Session -> Scan

    Uses:
    - Subject ID as subject identifier
    - Study UID as session identifier
    - Series UID as scan identifier

    Args:
        series_list: List of series metadata dictionaries

    Returns:
        Nested dictionary organized by subject/session/scan
    """
    logger.info("Organizing series into XNAT hierarchy")

    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for series in series_list:
        subject_id = series['Subject ID']
        study_uid = get_study_uid(series)
        series_uid = get_series_uid(series)

        # Store full series metadata for this scan
        hierarchy[subject_id][study_uid][series_uid].append(series)

    logger.info(f"Organized {len(hierarchy)} subjects")
    return hierarchy


def get_session_labels(hierarchy: Dict) -> Dict:
    """
    Create identity mapping for Study UIDs (used as-is for session labels).

    Args:
        hierarchy: Nested dict from group_series_by_hierarchy

    Returns:
        Mapping of study_uid -> study_uid (identity mapping)
    """
    logger.info("Using Study UIDs as session labels")

    session_label_map = {}

    for subject_id, sessions in hierarchy.items():
        for study_uid in sessions.keys():
            # Use Study UID directly as session label
            session_label_map[study_uid] = study_uid

    logger.info(f"Mapped {len(session_label_map)} sessions")
    return session_label_map


def copy_dicom_files(source_path: Path, dest_path: Path) -> int:
    """
    Copy DICOM files from source to destination directory.

    Args:
        source_path: Source directory containing DICOM files
        dest_path: Destination directory for DICOM files

    Returns:
        Number of files copied
    """
    if not source_path.exists():
        logger.warning(f"Source path does not exist: {source_path}")
        return 0

    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy all .dcm files
    dcm_files = list(source_path.glob('*.dcm'))

    for dcm_file in dcm_files:
        dest_file = dest_path / dcm_file.name
        shutil.copy2(dcm_file, dest_file)

    return len(dcm_files)


def create_session_metadata(sessions: Dict, session_label: str, study_uid: str) -> Dict:
    """
    Create metadata JSON for a session.

    Args:
        sessions: All sessions for a subject
        session_label: Session label (Study UID)
        study_uid: Study UID

    Returns:
        Dictionary with session metadata
    """
    scans = sessions[study_uid]
    first_series = next(iter(scans.values()))[0]

    metadata = {
        'session_label': session_label,
        'study_date': first_series['Study Date'],
        'study_description': first_series['Study Description'],
        'study_uid': study_uid,
        'modality': first_series['Modality'],
        'scans': {}
    }

    # Add scan-level metadata
    for series_uid, series_list in scans.items():
        series = series_list[0]  # Use first series for scan metadata
        metadata['scans'][series_uid] = {
            'series_description': series['Series Description'],
            'series_uid': series['Series UID'],
            'number_of_images': series['Number of Images'],
            'manufacturer': series['Manufacturer']
        }

    return metadata


def restructure_tcia_data(
    manifest_dir: Path,
    output_dir: Path,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Main function to restructure TCIA data into XNAT-ready structure.

    Args:
        manifest_dir: Path to manifest directory (contains metadata.csv and data)
        output_dir: Path to output directory (ready_for_ingest)
        dry_run: If True, don't copy files, just log what would happen

    Returns:
        Dictionary with statistics (subjects, sessions, scans, files)
    """
    logger.info("=" * 70)
    logger.info("Starting TCIA data restructuring")
    logger.info("=" * 70)

    # Paths
    metadata_path = manifest_dir / 'metadata.csv'
    data_source_dir = manifest_dir

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # Statistics
    stats = {
        'subjects': 0,
        'sessions': 0,
        'scans': 0,
        'files_copied': 0
    }

    # Step 1: Parse metadata
    series_list = parse_metadata_csv(metadata_path)

    # Step 2: Group into hierarchy
    hierarchy = group_series_by_hierarchy(series_list)
    stats['subjects'] = len(hierarchy)

    # Step 3: Get session labels (Study UIDs)
    session_label_map = get_session_labels(hierarchy)
    stats['sessions'] = len(session_label_map)

    # Step 4: Restructure files
    logger.info("=" * 70)
    logger.info("Restructuring files into XNAT structure")
    logger.info("=" * 70)

    for subject_id, sessions in hierarchy.items():
        logger.info(f"Processing subject: {subject_id}")

        subject_dir = output_dir / 'UPENN-GBM' / subject_id

        for study_uid, scans in sessions.items():
            session_label = session_label_map[study_uid]
            session_dir = subject_dir / session_label

            logger.info(f"Session: {study_uid}")

            for series_uid, series_list in scans.items():
                stats['scans'] += 1

                # Get source path from first series
                series = series_list[0]
                source_rel_path = series['File Location']

                # Convert relative path to absolute
                source_path = data_source_dir / source_rel_path.lstrip('./')

                # Destination path
                dest_path = session_dir / series_uid / 'DICOM'

                logger.info(f"Scan (Series UID): {series_uid}")
                logger.info(f"Description: {series['Series Description']}")

                if not dry_run:
                    # Copy DICOM files
                    num_files = copy_dicom_files(source_path, dest_path)
                    stats['files_copied'] += num_files
                    logger.info(f"Copied {num_files} DICOM files")
                else:
                    logger.info(f"[DRY RUN] Would copy from: {source_path}")

            # Create session metadata JSON
            if not dry_run:
                metadata = create_session_metadata(sessions, session_label, study_uid)
                metadata_file = session_dir / 'session_metadata.json'
                metadata_file.parent.mkdir(parents=True, exist_ok=True)

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

    # Summary
    logger.info("=" * 70)
    logger.info("Restructuring complete!")
    logger.info("=" * 70)
    logger.info(f"Subjects processed: {stats['subjects']}")
    logger.info(f"Sessions created: {stats['sessions']}")
    logger.info(f"Scans organized: {stats['scans']}")
    logger.info(f"Files copied: {stats['files_copied']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)

    return stats


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Reorganize TCIA data into XNAT-ready structure'
    )
    parser.add_argument(
        '--manifest-dir',
        type=Path,
        default=Path('data/manifest-1764020981210'),
        help='Path to manifest directory containing metadata.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/ready_for_ingest'),
        help='Output directory for reorganized data'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without copying files'
    )

    args = parser.parse_args()

    try:
        stats = restructure_tcia_data(
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            dry_run=args.dry_run
        )

        if args.dry_run:
            logger.info("\nThis was a dry run. Use without --dry-run to perform actual restructuring.")

    except Exception as e:
        logger.error(f"Error during restructuring: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
