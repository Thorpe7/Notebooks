# src/utils/dataframe.py

"""
Module for extracting DICOM metadata from XNAT project directories into pandas DataFrames.

This module provides utilities to walk XNAT's on-disk directory structure and extract
DICOM metadata directly from files, without requiring API calls to the XNAT server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
import pydicom


# DICOM tags commonly used for data exploration and filtering
DICOM_TAGS = {
    "PatientID": ("0010", "0020"),
    "PatientSex": ("0010", "0040"),
    "PatientAge": ("0010", "1010"),
    "StudyDate": ("0008", "0020"),
    "StudyTime": ("0008", "0030"),
    "StudyDescription": ("0008", "1030"),
    "SeriesDescription": ("0008", "103E"),
    "Modality": ("0008", "0060"),
    "BodyPartExamined": ("0018", "0015"),
    "InstitutionName": ("0008", "0080"),
    "Manufacturer": ("0008", "0070"),
    "SliceThickness": ("0018", "0050"),
    "PixelSpacing": ("0028", "0030"),
    "Rows": ("0028", "0010"),
    "Columns": ("0028", "0011"),
}


def _safe_get(ds: pydicom.dataset.Dataset, name: str) -> Optional[str]:
    """Safely extract a DICOM attribute value as a string."""
    value = getattr(ds, name, None)
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _parse_xnat_path(path: Path) -> Dict[str, Any]:
    """
    Parse XNAT project / experiment / scan info from a path like:

      /data/projects/<PROJECT_ID>/
          experiments/<EXPERIMENT_LABEL>/
              SCANS/<SCAN_DIR>/secondary/<file>.dcm

    Example:
      /data/projects/00001/experiments/22580520_BIRADS_5_2935194931/
          SCANS/1_2_826_0_1_3680043_.../secondary/22580520_...dcm

    Returns
    -------
    dict
        Dictionary with keys:
        - xnat_project_id
        - xnat_experiment_label
        - xnat_scan_dir
        - xnat_scan_id
        - xnat_scan_uid
    """
    parts = list(path.parts)
    info: Dict[str, Any] = {
        "xnat_project_id": None,
        "xnat_experiment_label": None,
        "xnat_scan_dir": None,
        "xnat_scan_id": None,
        "xnat_scan_uid": None,
    }

    # project
    if "projects" in parts:
        idx_proj = parts.index("projects")
        if idx_proj + 1 < len(parts):
            info["xnat_project_id"] = parts[idx_proj + 1]

    # experiment
    if "experiments" in parts:
        idx_exp = parts.index("experiments")
        if idx_exp + 1 < len(parts):
            info["xnat_experiment_label"] = parts[idx_exp + 1]

    # scan directory
    if "SCANS" in parts:
        idx_scans = parts.index("SCANS")
        if idx_scans + 1 < len(parts):
            scan_dir = parts[idx_scans + 1]
            info["xnat_scan_dir"] = scan_dir

            # Example scan_dir: "1_2_826_0_1_3680043_..." -> first token is scan ID
            tokens = scan_dir.split("_", 1)
            if tokens and tokens[0].isdigit():
                info["xnat_scan_id"] = int(tokens[0])
                if len(tokens) > 1:
                    info["xnat_scan_uid"] = tokens[1]
            else:
                info["xnat_scan_uid"] = scan_dir

    return info


def extract_dicom_metadata(
    dcm_path: Path,
    extra_tags: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Extract metadata from a single DICOM file.

    Parameters
    ----------
    dcm_path : Path
        Path to the DICOM file.
    extra_tags : Iterable[str], optional
        Additional DICOM attribute names to extract beyond the defaults.

    Returns
    -------
    dict
        Dictionary containing file path, XNAT hierarchy info, and DICOM metadata.
    """
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    record: Dict[str, Any] = {
        "file_path": str(dcm_path),
    }

    # XNAT-derived fields from path
    record.update(_parse_xnat_path(dcm_path))

    # Standard DICOM fields
    for name in DICOM_TAGS.keys():
        record[name] = _safe_get(ds, name)

    # Any extra fields requested
    if extra_tags:
        for name in extra_tags:
            if name in record:
                continue
            record[name] = _safe_get(ds, name)

    return record


def collect_dicom_metadata(
    root_dir: str | Path,
    extra_tags: Iterable[str] | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Walk `root_dir` recursively, collect metadata from all .dcm files,
    and return a pandas DataFrame.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to search for DICOM files (e.g., /data/projects/<PROJECT_ID>).
    extra_tags : Iterable[str], optional
        Additional DICOM attribute names to extract.
    max_files : int, optional
        Maximum number of files to process (useful for sampling/testing).

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per DICOM file containing:
        - file_path: Full path to the DICOM file
        - xnat_project_id, xnat_experiment_label, xnat_scan_dir, etc.
        - PatientID, Modality, SeriesDescription, and other DICOM tags
    """
    root = Path(root_dir)
    records: List[Dict[str, Any]] = []

    count = 0
    for dcm_path in root.rglob("*.dcm"):
        rec = extract_dicom_metadata(dcm_path, extra_tags=extra_tags)
        records.append(rec)
        count += 1
        if max_files is not None and count >= max_files:
            break

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    # Convert StudyDate to datetime for time-series analysis
    if "StudyDate" in df.columns:
        df["StudyDate"] = pd.to_datetime(df["StudyDate"], format="%Y%m%d", errors="coerce")

    return df


def build_project_dataframe(
    project_path: str | Path,
    extra_tags: Iterable[str] | None = None,
    max_files: int | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of DICOM metadata for an XNAT project.

    This is a convenience wrapper around collect_dicom_metadata that takes
    a project path directly.

    Parameters
    ----------
    project_path : str or Path
        Path to the XNAT project directory (e.g., /data/projects/00001).
    extra_tags : Iterable[str], optional
        Additional DICOM attribute names to extract.
    max_files : int, optional
        Maximum number of files to process.

    Returns
    -------
    pd.DataFrame
        DataFrame containing DICOM metadata for all files in the project.
    """
    return collect_dicom_metadata(
        root_dir=project_path,
        extra_tags=extra_tags,
        max_files=max_files,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract DICOM metadata from an XNAT project directory"
    )
    parser.add_argument(
        "project_path",
        help="Path to XNAT project directory (e.g., /data/projects/00001)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV path (optional)",
    )

    args = parser.parse_args()

    df = build_project_dataframe(
        args.project_path,
        max_files=args.max_files,
    )

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}")
    else:
        print(df.to_string())
