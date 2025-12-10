# src/utils/data_handling.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Union

import pandas as pd
import pydicom
import xnat


# DICOM tags we care about for dashboarding
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
}


def _safe_get(ds: pydicom.dataset.Dataset, name: str) -> Optional[str]:
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
    """Extract a single-row dict of metadata from one DICOM file."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)

    record: Dict[str, Any] = {
        "file_path": str(dcm_path),
    }

    # XNAT-derived fields
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

    # Optional: convert StudyDate to datetime for time-series plots
    if "StudyDate" in df.columns:
        df["StudyDate"] = pd.to_datetime(df["StudyDate"], format="%Y%m%d", errors="coerce")

    return df


def fetch_xnat_metadata(
    project_id: str,
    connection: Optional[xnat.XNATSession] = None,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch demographic and DICOM metadata from an XNAT project using XNATpy.

    This function connects to XNAT (using provided credentials or environment
    variables) and retrieves subject demographics and scan-level DICOM metadata
    for all experiments in the specified project.

    Parameters
    ----------
    project_id : str
        The XNAT project ID to fetch metadata from.
    connection : xnat.XNATSession, optional
        An existing XNATpy connection. If provided, this connection will be used
        instead of creating a new one. The connection will NOT be closed by this
        function when passed in.
    host : str, optional
        XNAT server URL. Defaults to XNAT_HOST environment variable.
    user : str, optional
        XNAT username. Defaults to XNAT_USER environment variable.
    password : str, optional
        XNAT password. Defaults to XNAT_PASS environment variable.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing subject demographics and scan metadata with columns:
        - subject_id: XNAT subject ID
        - subject_label: Subject label
        - gender: Subject gender (if available)
        - age: Subject age (if available)
        - handedness: Subject handedness (if available)
        - experiment_id: XNAT experiment/session ID
        - experiment_label: Experiment label
        - experiment_date: Date of the experiment
        - modality: Imaging modality
        - scan_id: Scan ID within the experiment
        - scan_type: Type/series description of the scan
        - scan_quality: Quality rating of the scan
        - scan_note: Notes associated with the scan
        - file_path: Mounted file path to the DICOM data

    Examples
    --------
    Using environment variables (recommended for XNAT Jupyter):
    >>> df = fetch_xnat_metadata("00001")

    Using an existing connection:
    >>> conn = xnat.connect(host, user=user, password=password)
    >>> df = fetch_xnat_metadata("00001", connection=conn)
    >>> conn.disconnect()

    Using explicit credentials:
    >>> df = fetch_xnat_metadata(
    ...     "00001",
    ...     host="https://xnat.example.com",
    ...     user="myuser",
    ...     password="mypass"
    ... )
    """
    # Determine whether we need to manage the connection
    close_connection = False

    if connection is None:
        # Use provided credentials or fall back to environment variables
        host = host or os.environ.get("XNAT_HOST")
        user = user or os.environ.get("XNAT_USER")
        password = password or os.environ.get("XNAT_PASS")

        if not all([host, user, password]):
            raise ValueError(
                "XNAT credentials not provided. Either pass host/user/password "
                "parameters or set XNAT_HOST, XNAT_USER, XNAT_PASS environment variables."
            )

        connection = xnat.connect(host, user=user, password=password)
        close_connection = True

    try:
        # Validate project exists
        if project_id not in connection.projects:
            available = list(connection.projects.keys())
            raise ValueError(
                f"Project '{project_id}' not found. Available projects: {available}"
            )

        project = connection.projects[project_id]
        records: List[Dict[str, Any]] = []

        # Iterate through all subjects in the project
        for subject in project.subjects.values():
            subject_data = {
                "subject_id": subject.id,
                "subject_label": subject.label,
                "gender": getattr(subject, "gender", None),
                "age": getattr(subject, "age", None),
                "handedness": getattr(subject, "handedness", None),
            }

            # Iterate through experiments/sessions for this subject
            for experiment in subject.experiments.values():
                experiment_data = {
                    **subject_data,
                    "experiment_id": experiment.id,
                    "experiment_label": experiment.label,
                    "experiment_date": getattr(experiment, "date", None),
                    "modality": getattr(experiment, "modality", None),
                }

                # Iterate through scans in this experiment
                for scan in experiment.scans.values():
                    scan_record = {
                        **experiment_data,
                        "scan_id": scan.id,
                        "scan_type": getattr(scan, "type", None),
                        "scan_quality": getattr(scan, "quality", None),
                        "scan_note": getattr(scan, "note", None),
                    }

                    # Build the mounted file path
                    # Standard XNAT mount path structure
                    file_path = (
                        f"/data/projects/{project_id}/experiments/"
                        f"{experiment.label}/SCANS/{scan.id}"
                    )
                    scan_record["file_path"] = file_path

                    records.append(scan_record)

        # Create DataFrame
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)

        # Convert date column if present
        if "experiment_date" in df.columns:
            df["experiment_date"] = pd.to_datetime(
                df["experiment_date"], errors="coerce"
            )

        return df

    finally:
        if close_connection and connection is not None:
            connection.disconnect()
