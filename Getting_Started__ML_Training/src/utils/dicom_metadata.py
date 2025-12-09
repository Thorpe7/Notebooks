# src/utils/dicom_metadata.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
import pydicom


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
