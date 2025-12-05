# src/utils/dicom_metadata.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
import pydicom


# DICOM tags we care about for dashboarding
# (add/remove fields as needed)
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
    """Return str(ds.<name>) or None if missing."""
    value = getattr(ds, name, None)
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _parse_xnat_path(path: Path) -> Dict[str, Any]:
    """
    Parse subject/session/scan info from an XNAT archive-style path.

    Expected pattern (example):
      <root>/<subject_archive>/<subject_label>/<session_label>/
          scans/<scan_dir>/resources/DICOM/<file>.dcm

    We locate the 'scans' component and work backwards from there.
    """
    parts = list(path.parts)
    info: Dict[str, Any] = {
        "xnat_subject_label": None,
        "xnat_session_label": None,
        "xnat_scan_dir": None,
        "xnat_scan_id": None,
        "xnat_scan_description": None,
    }

    if "scans" in parts:
        idx = parts.index("scans")
        if idx + 1 < len(parts):
            scan_dir = parts[idx + 1]  # e.g. '1-ep2d_perf_12_CC_BOLUS'
            info["xnat_scan_dir"] = scan_dir
            # split leading numeric ID from description if present
            if "-" in scan_dir:
                first, *rest = scan_dir.split("-", 1)
                if first.isdigit():
                    info["xnat_scan_id"] = int(first)
                    info["xnat_scan_description"] = rest[0] if rest else None
                else:
                    info["xnat_scan_description"] = scan_dir

        if idx - 1 >= 0:
            info["xnat_session_label"] = parts[idx - 1]  # e.g. 'Test_Subject_999_MR1'
        if idx - 2 >= 0:
            info["xnat_subject_label"] = parts[idx - 2]  # e.g. 'Test_Subject_999'

    return info


def extract_dicom_metadata(
    dcm_path: Path,
    extra_tags: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Extract a single-row dict of metadata from one DICOM file.

    Parameters
    ----------
    dcm_path : Path
        Path to a .dcm file.
    extra_tags : iterable of attribute names, optional
        Additional pydicom attribute names to pull out, if needed.

    Returns
    -------
    dict
        Dictionary with core DICOM tags + parsed XNAT path fields + file_path.
    """
    # Only read headers, skip pixel data to keep it fast
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

    Parameters
    ----------
    root_dir : str or Path
        Root directory of the XNAT archive (or any DICOM tree).
    extra_tags : iterable of attribute names, optional
        Extra pydicom Dataset attributes to include.
    max_files : int, optional
        If not None, stop after this many files (useful for quick tests).

    Returns
    -------
    pandas.DataFrame
        One row per DICOM file with XNAT path fields + DICOM metadata.
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
