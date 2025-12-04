# structuring.py

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import pydicom
from pydicom.uid import generate_uid


def sanitize_label(label: str) -> str:
    """Sanitize a label for use in file paths (remove/replace problematic characters)."""
    sanitized = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-.")
    return sanitized


def regenerate_uids(
    ds: pydicom.Dataset,
    new_study_uid: str,
    new_series_uid: str,
) -> pydicom.Dataset:
    """Regenerate Study/Series/SOP UIDs to avoid collisions on repeated uploads."""
    ds.SOPInstanceUID = generate_uid()

    if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "MediaStorageSOPInstanceUID"):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    ds.StudyInstanceUID = new_study_uid
    ds.SeriesInstanceUID = new_series_uid

    if hasattr(ds, "FrameOfReferenceUID"):
        ds.FrameOfReferenceUID = generate_uid()

    return ds


def structure_dicom_for_xnat(
    dicom_path: Union[str, Path],
    subject_label: str,
    session_label: Optional[str] = None,
    scan_number: int = 1,
    output_dir: Union[str, Path] = "data",
    create_zip: bool = False,
    validate_all: bool = True,
) -> Path:
    """
    Organize a DICOM file or directory into XNAT's pre-structured archive format:

        <subject>_archive/<subject>/<session>/scans/<scan>-<series_desc>/resources/DICOM/*.dcm

    This is best for a single series. For multi-series folders, use
    `structure_dicom_directory_for_xnat`.
    """
    dicom_path = Path(dicom_path)
    output_dir = Path(output_dir)

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM path not found: {dicom_path}")

    # Collect candidate .dcm files
    if dicom_path.is_file():
        dicom_files = [dicom_path]
    else:
        dicom_files = list(dicom_path.glob("*.dcm")) or list(dicom_path.glob("**/*.dcm"))

    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {dicom_path}")

    # Optionally validate all DICOMs and drop unreadable ones
    valid_files = []
    for f in dicom_files:
        try:
            pydicom.dcmread(str(f), stop_before_pixels=True)
            valid_files.append(f)
        except Exception as e:
            print(f"Warning: skipping unreadable DICOM {f}: {e}")

    if not valid_files:
        raise ValueError("No valid DICOM files after validation")

    dicom_files = valid_files

    # Use first DICOM for naming metadata
    ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
    series_description = getattr(ds, "SeriesDescription", "Unknown_Series")
    series_description = sanitize_label(series_description)

    if session_label is None:
        session_label = f"{subject_label}_MR1"

    subject_label_safe = sanitize_label(subject_label)
    session_label_safe = sanitize_label(session_label)
    archive_name = f"{subject_label_safe}_archive"
    scan_folder_name = f"{scan_number}-{series_description}"

    archive_base = output_dir / archive_name
    dicom_dest = (
        archive_base
        / subject_label_safe
        / session_label_safe
        / "scans"
        / scan_folder_name
        / "resources"
        / "DICOM"
    )
    dicom_dest.mkdir(parents=True, exist_ok=True)

    for idx, src_file in enumerate(sorted(dicom_files), start=1):
        dest_filename = f"{idx:04d}.dcm"
        shutil.copy2(src_file, dicom_dest / dest_filename)

    print(f"Structured {len(dicom_files)} DICOM file(s) into: {archive_base}")
    print(f"  Subject: {subject_label_safe}")
    print(f"  Session: {session_label_safe}")
    print(f"  Scan: {scan_folder_name}")

    if create_zip:
        zip_path = output_dir / f"{archive_name}.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir, archive_name)
        print(f"Created ZIP archive: {zip_path}")
        return zip_path

    return archive_base


def structure_dicom_directory_for_xnat(
    dicom_dir: Union[str, Path],
    subject_label: str,
    session_label: Optional[str] = None,
    output_dir: Union[str, Path] = "data",
    create_zip: bool = False,
    new_uids: bool = True,
) -> Path:
    """
    Auto-discover and organize all DICOM series in a directory into XNAT's
    pre-structured archive format:

        <subject>_archive/<subject>/<session>/scans/<scan>-<series_desc>/resources/DICOM/*.dcm

    Each SeriesInstanceUID becomes a separate scan.
    """
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    if not dicom_dir.is_dir():
        raise ValueError(f"Path is not a directory: {dicom_dir}")

    dicom_files = list(dicom_dir.glob("**/*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {dicom_dir}")

    print(f"Found {len(dicom_files)} DICOM files in {dicom_dir}")

    series_groups: dict[str, list[Path]] = defaultdict(list)
    series_metadata: dict[str, dict] = {}

    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            series_uid = getattr(ds, "SeriesInstanceUID", "Unknown")
            series_groups[series_uid].append(dcm_file)

            if series_uid not in series_metadata:
                series_metadata[series_uid] = {
                    "SeriesDescription": getattr(ds, "SeriesDescription", "Unknown_Series"),
                    "SeriesNumber": getattr(ds, "SeriesNumber", None),
                    "Modality": getattr(ds, "Modality", "Unknown"),
                }
        except Exception as e:
            print(f"  Warning: Could not read {dcm_file}: {e}")
            continue

    if not series_groups:
        raise ValueError("No valid DICOM files could be read")

    print(f"Found {len(series_groups)} series")

    if session_label is None:
        session_label = f"{subject_label}_MR1"

    subject_label_safe = sanitize_label(subject_label)
    session_label_safe = sanitize_label(session_label)
    archive_name = f"{subject_label_safe}_archive"
    archive_base = output_dir / archive_name

    def sort_key(item):
        uid, _ = item
        series_num = series_metadata.get(uid, {}).get("SeriesNumber")
        if series_num is not None:
            return (0, series_num, uid)
        return (1, 0, uid)

    sorted_series = sorted(series_groups.items(), key=sort_key)
    new_study_uid = generate_uid() if new_uids else None

    if new_uids:
        print("Generating new UIDs for all DICOM files...")

    for scan_number, (series_uid, files) in enumerate(sorted_series, start=1):
        metadata = series_metadata.get(series_uid, {})
        series_description = sanitize_label(metadata.get("SeriesDescription", "Unknown_Series"))
        modality = metadata.get("Modality", "Unknown")

        scan_folder_name = f"{scan_number}-{series_description}"
        dicom_dest = (
            archive_base
            / subject_label_safe
            / session_label_safe
            / "scans"
            / scan_folder_name
            / "resources"
            / "DICOM"
        )
        dicom_dest.mkdir(parents=True, exist_ok=True)

        new_series_uid = generate_uid() if new_uids else None

        for idx, src_file in enumerate(sorted(files), start=1):
            dest_filename = f"{idx:04d}.dcm"
            dest_path = dicom_dest / dest_filename

            if new_uids:
                ds = pydicom.dcmread(str(src_file))
                ds = regenerate_uids(ds, new_study_uid, new_series_uid)
                ds.save_as(str(dest_path))
            else:
                shutil.copy2(src_file, dest_path)

        print(f"  Scan {scan_number}: {scan_folder_name} ({len(files)} files, {modality})")

    print(f"\nStructured {len(sorted_series)} scan(s) into: {archive_base}")
    print(f"  Subject: {subject_label_safe}")
    print(f"  Session: {session_label_safe}")

    if create_zip:
        zip_path = output_dir / f"{archive_name}.zip"
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir, archive_name)
        print(f"Created ZIP archive: {zip_path}")
        return zip_path

    return archive_base
