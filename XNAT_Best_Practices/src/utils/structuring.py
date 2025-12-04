"""
Organize DICOM files into the structured archive format expected by XNAT.

XNAT expects the following hierarchy for pre-structured archives:
    subject_label/
    └── session_label/
        └── scans/
            └── {scan_number}-{series_description}/
                └── resources/
                    └── DICOM/
                        ├── 0001.dcm
                        ├── 0002.dcm
                        └── ...

Usage:
    from src.utils.structuring import structure_dicom_for_xnat

    # For a directory with DICOMs (like example_subject):
    archive_path = structure_dicom_for_xnat(
        dicom_path="example_subject",
        subject_label="Example_Subject_001",
        session_label="Example_Subject_001_MR1",
        output_dir="data",
        create_zip=True,
    )
"""

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import pydicom
from pydicom.uid import generate_uid


def sanitize_label(label: str) -> str:
    """Sanitize a label for use in file paths (remove/replace problematic characters)."""
    # Replace spaces and special characters with underscores
    sanitized = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    # Remove any other problematic characters
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-.")
    return sanitized


def regenerate_uids(
    ds: pydicom.Dataset,
    new_study_uid: str,
    new_series_uid: str,
) -> pydicom.Dataset:
    """
    Regenerate Study and Series Instance UIDs for a DICOM dataset.

    This function updates the UIDs in a DICOM dataset to new values, which is useful
    when uploading example data multiple times to avoid conflicts with existing data.

    Parameters
    ----------
    ds : pydicom.Dataset
        The DICOM dataset to modify.
    new_study_uid : str
        The new Study Instance UID to assign.
    new_series_uid : str
        The new Series Instance UID to assign.

    Returns
    -------
    pydicom.Dataset
        The modified dataset with new UIDs.
    """
    # Generate a new SOP Instance UID for each file (must be unique per instance)
    ds.SOPInstanceUID = generate_uid()

    # Update the MediaStorageSOPInstanceUID in file meta to match
    if hasattr(ds, "file_meta") and hasattr(ds.file_meta, "MediaStorageSOPInstanceUID"):
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

    # Update Study Instance UID
    ds.StudyInstanceUID = new_study_uid

    # Update Series Instance UID
    ds.SeriesInstanceUID = new_series_uid

    # Update Frame of Reference UID if present (should be consistent within a series)
    if hasattr(ds, "FrameOfReferenceUID"):
        # Generate a deterministic UID based on series UID for consistency within series
        ds.FrameOfReferenceUID = generate_uid()

    return ds


def structure_dicom_for_xnat(
    dicom_path: Union[str, Path],
    subject_label: str,
    session_label: Optional[str] = None,
    scan_number: int = 1,
    output_dir: Union[str, Path] = "data",
    create_zip: bool = False,
) -> Path:
    """
    Organize a DICOM file (or directory of DICOMs) into XNAT's expected archive structure.

    Parameters
    ----------
    dicom_path : str or Path
        Path to a single DICOM file or a directory containing DICOM files.
        If a directory, all .dcm files will be included in the same scan.
    subject_label : str
        The subject/patient label to use in the archive structure.
    session_label : str, optional
        The session/experiment label. If not provided, defaults to "{subject_label}_session".
    scan_number : int
        The scan number to use in the directory name. Default is 1.
    output_dir : str or Path
        Base directory where the structured archive will be created.
        Default is "data".
    create_zip : bool
        If True, also create a ZIP archive of the structured directory.
        Default is False.

    Returns
    -------
    Path
        Path to the created archive directory (or ZIP file if create_zip=True).

    Example
    -------
    >>> from src.utils.structuring import structure_dicom_for_xnat
    >>> archive_path = structure_dicom_for_xnat(
    ...     dicom_path="data/Test_Subject_999.dcm",
    ...     subject_label="Test_Subject_999",
    ...     session_label="Test_Subject_999_MR1",
    ... )
    >>> print(archive_path)
    data/Test_Subject_999_archive/Test_Subject_999/Test_Subject_999_MR1/scans/1-Synthetic_T1W_Brain/resources/DICOM
    """
    dicom_path = Path(dicom_path)
    output_dir = Path(output_dir)

    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM path not found: {dicom_path}")

    # Collect DICOM files
    if dicom_path.is_file():
        dicom_files = [dicom_path]
    else:
        dicom_files = list(dicom_path.glob("*.dcm"))
        if not dicom_files:
            dicom_files = list(dicom_path.glob("**/*.dcm"))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in: {dicom_path}")

    # Read the first DICOM to extract metadata for naming
    ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)

    # Extract series description for scan folder naming
    series_description = getattr(ds, "SeriesDescription", "Unknown_Series")
    series_description = sanitize_label(series_description)

    # Set default session label if not provided
    if session_label is None:
        session_label = f"{subject_label}_session"

    # Sanitize labels
    subject_label_safe = sanitize_label(subject_label)
    session_label_safe = sanitize_label(session_label)

    # Build the XNAT archive directory structure
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

    # Create the directory structure
    dicom_dest.mkdir(parents=True, exist_ok=True)

    # Copy DICOM files with sequential naming
    for idx, src_file in enumerate(sorted(dicom_files), start=1):
        dest_filename = f"{idx:04d}.dcm"
        dest_path = dicom_dest / dest_filename
        shutil.copy2(src_file, dest_path)

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


def structure_multiple_series(
    dicom_paths: list[Union[str, Path]],
    subject_label: str,
    session_label: Optional[str] = None,
    output_dir: Union[str, Path] = "data",
    create_zip: bool = False,
) -> Path:
    """
    Organize multiple DICOM series into a single XNAT archive structure.

    Each item in dicom_paths will become a separate scan within the same session.

    Parameters
    ----------
    dicom_paths : list of str or Path
        List of paths to DICOM files or directories. Each path becomes a separate scan.
    subject_label : str
        The subject/patient label.
    session_label : str, optional
        The session/experiment label. Defaults to "{subject_label}_session".
    output_dir : str or Path
        Base directory for the archive.
    create_zip : bool
        If True, create a ZIP archive.

    Returns
    -------
    Path
        Path to the created archive directory (or ZIP file).
    """
    if not dicom_paths:
        raise ValueError("No DICOM paths provided")

    output_dir = Path(output_dir)

    if session_label is None:
        session_label = f"{subject_label}_session"

    subject_label_safe = sanitize_label(subject_label)
    session_label_safe = sanitize_label(session_label)
    archive_name = f"{subject_label_safe}_archive"
    archive_base = output_dir / archive_name

    for scan_number, dicom_path in enumerate(dicom_paths, start=1):
        dicom_path = Path(dicom_path)

        if not dicom_path.exists():
            raise FileNotFoundError(f"DICOM path not found: {dicom_path}")

        # Collect DICOM files for this scan
        if dicom_path.is_file():
            dicom_files = [dicom_path]
        else:
            dicom_files = list(dicom_path.glob("*.dcm"))
            if not dicom_files:
                dicom_files = list(dicom_path.glob("**/*.dcm"))
            if not dicom_files:
                raise ValueError(f"No DICOM files found in: {dicom_path}")

        # Read first DICOM to get series description
        ds = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
        series_description = getattr(ds, "SeriesDescription", "Unknown_Series")
        series_description = sanitize_label(series_description)

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

        for idx, src_file in enumerate(sorted(dicom_files), start=1):
            dest_filename = f"{idx:04d}.dcm"
            shutil.copy2(src_file, dicom_dest / dest_filename)

        print(f"  Scan {scan_number}: {scan_folder_name} ({len(dicom_files)} files)")

    print(f"Structured {len(dicom_paths)} scan(s) into: {archive_base}")

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
    Auto-discover and organize all DICOM series from a directory into XNAT archive format.

    This function scans a directory (recursively) for DICOM files, groups them by
    SeriesInstanceUID, and creates the proper XNAT archive structure with each
    series as a separate scan.

    Parameters
    ----------
    dicom_dir : str or Path
        Path to a directory containing DICOM files (can be nested).
    subject_label : str
        The subject/patient label to use in the archive structure.
    session_label : str, optional
        The session/experiment label. If not provided, defaults to "{subject_label}_MR1".
    output_dir : str or Path
        Base directory where the structured archive will be created.
        Default is "data".
    create_zip : bool
        If True, also create a ZIP archive of the structured directory.
        Default is False.
    new_uids : bool
        If True (default), generate new Study and Series Instance UIDs for all files.
        This prevents conflicts when uploading the same example data multiple times.

    Returns
    -------
    Path
        Path to the created archive directory (or ZIP file if create_zip=True).

    Example
    -------
    >>> from src.utils.structuring import structure_dicom_directory_for_xnat
    >>> archive_path = structure_dicom_directory_for_xnat(
    ...     dicom_dir="example_subject",
    ...     subject_label="Example_Subject_001",
    ...     session_label="Example_Subject_001_MR1",
    ...     output_dir="data",
    ...     create_zip=True,
    ... )
    """
    dicom_dir = Path(dicom_dir)
    output_dir = Path(output_dir)

    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    if not dicom_dir.is_dir():
        raise ValueError(f"Path is not a directory: {dicom_dir}")

    # Find all DICOM files recursively
    dicom_files = list(dicom_dir.glob("**/*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in: {dicom_dir}")

    print(f"Found {len(dicom_files)} DICOM files in {dicom_dir}")

    # Group files by SeriesInstanceUID
    series_groups: dict[str, list[Path]] = defaultdict(list)
    series_metadata: dict[str, dict] = {}

    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            series_uid = getattr(ds, "SeriesInstanceUID", "Unknown")

            series_groups[series_uid].append(dcm_file)

            # Store metadata from first file of each series
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

    # Set default session label
    if session_label is None:
        session_label = f"{subject_label}_MR1"

    # Sanitize labels
    subject_label_safe = sanitize_label(subject_label)
    session_label_safe = sanitize_label(session_label)
    archive_name = f"{subject_label_safe}_archive"
    archive_base = output_dir / archive_name

    # Sort series by SeriesNumber if available, otherwise by UID
    def sort_key(item):
        uid, _ = item
        series_num = series_metadata.get(uid, {}).get("SeriesNumber")
        if series_num is not None:
            return (0, series_num, uid)
        return (1, 0, uid)

    sorted_series = sorted(series_groups.items(), key=sort_key)

    # Generate new Study UID if requested (same for all series in this session)
    new_study_uid = generate_uid() if new_uids else None

    if new_uids:
        print("Generating new UIDs for all DICOM files...")

    # Create archive structure for each series
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

        # Generate new Series UID for this series if requested
        new_series_uid = generate_uid() if new_uids else None

        # Copy/write files with sequential naming
        for idx, src_file in enumerate(sorted(files), start=1):
            dest_filename = f"{idx:04d}.dcm"
            dest_path = dicom_dest / dest_filename

            if new_uids:
                # Read, modify UIDs, and write the DICOM file
                ds = pydicom.dcmread(str(src_file))
                ds = regenerate_uids(ds, new_study_uid, new_series_uid)
                ds.save_as(str(dest_path))
            else:
                # Just copy the file as-is
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
