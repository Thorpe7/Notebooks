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

    archive_path = structure_dicom_for_xnat(
        dicom_path="data/Test_Subject_999.dcm",
        subject_label="Test_Subject_999",
        session_label="Test_Subject_999_MR1",
        output_dir="data",
    )
"""

import shutil
from pathlib import Path
from typing import Optional, Union

import pydicom


def sanitize_label(label: str) -> str:
    """Sanitize a label for use in file paths (remove/replace problematic characters)."""
    # Replace spaces and special characters with underscores
    sanitized = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
    # Remove any other problematic characters
    sanitized = "".join(c for c in sanitized if c.isalnum() or c in "_-.")
    return sanitized


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
