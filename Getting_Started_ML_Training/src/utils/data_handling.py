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


def _extract_dicom_image_metrics(scan_path: Path) -> Dict[str, Any]:
    """
    Extract image metrics from the first DICOM file found in a scan directory.

    Parameters
    ----------
    scan_path : Path
        Path to the scan directory (e.g., /data/projects/.../SCANS/1)

    Returns
    -------
    dict
        Dictionary containing image metrics extracted from DICOM headers.
    """
    metrics: Dict[str, Any] = {
        # Image dimensions
        "rows": None,
        "columns": None,
        "num_slices": None,
        # Pixel spacing and resolution
        "pixel_spacing_row": None,
        "pixel_spacing_col": None,
        "slice_thickness": None,
        "spacing_between_slices": None,
        # Image characteristics
        "bits_allocated": None,
        "bits_stored": None,
        "high_bit": None,
        "pixel_representation": None,
        "photometric_interpretation": None,
        "samples_per_pixel": None,
        # Acquisition parameters
        "kvp": None,
        "exposure": None,
        "exposure_time": None,
        "tube_current": None,
        "convolution_kernel": None,
        "reconstruction_diameter": None,
        # Window settings
        "window_center": None,
        "window_width": None,
        # Scanner info
        "manufacturer": None,
        "manufacturer_model_name": None,
        "station_name": None,
        "software_versions": None,
        # Series info
        "series_description": None,
        "series_number": None,
        "acquisition_number": None,
        "instance_number": None,
        # Patient position
        "patient_position": None,
        "image_orientation_patient": None,
        "body_part_examined": None,
    }

    # Find DICOM files in the scan directory
    dcm_files = []
    for subdir in ["DICOM", "secondary", "NIFTI", ""]:
        search_path = scan_path / subdir if subdir else scan_path
        if search_path.exists():
            dcm_files.extend(list(search_path.glob("*.dcm")))
            if dcm_files:
                break

    if not dcm_files:
        return metrics

    # Count total slices
    metrics["num_slices"] = len(dcm_files)

    # Read the first DICOM file
    try:
        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    except Exception:
        return metrics

    # Helper to safely extract values
    def safe_get(attr: str, default=None):
        val = getattr(ds, attr, default)
        if val is None:
            return default
        try:
            # Handle multi-valued attributes
            if hasattr(val, "__iter__") and not isinstance(val, str):
                return list(val) if len(val) > 1 else val[0]
            return val
        except Exception:
            return default

    # Image dimensions
    metrics["rows"] = safe_get("Rows")
    metrics["columns"] = safe_get("Columns")

    # Pixel spacing
    pixel_spacing = safe_get("PixelSpacing")
    if pixel_spacing is not None:
        if hasattr(pixel_spacing, "__iter__") and not isinstance(pixel_spacing, str):
            pixel_spacing = list(pixel_spacing)
            if len(pixel_spacing) >= 2:
                metrics["pixel_spacing_row"] = float(pixel_spacing[0])
                metrics["pixel_spacing_col"] = float(pixel_spacing[1])
        else:
            metrics["pixel_spacing_row"] = float(pixel_spacing)
            metrics["pixel_spacing_col"] = float(pixel_spacing)

    metrics["slice_thickness"] = safe_get("SliceThickness")
    metrics["spacing_between_slices"] = safe_get("SpacingBetweenSlices")

    # Image characteristics
    metrics["bits_allocated"] = safe_get("BitsAllocated")
    metrics["bits_stored"] = safe_get("BitsStored")
    metrics["high_bit"] = safe_get("HighBit")
    metrics["pixel_representation"] = safe_get("PixelRepresentation")
    metrics["photometric_interpretation"] = safe_get("PhotometricInterpretation")
    metrics["samples_per_pixel"] = safe_get("SamplesPerPixel")

    # Acquisition parameters (CT/X-ray specific)
    metrics["kvp"] = safe_get("KVP")
    metrics["exposure"] = safe_get("Exposure")
    metrics["exposure_time"] = safe_get("ExposureTime")
    metrics["tube_current"] = safe_get("XRayTubeCurrent")
    metrics["convolution_kernel"] = safe_get("ConvolutionKernel")
    metrics["reconstruction_diameter"] = safe_get("ReconstructionDiameter")

    # Window settings
    metrics["window_center"] = safe_get("WindowCenter")
    metrics["window_width"] = safe_get("WindowWidth")

    # Scanner info
    metrics["manufacturer"] = safe_get("Manufacturer")
    metrics["manufacturer_model_name"] = safe_get("ManufacturerModelName")
    metrics["station_name"] = safe_get("StationName")
    metrics["software_versions"] = safe_get("SoftwareVersions")

    # Series info
    metrics["series_description"] = safe_get("SeriesDescription")
    metrics["series_number"] = safe_get("SeriesNumber")
    metrics["acquisition_number"] = safe_get("AcquisitionNumber")
    metrics["instance_number"] = safe_get("InstanceNumber")

    # Patient position
    metrics["patient_position"] = safe_get("PatientPosition")
    iop = safe_get("ImageOrientationPatient")
    if iop is not None:
        if hasattr(iop, "__iter__") and not isinstance(iop, str):
            metrics["image_orientation_patient"] = ",".join(str(x) for x in iop)
        else:
            metrics["image_orientation_patient"] = str(iop)
    metrics["body_part_examined"] = safe_get("BodyPartExamined")

    return metrics


def fetch_xnat_metadata(
    project_ids: Union[str, List[str]],
    connection: Optional[xnat.XNATSession] = None,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    include_dicom_metrics: bool = True,
) -> pd.DataFrame:
    """
    Fetch demographic and DICOM metadata from one or more XNAT projects using XNATpy.

    This function connects to XNAT (using provided credentials or environment
    variables) and retrieves subject demographics and scan-level DICOM metadata
    for all experiments in the specified project(s). When multiple projects are
    provided, data from all projects is aggregated into a single DataFrame.

    Parameters
    ----------
    project_ids : str or list of str
        The XNAT project ID(s) to fetch metadata from. Can be a single project ID
        string or a list of project IDs to aggregate data from multiple projects.
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
    include_dicom_metrics : bool, default True
        If True, reads a sample DICOM file from each scan to extract additional
        image metrics (resolution, pixel spacing, bits, acquisition parameters, etc.).
        Set to False for faster metadata retrieval when image metrics aren't needed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing subject demographics and scan metadata with columns:

        Project fields:
        - project_id: XNAT project ID
        - project_name: Human-readable project name/title

        Subject fields:
        - subject_id: XNAT subject ID
        - subject_label: Subject label
        - gender: Subject gender (if available)
        - age: Subject age (if available)
        - handedness: Subject handedness (if available)

        Experiment fields:
        - experiment_id: XNAT experiment/session ID
        - experiment_label: Experiment label
        - experiment_date: Date of the experiment
        - modality: Imaging modality

        Scan fields:
        - scan_id: Scan ID within the experiment
        - scan_type: Type/series description of the scan
        - scan_quality: Quality rating of the scan
        - scan_note: Notes associated with the scan
        - file_path: Mounted file path to the scan's DICOM directory
        - dicom_file_count: Number of DICOM files found via XNATpy
        - dicom_files_sample: List of up to 3 sample DICOM file paths

        Image metrics (when include_dicom_metrics=True):
        - rows, columns: Image dimensions in pixels
        - num_slices: Number of DICOM files/slices in the scan
        - pixel_spacing_row, pixel_spacing_col: Pixel spacing in mm
        - slice_thickness: Slice thickness in mm
        - spacing_between_slices: Distance between slices in mm
        - bits_allocated, bits_stored, high_bit: Bit depth information
        - pixel_representation: 0=unsigned, 1=signed
        - photometric_interpretation: e.g., MONOCHROME2, RGB
        - samples_per_pixel: Number of samples per pixel
        - kvp: Peak kilovoltage (CT/X-ray)
        - exposure, exposure_time, tube_current: X-ray exposure parameters
        - convolution_kernel: Reconstruction kernel (CT)
        - reconstruction_diameter: Reconstruction diameter in mm
        - window_center, window_width: Default window settings
        - manufacturer, manufacturer_model_name: Scanner info
        - station_name, software_versions: Station info
        - series_description, series_number: Series info
        - patient_position, body_part_examined: Positioning info

    Examples
    --------
    Single project using environment variables:
    >>> df = fetch_xnat_metadata("00001")

    Multiple projects:
    >>> PROJECTS_LIST = ["00001", "00002", "RIDER-LUNG-CT"]
    >>> df = fetch_xnat_metadata(PROJECTS_LIST, connection=connection)

    Fast retrieval without DICOM metrics:
    >>> df = fetch_xnat_metadata(["00001", "00002"], include_dicom_metrics=False)

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
    # Normalize project_ids to a list
    if isinstance(project_ids, str):
        project_ids = [project_ids]

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
        records: List[Dict[str, Any]] = []

        # Iterate through all requested projects
        for project_id in project_ids:
            # Validate project exists
            if project_id not in connection.projects:
                available = list(connection.projects.keys())
                raise ValueError(
                    f"Project '{project_id}' not found. Available projects: {available}"
                )

            project = connection.projects[project_id]
            project_name = getattr(project, "name", project_id)

            # Iterate through all subjects in the project
            for subject in project.subjects.values():
                subject_data = {
                    "project_id": project_id,
                    "project_name": project_name,
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

                        # Get file paths from XNATpy resources
                        dicom_files: List[str] = []
                        scan_dir_path = None

                        # Iterate through resources (DICOM, secondary, etc.)
                        try:
                            for resource in scan.resources.values():
                                resource_label = getattr(resource, "label", "")
                                for file_obj in resource.files.values():
                                    # Get the URI which maps to the mounted path
                                    file_uri = getattr(file_obj, "uri", None)
                                    if file_uri:
                                        # Convert URI to mounted path
                                        # URI format: /data/experiments/.../scans/.../resources/.../files/...
                                        # Mounted path: /data/projects/.../experiments/.../SCANS/.../DICOM/...
                                        file_path = f"/data{file_uri}" if not file_uri.startswith("/data") else file_uri

                                        # Check if it's a DICOM file
                                        if file_path.lower().endswith(".dcm"):
                                            dicom_files.append(file_path)

                                            # Extract the scan directory from the first file
                                            if scan_dir_path is None:
                                                scan_dir_path = str(Path(file_path).parent)
                        except Exception:
                            # Fall back to constructed path if resource iteration fails
                            pass

                        # Fall back to constructed path if no files found via XNATpy
                        if scan_dir_path is None:
                            scan_dir_path = (
                                f"/data/projects/{project_id}/experiments/"
                                f"{experiment.label}/SCANS/{scan.id}"
                            )

                        scan_record["file_path"] = scan_dir_path
                        scan_record["dicom_file_count"] = len(dicom_files) if dicom_files else None

                        # Store first few DICOM file paths for reference
                        if dicom_files:
                            scan_record["dicom_files_sample"] = dicom_files[:3]

                        # Extract DICOM image metrics if requested
                        if include_dicom_metrics:
                            dicom_metrics = _extract_dicom_image_metrics(Path(scan_dir_path))
                            scan_record.update(dicom_metrics)

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

        # Reorder columns to put project info first
        cols = df.columns.tolist()
        priority_cols = ["project_id", "project_name"]
        other_cols = [c for c in cols if c not in priority_cols]
        df = df[priority_cols + other_cols]

        return df

    finally:
        if close_connection and connection is not None:
            connection.disconnect()
