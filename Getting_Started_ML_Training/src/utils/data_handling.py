# src/utils/data_handling.py

from __future__ import annotations

import io
import json
import os
from datetime import datetime
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


def _convert_dicom_name_to_snake_case(name: str) -> str:
    """Convert DICOM attribute name (CamelCase) to snake_case for DataFrame columns."""
    import re
    # Insert underscore before uppercase letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _extract_dicom_value(element) -> Any:
    """
    Safely extract a value from a DICOM element for DataFrame storage.

    Handles sequences, multi-valued elements, and special types.
    """
    if element is None:
        return None

    try:
        value = element.value

        # Skip empty values
        if value is None or value == "":
            return None

        # Handle sequences - extract key info or skip
        if element.VR == "SQ":
            # For sequences, try to extract first item's key values
            if len(value) > 0:
                # Return count of items in sequence
                return f"[Sequence: {len(value)} items]"
            return None

        # Handle byte arrays (pixel data, etc.)
        if isinstance(value, bytes):
            return f"[Binary: {len(value)} bytes]"

        # Handle multi-valued elements
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value_list = list(value)
            if len(value_list) == 0:
                return None
            elif len(value_list) == 1:
                return _safe_convert_value(value_list[0])
            else:
                # Join multi-valued items with backslash (DICOM convention)
                return "\\".join(str(_safe_convert_value(v)) for v in value_list)

        return _safe_convert_value(value)
    except Exception:
        return None


def _safe_convert_value(value) -> Any:
    """Convert DICOM value to a Python-native type safe for DataFrames."""
    if value is None:
        return None

    # Handle pydicom special types
    type_name = type(value).__name__

    if type_name in ('DSfloat', 'DSdecimal', 'IS'):
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)
    elif type_name == 'PersonName':
        return str(value)
    elif type_name == 'DA':  # Date
        return str(value)
    elif type_name == 'TM':  # Time
        return str(value)
    elif type_name == 'DT':  # DateTime
        return str(value)
    elif type_name == 'UID':
        return str(value)
    elif isinstance(value, (int, float, str, bool)):
        return value
    else:
        try:
            return str(value)
        except Exception:
            return None


def _discover_dicom_tags(ds: pydicom.Dataset) -> Dict[str, str]:
    """
    Discover all DICOM tags from a dataset.

    Returns a dict mapping snake_case column names to DICOM attribute names.
    """
    # Tags to exclude (pixel data, large binary data, internal tags)
    exclude_keywords = {
        'PixelData', 'pixel_data',
        'OverlayData', 'overlay_data',
        'EncapsulatedDocument', 'encapsulated_document',
        'SpectroscopyData', 'spectroscopy_data',
        'AudioSampleData', 'audio_sample_data',
        'CurveData', 'curve_data',
        'ChannelSensitivityCorrectionFactor',
        'WaveformData', 'waveform_data',
        'FloatPixelData', 'float_pixel_data',
        'DoubleFloatPixelData', 'double_float_pixel_data',
        'RedPaletteColorLookupTableData',
        'GreenPaletteColorLookupTableData',
        'BluePaletteColorLookupTableData',
    }

    tags = {}

    for elem in ds:
        # Skip group length elements
        if elem.tag.element == 0:
            continue

        # Get keyword (attribute name)
        keyword = elem.keyword
        if not keyword or keyword in exclude_keywords:
            continue

        # Skip private tags (odd group numbers)
        if elem.tag.group % 2 == 1:
            continue

        # Convert to snake_case for column name
        col_name = _convert_dicom_name_to_snake_case(keyword)

        # Skip if already have this column
        if col_name in tags:
            continue

        tags[col_name] = keyword

    # Also check file_meta if available
    if hasattr(ds, 'file_meta'):
        for elem in ds.file_meta:
            if elem.tag.element == 0:
                continue
            keyword = elem.keyword
            if not keyword or keyword in exclude_keywords:
                continue
            col_name = _convert_dicom_name_to_snake_case(keyword)
            if col_name not in tags:
                tags[col_name] = keyword

    return tags


def _read_dicom_header_from_xnat(
    file_obj,
    verbose: bool = False
) -> Optional[pydicom.Dataset]:
    """
    Read DICOM header from XNATpy file object using streaming.

    Uses file_obj.open() with pydicom's stop_before_pixels=True to read
    only the DICOM header without downloading the entire file.

    Parameters
    ----------
    file_obj : XNATpy file object
        The file object from scan.resources[].files[]
    verbose : bool
        If True, print status messages

    Returns
    -------
    pydicom.Dataset or None
        The DICOM dataset (header only) if successful, None otherwise.
    """
    # Use file.open() streaming with pydicom (confirmed working method)
    if hasattr(file_obj, 'open'):
        try:
            with file_obj.open() as fin:
                ds = pydicom.dcmread(fin, stop_before_pixels=True)
            if verbose:
                print(f"    Read DICOM header via file.open() streaming")
            return ds
        except Exception as e:
            if verbose:
                print(f"    file.open() failed: {e}")

    return None


def _extract_dicom_image_metrics_from_xnat(
    scan,
    known_tags: Dict[str, str],
    seen_modalities: set,
    discovery_log: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False,
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Extract image metrics from the first DICOM file in an XNATpy scan object.

    Uses XNATpy's file.open() streaming with pydicom's stop_before_pixels=True
    to read only the DICOM header without downloading the entire file.

    Dynamically discovers DICOM tags based on modality/scan type. For new
    modalities, all available tags are discovered and added to known_tags.

    Parameters
    ----------
    scan : xnat scan object
        An XNATpy scan object from which to extract DICOM metadata.
    known_tags : Dict[str, str]
        Dictionary mapping column names to DICOM keywords. Updated in place
        when new modalities are discovered.
    seen_modalities : set
        Set of modalities/scan types already processed. Updated in place.
    discovery_log : list, optional
        If provided, appends discovery events with scan type info, paths,
        and new headers found.
    verbose : bool, default False
        If True, prints status messages during extraction.

    Returns
    -------
    tuple[Dict[str, Any], Dict[str, str]]
        Tuple of (metrics dict, updated known_tags dict)
    """
    # Initialize metrics with None for all known tags
    metrics: Dict[str, Any] = {col: None for col in known_tags}
    metrics["num_slices"] = None

    # Find DICOM files via XNATpy resources
    dicom_file = None
    dicom_file_count = 0
    dicom_file_id = None
    dicom_file_uri = None
    resource_label_used = None

    try:
        # Iterate through resources (DICOM, secondary, etc.)
        for resource in scan.resources.values():
            resource_label = getattr(resource, "label", "")
            resource_format = getattr(resource, "format", "")

            # Check if this is a DICOM resource
            is_dicom_resource = resource_label.upper() in ["DICOM", "SECONDARY"] or \
                                resource_format.upper() == "DICOM"

            for file_key, file_obj in resource.files.items():
                # Get file identifier from id, path, or key
                file_id = getattr(file_obj, "id", None) or \
                          getattr(file_obj, "path", None) or \
                          str(file_key)

                # Check if it's a DICOM file
                is_dicom_file = file_id.lower().endswith(".dcm") or is_dicom_resource

                if is_dicom_file:
                    dicom_file_count += 1
                    # Keep the first DICOM file for metadata extraction
                    if dicom_file is None:
                        dicom_file = file_obj
                        dicom_file_id = file_id
                        resource_label_used = resource_label
                        dicom_file_uri = getattr(file_obj, 'uri', None) or \
                                         getattr(file_obj, 'path', None) or \
                                         f"{resource_label}/{file_id}"

    except Exception as e:
        if verbose:
            print(f"  [ERROR] Failed to enumerate resources for scan {getattr(scan, 'id', '?')}: {e}")
        if discovery_log is not None:
            discovery_log.append({
                "event": "error",
                "scan_id": getattr(scan, 'id', None),
                "error_type": "resource_enumeration",
                "error_message": str(e),
            })
        return metrics, known_tags

    if dicom_file is None:
        if verbose:
            print(f"  [WARN] No DICOM files found for scan {getattr(scan, 'id', '?')}")
        return metrics, known_tags

    # Record the file count
    metrics["num_slices"] = dicom_file_count

    # Read DICOM header using streaming (no full file download)
    ds = _read_dicom_header_from_xnat(dicom_file, verbose=verbose)

    if ds is None:
        if verbose:
            print(f"  [ERROR] Failed to read DICOM header from {dicom_file_uri}")
        if discovery_log is not None:
            discovery_log.append({
                "event": "error",
                "scan_id": getattr(scan, 'id', None),
                "error_type": "read_header_failed",
                "file_uri": dicom_file_uri,
                "file_id": dicom_file_id,
                "error_message": "file.open() streaming failed",
            })
        return metrics, known_tags

    # Determine modality/scan type for tag discovery
    modality = getattr(ds, 'Modality', 'UNKNOWN')
    sop_class_uid = getattr(ds, 'SOPClassUID', '')
    sop_class_name = str(sop_class_uid.name) if hasattr(sop_class_uid, 'name') else str(sop_class_uid)
    scan_type_key = f"{modality}_{sop_class_uid}"

    # If this is a new modality/scan type, discover all available tags
    if scan_type_key not in seen_modalities:
        seen_modalities.add(scan_type_key)
        new_tags = _discover_dicom_tags(ds)

        # Track which tags are actually new
        newly_added_tags = []
        for col_name, keyword in new_tags.items():
            if col_name not in known_tags:
                known_tags[col_name] = keyword
                metrics[col_name] = None  # Initialize in current metrics
                newly_added_tags.append(col_name)

        # Log the discovery
        if verbose:
            print(f"\n{'='*60}")
            print(f"NEW SCAN TYPE DISCOVERED")
            print(f"{'='*60}")
            print(f"  Modality:       {modality}")
            print(f"  SOP Class:      {sop_class_name}")
            print(f"  SOP Class UID:  {sop_class_uid}")
            print(f"  File ID:        {dicom_file_id}")
            print(f"  File URI:       {dicom_file_uri}")
            print(f"  Resource:       {resource_label_used}")
            print(f"  New headers ({len(newly_added_tags)}): {newly_added_tags[:20]}")
            if len(newly_added_tags) > 20:
                print(f"                  ... and {len(newly_added_tags) - 20} more")
            print(f"{'='*60}\n")

        if discovery_log is not None:
            discovery_log.append({
                "event": "new_scan_type",
                "modality": modality,
                "sop_class_name": sop_class_name,
                "sop_class_uid": str(sop_class_uid),
                "scan_type_key": scan_type_key,
                "file_id": dicom_file_id,
                "file_uri": dicom_file_uri,
                "resource_label": resource_label_used,
                "new_headers_count": len(newly_added_tags),
                "new_headers": newly_added_tags,
                "total_known_tags": len(known_tags),
            })

    # Extract values for all known tags
    for col_name, keyword in known_tags.items():
        if col_name == "num_slices":
            continue  # Already set

        # Try to get from main dataset
        if hasattr(ds, keyword):
            elem = ds.data_element(keyword)
            if elem is not None:
                metrics[col_name] = _extract_dicom_value(elem)

        # Try file_meta for certain tags
        elif hasattr(ds, 'file_meta') and hasattr(ds.file_meta, keyword):
            elem = ds.file_meta.data_element(keyword)
            if elem is not None:
                metrics[col_name] = _extract_dicom_value(elem)

    # Handle special cases for pixel spacing (split into row/col)
    if "pixel_spacing" in metrics and metrics["pixel_spacing"]:
        ps = metrics["pixel_spacing"]
        if isinstance(ps, str) and "\\" in ps:
            parts = ps.split("\\")
            if len(parts) >= 2:
                try:
                    metrics["pixel_spacing_row"] = float(parts[0])
                    metrics["pixel_spacing_col"] = float(parts[1])
                except ValueError:
                    pass
        # Add these to known_tags if not present
        if "pixel_spacing_row" not in known_tags:
            known_tags["pixel_spacing_row"] = "_derived_pixel_spacing_row"
            known_tags["pixel_spacing_col"] = "_derived_pixel_spacing_col"

    return metrics, known_tags


def fetch_xnat_metadata(
    project_ids: Union[str, List[str]],
    connection: Optional[xnat.XNATSession] = None,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    include_dicom_metrics: bool = True,
    save_csv: bool = False,
    csv_filename: Optional[str] = None,
    save_dir: str = "logs/saved_df",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetch demographic and DICOM metadata from one or more XNAT projects using XNATpy.

    This function connects to XNAT (using provided credentials or environment
    variables) and retrieves subject demographics and scan-level DICOM metadata
    for all experiments in the specified project(s). When multiple projects are
    provided, data from all projects is aggregated into a single DataFrame.

    Supports caching: if save_csv=True and csv_filename is provided, the function
    will first check if the CSV already exists and load it instead of fetching
    from XNAT. This is useful for large datasets that take time to fetch.

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
        If True, downloads a sample DICOM file from each scan via XNATpy to extract
        additional image metrics (resolution, pixel spacing, bits, acquisition
        parameters, etc.). Does not require mounted filesystem access.
        Set to False for faster metadata retrieval when image metrics aren't needed.
    save_csv : bool, default False
        If True and csv_filename is provided, saves the DataFrame to a CSV file.
        On subsequent calls, if the CSV exists, it will be loaded instead of
        fetching data from XNAT.
    csv_filename : str, optional
        Name of the CSV file (with or without .csv extension). Required if
        save_csv=True. Example: "my_metadata" or "my_metadata.csv"
    save_dir : str, default "logs/saved_df"
        Directory where the CSV file will be saved/loaded from.
    verbose : bool, default False
        If True, prints detailed status messages during DICOM metadata extraction,
        including when new scan types are discovered. Also saves a JSON log file
        with discovery details (scan types found, file URIs used, headers discovered).

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

        Image metrics (when include_dicom_metrics=True, fetched via XNATpy):
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

    Save to CSV and load from cache on subsequent calls:
    >>> df = fetch_xnat_metadata(
    ...     ["00001", "00002"],
    ...     connection=connection,
    ...     save_csv=True,
    ...     csv_filename="multi_project_metadata"
    ... )

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
    # Handle CSV caching
    csv_path = None
    if save_csv and csv_filename:
        # Ensure filename has .csv extension
        if not csv_filename.endswith(".csv"):
            csv_filename = f"{csv_filename}.csv"

        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        csv_path = save_path / csv_filename

        # Check if cached CSV exists
        if csv_path.exists():
            print(f"Loading cached metadata from: {csv_path}")
            df = pd.read_csv(csv_path)

            # Convert date column if present
            if "experiment_date" in df.columns:
                df["experiment_date"] = pd.to_datetime(
                    df["experiment_date"], errors="coerce"
                )

            return df

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

        # Dynamic tag discovery: track known tags and seen modalities
        known_tags: Dict[str, str] = {
            "num_slices": "_derived_num_slices",
        }
        seen_modalities: set = set()

        # Discovery log for tracking new scan types and metadata headers
        discovery_log: List[Dict[str, Any]] = []

        if verbose:
            print(f"\n{'#'*60}")
            print(f"# XNAT METADATA EXTRACTION - VERBOSE MODE")
            print(f"# Projects: {project_ids}")
            print(f"{'#'*60}\n")

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

                        # Extract DICOM image metrics if requested (uses XNATpy directly)
                        if include_dicom_metrics:
                            dicom_metrics, known_tags = _extract_dicom_image_metrics_from_xnat(
                                scan,
                                known_tags,
                                seen_modalities,
                                discovery_log=discovery_log,
                                verbose=verbose,
                            )
                            scan_record.update(dicom_metrics)

                        records.append(scan_record)

        # Create DataFrame
        if not records:
            return pd.DataFrame()

        # Normalize records: ensure all records have all discovered columns
        # (earlier records may be missing columns discovered later)
        if include_dicom_metrics:
            all_columns = set()
            for record in records:
                all_columns.update(record.keys())

            for record in records:
                for col in all_columns:
                    if col not in record:
                        record[col] = None

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

        # Save to CSV if requested
        if csv_path is not None:
            df.to_csv(csv_path, index=False)
            print(f"Saved metadata to: {csv_path}")

        # Save discovery log if there are entries
        if discovery_log and (verbose or save_csv):
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = csv_filename.replace(".csv", "") if csv_filename else "metadata"
            log_path = save_path / f"{log_filename}_discovery_log_{timestamp}.json"

            # Add summary to the log
            discovery_summary = {
                "extraction_timestamp": timestamp,
                "projects": project_ids if isinstance(project_ids, list) else [project_ids],
                "total_records": len(records),
                "total_scan_types_discovered": len([e for e in discovery_log if e.get("event") == "new_scan_type"]),
                "total_errors": len([e for e in discovery_log if e.get("event") == "error"]),
                "final_column_count": len(known_tags),
                "all_discovered_columns": list(known_tags.keys()),
                "events": discovery_log,
            }

            with open(log_path, "w") as f:
                json.dump(discovery_summary, f, indent=2, default=str)

            print(f"Saved discovery log to: {log_path}")

            if verbose:
                print(f"\n{'='*60}")
                print(f"EXTRACTION SUMMARY")
                print(f"{'='*60}")
                print(f"  Total records:        {len(records)}")
                print(f"  Scan types found:     {discovery_summary['total_scan_types_discovered']}")
                print(f"  Errors encountered:   {discovery_summary['total_errors']}")
                print(f"  Total columns:        {len(known_tags)}")
                print(f"{'='*60}\n")

        return df

    finally:
        if close_connection and connection is not None:
            connection.disconnect()
