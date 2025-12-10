# xnat_dicom_utils.py

from typing import Optional

def count_dicoms_in_project(
    project,
    dicom_resource_label: str = "DICOM",
    max_experiments: Optional[int] = None,
    max_scans_per_experiment: Optional[int] = None,
) -> int:
    """
    Count the number of DICOM files in an XNAT project.

    Parameters
    ----------
    project : xnat.Project
        XNAT project object from an active xnat.Connection.
    dicom_resource_label : str
        Resource label used for DICOM (typically "DICOM").
    max_experiments : int, optional
        If provided, limit counting to the first `max_experiments` experiments.
    max_scans_per_experiment : int, optional
        If provided, limit counting to the first `max_scans_per_experiment` scans per experiment.

    Returns
    -------
    int
        Total number of DICOM files discovered in the project (or in the sampled subset,
        if `max_experiments` / `max_scans_per_experiment` are set).
    """
    total_dicom_files = 0
    dicom_label_upper = dicom_resource_label.upper()

    experiments = project.experiments.values()
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    for exp in experiments:
        scans = list(exp.scans.values())
        if max_scans_per_experiment is not None:
            scans = scans[:max_scans_per_experiment]

        for scan in scans:
            for res_label, res in scan.resources.items():
                # res.files is a container of file objects
                total_dicom_files += len(list(res.files))

    return total_dicom_files