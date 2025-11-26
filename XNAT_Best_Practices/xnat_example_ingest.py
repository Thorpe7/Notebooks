import uuid
import os, re, zipfile
import xnat
import tempfile
import shutil
import pydicom
import time

from pathlib import Path
from xnat.mixin import ProjectData
from xnat.session import XNATSession
from typing import Dict, List, Tuple
from pydicom.uid import generate_uid

def get_project(url: str, username: str|None, password: str|None, prj_name: str) -> Tuple[XNATSession,ProjectData]:
    """Login to the XNAT server and return the session object."""
    if username is None or password is None:
        raise ValueError("Username/password not found or provided.")
    s = xnat.connect(url, user=username, password=password)
    proj = s.projects[prj_name] if prj_name in s.projects else None
    if proj:
        return (s, proj)
    else:
        raise ValueError(f"Project {prj_name} not found on server.")

def extract_filename_elements(filepath: str) -> Dict[str, str]:
    """Extract components from the filename using regex."""
    re_filename = re.compile(
        r'^(?P<imgid>[^_]+)_(?P<patient>[^_]+)_(?P<modality>[^_]+)_(?P<lat>L|R)_(?P<view>[^_]+)_ANON\.dcm$',
        re.IGNORECASE
    )
    m = re_filename.match(filepath)
    filename_elements = m.groupdict() if m else None
    if filename_elements:
        return filename_elements
    else:
        raise ValueError(f"Filename does not match expected pattern: {filepath}")

def sort_pt_to_buckets(parent_dir: str):
    """Sort files into buckets based on filename elements."""
    buckets = {}
    parent_path = Path(parent_dir)
    print(f"Parent path: {parent_path}")
    print(f"Is directory: {parent_path.is_dir()}")
    if parent_path.is_dir():
       for birad_dir in sorted(parent_path.glob("BIRADS_*")):
           birads_level = birad_dir.name
           for dcm_file in birad_dir.glob("*.dcm"):
               file_elements = extract_filename_elements(str(dcm_file.name))
               patient = file_elements.get("patient")
               fname = file_elements.get("imgid")
               key = (patient, fname, birads_level)
               buckets.setdefault(key, []).append(str(dcm_file))

    return buckets

def add_files_to_xnat(s, XNATSession, proj, buckets):
    """Ingest files into XNAT project."""
    tmp_dir = tempfile.mkdtemp(prefix="xnat_ingest_")
    try:
        for (patient, fname, birads), files in sorted(buckets.items()):
            subject_label = patient
            experiment_label = f"{fname}_{birads}_{str(uuid.uuid4().int)[:10]}"

            if subject_label in proj.subjects:
                subj = proj.subjects[subject_label]
            else:
                subj = s.classes.SubjectData(parent=proj, label=subject_label)

            src_path = files[0]

            # Generating new SOP & Series instance UID to avoid xnat session conflicts
            ds = pydicom.dcmread(src_path)
            ds.SOPInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.save_as(src_path)

            zip_name = f"{experiment_label}.zip"
            zip_path = os.path.join(tmp_dir, zip_name)

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(src_path, arcname=os.path.basename(src_path))

            s.services.import_(
                zip_path,
                project=proj.id,
                subject=subject_label,
                experiment=experiment_label,
                destination="/prearchive",
                overwrite="delete"
            )
            print(f"Ingested 1 (zipped) → {subject_label}/{experiment_label} ({birads}) from {src_path}")
            os.remove(zip_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    start_time = time.time()

    username = os.getenv("TAP_ALIAS", None)
    password = os.getenv("TAP_SECRET", None)
    s, project = get_project("https://tap.embarklabs.ai/", username, password, "InBreastProject")
    parent_data_dir = "/home/maxwell/code/local/INbreast_to_XNAT/data/OrganizedByBiRads"
    pt_buckets = sort_pt_to_buckets(parent_data_dir)
    add_files_to_xnat(s, XNATSession, project, pt_buckets)

    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
