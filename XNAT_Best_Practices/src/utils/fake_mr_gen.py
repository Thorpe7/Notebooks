"""
Synthetic MR DICOM generator for testing purposes.
Place in utils/ and import: from utils.fake_mr_dicom import create_fake_mr_dicom

Requires: pip install pydicom numpy
"""

import numpy as np
from pydicom import Dataset, FileDataset, uid
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# Image Data Generation
# -----------------------------------------------------------------------------

def generate_synthetic_brain_slice(rows: int = 256, cols: int = 256, seed: Optional[int] = None) -> np.ndarray:
    """Generate a synthetic brain-like MR slice with elliptical structures."""
    if seed is not None:
        np.random.seed(seed)
    
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows // 2, cols // 2
    
    # Skull (outer ellipse)
    skull_mask = ((x - cx) / (cols * 0.4)) ** 2 + ((y - cy) / (rows * 0.45)) ** 2 <= 1
    
    # Brain tissue (inner ellipse)
    brain_mask = ((x - cx) / (cols * 0.35)) ** 2 + ((y - cy) / (rows * 0.4)) ** 2 <= 1
    
    # Ventricles (small central ellipses)
    vent_l = ((x - cx + 30) / 15) ** 2 + ((y - cy) / 40) ** 2 <= 1
    vent_r = ((x - cx - 30) / 15) ** 2 + ((y - cy) / 40) ** 2 <= 1
    
    # Build image with intensity values
    img = np.zeros((rows, cols), dtype=np.float32)
    img[skull_mask] = 800    # Skull/CSF boundary
    img[brain_mask] = 1200   # White/gray matter
    img[vent_l | vent_r] = 400  # CSF in ventricles
    
    # Add realistic noise and texture
    img += np.random.normal(0, 50, img.shape)
    img = np.clip(img, 0, 4095).astype(np.uint16)
    
    return img


def generate_noise_image(rows: int = 256, cols: int = 256, seed: Optional[int] = None) -> np.ndarray:
    """Generate simple noise image (fallback/debug option)."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(0, 4096, (rows, cols), dtype=np.uint16)


# -----------------------------------------------------------------------------
# DICOM Module Builders
# -----------------------------------------------------------------------------

def build_file_meta() -> Dataset:
    """Create DICOM file meta information header."""
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = uid.generate_uid()
    file_meta.ImplementationVersionName = "SYNTHETIC_MR_1.0"
    return file_meta


def add_patient_module(ds: Dataset, patient_name: str, patient_id: str, 
                       birth_date: str = "19800101", sex: str = "O") -> None:
    """Add Patient Module attributes (C.7.1.1)."""
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.PatientBirthDate = birth_date
    ds.PatientSex = sex


def add_study_module(ds: Dataset, study_description: str = "Synthetic MR Study",
                     study_date: Optional[str] = None, study_time: Optional[str] = None,
                     accession_number: str = "ACC001") -> None:
    """Add General Study Module attributes (C.7.2.1)."""
    now = datetime.now()
    ds.StudyDate = study_date or now.strftime("%Y%m%d")
    ds.StudyTime = study_time or now.strftime("%H%M%S.%f")
    ds.StudyDescription = study_description
    ds.StudyInstanceUID = uid.generate_uid()
    ds.StudyID = "1"
    ds.AccessionNumber = accession_number
    ds.ReferringPhysicianName = "Synthetic^Physician"


def add_series_module(ds: Dataset, series_description: str = "Synthetic T1W",
                      series_number: int = 1, modality: str = "MR") -> None:
    """Add General Series Module attributes (C.7.3.1)."""
    now = datetime.now()
    ds.Modality = modality
    ds.SeriesInstanceUID = uid.generate_uid()
    ds.SeriesNumber = series_number
    ds.SeriesDescription = series_description
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.SeriesTime = now.strftime("%H%M%S.%f")
    ds.BodyPartExamined = "BRAIN"
    ds.PatientPosition = "HFS"


def add_frame_of_reference(ds: Dataset) -> None:
    """Add Frame of Reference Module attributes (C.7.4.1)."""
    ds.FrameOfReferenceUID = uid.generate_uid()
    ds.PositionReferenceIndicator = ""


def add_equipment_module(ds: Dataset, institution: str = "Synthetic Hospital",
                         manufacturer: str = "SyntheticVendor",
                         station_name: str = "SYN_MR_01") -> None:
    """Add General Equipment Module attributes (C.7.5.1)."""
    ds.Manufacturer = manufacturer
    ds.InstitutionName = institution
    ds.StationName = station_name
    ds.ManufacturerModelName = "Synthetic Scanner 3T"
    ds.DeviceSerialNumber = "SN123456"
    ds.SoftwareVersions = "1.0.0"


def add_image_module(ds: Dataset, rows: int, cols: int, instance_number: int = 1,
                     slice_location: float = 0.0, slice_thickness: float = 5.0,
                     pixel_spacing: Tuple[float, float] = (1.0, 1.0)) -> None:
    """Add General/MR Image Module attributes (C.7.6 / C.8.3)."""
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = uid.generate_uid()
    ds.InstanceNumber = instance_number
    ds.ImageType = ["ORIGINAL", "PRIMARY", "M", "ND"]
    ds.ContentDate = datetime.now().strftime("%Y%m%d")
    ds.ContentTime = datetime.now().strftime("%H%M%S.%f")
    ds.AcquisitionNumber = 1
    
    # Image Plane Module
    ds.SliceThickness = slice_thickness
    ds.SliceLocation = slice_location
    ds.ImagePositionPatient = [0.0, 0.0, slice_location]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = list(pixel_spacing)
    
    # Image Pixel Module
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.WindowCenter = 600
    ds.WindowWidth = 1600


def add_mr_parameters(ds: Dataset, tr: float = 2000.0, te: float = 30.0,
                      flip_angle: float = 90.0, field_strength: float = 3.0,
                      sequence_name: str = "SE") -> None:
    """Add MR-specific acquisition parameters (C.8.3.1)."""
    ds.MagneticFieldStrength = field_strength
    ds.RepetitionTime = tr
    ds.EchoTime = te
    ds.FlipAngle = flip_angle
    ds.EchoTrainLength = 1
    ds.SequenceName = sequence_name
    ds.ScanningSequence = "SE"
    ds.SequenceVariant = "SK"
    ds.ScanOptions = ""
    ds.MRAcquisitionType = "2D"
    ds.NumberOfAverages = 1
    ds.ImagingFrequency = 127.74  # ~3T proton frequency
    ds.ImagedNucleus = "1H"
    ds.PercentSampling = 100.0
    ds.PercentPhaseFieldOfView = 100.0


def add_pixel_data(ds: Dataset, pixel_array: np.ndarray) -> None:
    """Add pixel data to the dataset."""
    ds.PixelData = pixel_array.tobytes()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def create_fake_mr_dicom(
    output_path: str = "fake_mr.dcm",
    rows: int = 256,
    cols: int = 256,
    patient_name: str = "Synthetic^Patient",
    patient_id: str = "SYN001",
    series_description: str = "Synthetic T1W Brain",
    instance_number: int = 1,
    slice_location: float = 0.0,
    tr: float = 2000.0,
    te: float = 30.0,
    use_brain_phantom: bool = True,
    seed: Optional[int] = None,
    save: bool = True,
) -> FileDataset:
    """
    Create a synthetic MR DICOM file with realistic metadata.
    
    Parameters
    ----------
    output_path : str
        Output file path for the DICOM file.
    rows, cols : int
        Image dimensions.
    patient_name : str
        Patient name in DICOM format (Family^Given).
    patient_id : str
        Patient ID string.
    series_description : str
        Description for the series.
    instance_number : int
        Instance number for multi-slice series.
    slice_location : float
        Slice position in mm.
    tr, te : float
        Repetition time and echo time in ms.
    use_brain_phantom : bool
        If True, generate brain-like phantom; otherwise noise.
    seed : int, optional
        Random seed for reproducibility.
    save : bool
        If True, write file to disk.
    
    Returns
    -------
    FileDataset
        The created DICOM dataset (also saved to disk if save=True).
    
    Example
    -------
    >>> from utils.fake_mr_dicom import create_fake_mr_dicom
    >>> ds = create_fake_mr_dicom("test.dcm", rows=512, cols=512)
    >>> print(ds.PatientName)
    """
    # Build file meta and dataset
    file_meta = build_file_meta()
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add all DICOM modules
    add_patient_module(ds, patient_name, patient_id)
    add_study_module(ds)
    add_series_module(ds, series_description)
    add_frame_of_reference(ds)
    add_equipment_module(ds)
    add_image_module(ds, rows, cols, instance_number, slice_location)
    add_mr_parameters(ds, tr=tr, te=te)
    
    # Generate and add pixel data
    if use_brain_phantom:
        pixel_array = generate_synthetic_brain_slice(rows, cols, seed)
    else:
        pixel_array = generate_noise_image(rows, cols, seed)
    add_pixel_data(ds, pixel_array)
    
    # Ensure required is_little_endian and is_implicit_VR are set
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    
    if save:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(output_path)
        print(f"Saved synthetic MR DICOM: {output_path}")
    
    return ds


# -----------------------------------------------------------------------------
# CLI Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test when run directly
    ds = create_fake_mr_dicom("test_output/fake_mr.dcm", seed=42)
    print(f"Created: {ds.PatientName}, {ds.Rows}x{ds.Columns}, Series: {ds.SeriesDescription}")