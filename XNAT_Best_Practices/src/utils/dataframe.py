"""
dataframe.py

Module for extracting imaging metadata from XNAT projects into pandas DataFrames.
"""

import os
from typing import Optional
import pandas as pd
import xnat


def get_connection_params(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> dict:
    """Resolve connection parameters from arguments or environment variables."""
    return {
        'server': host or os.environ.get('XNAT_HOST'),
        'user': user or os.environ.get('XNAT_USER'),
        'password': password or os.environ.get('XNAT_PASSWORD'),
    }


def extract_scan_metadata(scan, subject_label: str, session_label: str) -> dict:
    """Extract metadata from a single scan object."""
    return {
        'subject': subject_label,
        'session': session_label,
        'scan_id': scan.id,
        'scan_type': getattr(scan, 'type', None),
        'series_description': getattr(scan, 'series_description', None),
        'modality': getattr(scan, 'modality', None),
        'quality': getattr(scan, 'quality', None),
        'frames': getattr(scan, 'frames', None),
        'note': getattr(scan, 'note', None),
    }


def extract_session_metadata(experiment) -> dict:
    """Extract session-level metadata from an experiment object."""
    return {
        'session_label': experiment.label,
        'session_type': experiment.xsi_type,
        'date': getattr(experiment, 'date', None),
        'age': getattr(experiment, 'age', None),
    }


def collect_project_metadata(session, project_id: str) -> list[dict]:
    """Iterate through a project and collect metadata for all scans."""
    project = session.projects[project_id]
    records = []

    for subject in project.subjects.values():
        subject_label = subject.label

        for experiment in subject.experiments.values():
            session_meta = extract_session_metadata(experiment)

            for scan in experiment.scans.values():
                record = extract_scan_metadata(scan, subject_label, session_meta['session_label'])
                record['session_type'] = session_meta['session_type']
                record['session_date'] = session_meta['date']
                record['subject_age'] = session_meta['age']
                records.append(record)

    return records


def build_project_dataframe(
    project_id: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main function: Connect to XNAT and build a DataFrame of scan metadata.

    Parameters
    ----------
    project_id : str
        The XNAT project ID to query.
    host : str, optional
        XNAT server URL. Falls back to XNAT_HOST env var.
    user : str, optional
        XNAT username. Falls back to XNAT_USER env var.
    password : str, optional
        XNAT password. Falls back to XNAT_PASSWORD env var.

    Returns
    -------
    pd.DataFrame
        DataFrame containing scan-level metadata with columns:
        subject, session, scan_id, scan_type, series_description,
        modality, quality, frames, note, session_type, session_date, subject_age
    """
    conn_params = get_connection_params(host, user, password)

    with xnat.connect(**conn_params) as session:
        records = collect_project_metadata(session, project_id)

    df = pd.DataFrame(records)

    # Reorder columns for readability
    column_order = [
        'subject', 'session', 'session_type', 'session_date', 'subject_age',
        'scan_id', 'scan_type', 'series_description', 'modality', 'quality', 'frames', 'note'
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract XNAT project metadata to DataFrame')
    parser.add_argument('project_id', help='XNAT project ID')
    parser.add_argument('--host', help='XNAT server URL')
    parser.add_argument('--user', help='XNAT username')
    parser.add_argument('--password', help='XNAT password')
    parser.add_argument('--output', '-o', help='Output CSV path (optional)')

    args = parser.parse_args()

    df = build_project_dataframe(
        args.project_id,
        host=args.host,
        user=args.user,
        password=args.password,
    )

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
    else:
        print(df.to_string())