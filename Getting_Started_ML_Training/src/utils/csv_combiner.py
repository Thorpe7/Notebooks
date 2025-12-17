# src/utils/csv_combiner.py
"""
Utility script for combining multiple CSV files with varying columns.

Handles CSVs that share common columns (project_id, project_name, subject_id,
subject_label) but may have different additional columns.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def _load_csv_with_source(
    file_path: Union[str, Path],
    add_source_column: bool = False,
) -> pd.DataFrame:
    """
    Load a single CSV file and optionally add a source column.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file
    add_source_column : bool
        If True, adds a 'source_file' column with the filename

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    if add_source_column:
        df["source_file"] = file_path.name

    return df


def _validate_required_columns(
    df: pd.DataFrame,
    file_path: str,
    required_columns: List[str],
) -> None:
    """
    Validate that a DataFrame contains the required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    file_path : str
        Path to the source file (for error messages)
    required_columns : list of str
        List of column names that must be present

    Raises
    ------
    ValueError
        If any required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"CSV '{file_path}' is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def combine_csvs(
    file_paths: List[Union[str, Path]],
    required_columns: Optional[List[str]] = None,
    add_source_column: bool = False,
    sort_by: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Combine multiple CSV files into a single DataFrame.

    All columns from all CSVs are preserved. If a column exists in one CSV
    but not another, the missing values will be NaN for rows from files
    that don't have that column.

    Parameters
    ----------
    file_paths : list of str or Path
        List of paths to CSV files to combine
    required_columns : list of str, optional
        Columns that must exist in every CSV. Defaults to
        ["project_id", "project_name", "subject_id", "subject_label"]
    add_source_column : bool, default False
        If True, adds a 'source_file' column indicating which file each row came from
    sort_by : list of str, optional
        Columns to sort the final DataFrame by. If None, no sorting is applied.
    verbose : bool, default False
        If True, prints progress information

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all rows and all columns from all input CSVs

    Examples
    --------
    >>> combined_df = combine_csvs([
    ...     "data/project1_metadata.csv",
    ...     "data/project2_metadata.csv",
    ...     "data/project3_metadata.csv",
    ... ])

    >>> # With source tracking and sorting
    >>> combined_df = combine_csvs(
    ...     file_paths=["file1.csv", "file2.csv"],
    ...     add_source_column=True,
    ...     sort_by=["project_id", "subject_id"],
    ...     verbose=True,
    ... )
    """
    if required_columns is None:
        required_columns = ["project_id", "project_name", "subject_id", "subject_label"]

    if not file_paths:
        raise ValueError("No file paths provided")

    dataframes = []
    all_columns = set()

    # Load all CSVs and collect column names
    for file_path in file_paths:
        if verbose:
            print(f"Loading: {file_path}")

        df = _load_csv_with_source(file_path, add_source_column=add_source_column)
        _validate_required_columns(df, str(file_path), required_columns)

        all_columns.update(df.columns)
        dataframes.append(df)

        if verbose:
            print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    if verbose:
        print(f"\nTotal unique columns across all files: {len(all_columns)}")

    # Concatenate all DataFrames (pandas handles missing columns automatically)
    combined = pd.concat(dataframes, ignore_index=True, sort=False)

    if verbose:
        print(f"Combined DataFrame: {len(combined)} rows, {len(combined.columns)} columns")

    # Reorder columns: required columns first, then alphabetically
    final_columns = []
    for col in required_columns:
        if col in combined.columns:
            final_columns.append(col)

    # Add source_file column early if it exists
    if add_source_column and "source_file" in combined.columns:
        final_columns.append("source_file")

    # Add remaining columns alphabetically
    remaining = sorted([c for c in combined.columns if c not in final_columns])
    final_columns.extend(remaining)

    combined = combined[final_columns]

    # Sort if requested
    if sort_by:
        sort_cols = [c for c in sort_by if c in combined.columns]
        if sort_cols:
            combined = combined.sort_values(sort_cols).reset_index(drop=True)
            if verbose:
                print(f"Sorted by: {sort_cols}")

    return combined


def combine_csvs_from_directory(
    directory: Union[str, Path],
    pattern: str = "*.csv",
    required_columns: Optional[List[str]] = None,
    add_source_column: bool = False,
    sort_by: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Combine all CSV files matching a pattern in a directory.

    Parameters
    ----------
    directory : str or Path
        Directory to search for CSV files
    pattern : str, default "*.csv"
        Glob pattern for matching CSV files
    required_columns : list of str, optional
        Columns that must exist in every CSV
    add_source_column : bool, default False
        If True, adds a 'source_file' column
    sort_by : list of str, optional
        Columns to sort the final DataFrame by
    verbose : bool, default False
        If True, prints progress information

    Returns
    -------
    pd.DataFrame
        Combined DataFrame
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    file_paths = sorted(directory.glob(pattern))

    if not file_paths:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")

    if verbose:
        print(f"Found {len(file_paths)} files matching '{pattern}' in {directory}")

    return combine_csvs(
        file_paths=file_paths,
        required_columns=required_columns,
        add_source_column=add_source_column,
        sort_by=sort_by,
        verbose=verbose,
    )


def save_combined_csv(
    combined_df: pd.DataFrame,
    output_path: Union[str, Path],
    verbose: bool = False,
) -> Path:
    """
    Save a combined DataFrame to CSV.

    Parameters
    ----------
    combined_df : pd.DataFrame
        DataFrame to save
    output_path : str or Path
        Path for the output CSV file
    verbose : bool, default False
        If True, prints save confirmation

    Returns
    -------
    Path
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Saved combined CSV to: {output_path}")
        print(f"  Rows: {len(combined_df)}, Columns: {len(combined_df.columns)}")

    return output_path


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csv_combiner.py <directory> [--output combined.csv] [--pattern *.csv]")
        print("\nExample:")
        print("  python csv_combiner.py /path/to/csv_folder --output merged.csv")
        print("  python csv_combiner.py /path/to/csv_folder --pattern 'metadata_*.csv'")
        sys.exit(1)

    # Parse arguments
    directory = sys.argv[1]
    output_path = "combined_output.csv"
    pattern = "*.csv"

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--pattern" and i + 1 < len(sys.argv):
            pattern = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Combine CSVs from directory
    try:
        combined = combine_csvs_from_directory(
            directory=directory,
            pattern=pattern,
            add_source_column=True,
            sort_by=["project_id", "subject_id"],
            verbose=True,
        )

        save_combined_csv(combined, output_path, verbose=True)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
