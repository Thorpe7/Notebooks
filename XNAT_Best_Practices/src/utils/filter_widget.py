"""
widget.py

Interactive Jupyter widget for filtering and visualizing XNAT metadata DataFrames.
"""

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


def create_multiselect(df: pd.DataFrame, column: str, description: str) -> widgets.SelectMultiple:
    """Create a multi-select widget for a categorical column."""
    unique_vals = sorted(df[column].dropna().unique().tolist())
    return widgets.SelectMultiple(
        options=unique_vals,
        value=unique_vals,
        description=description,
        layout=widgets.Layout(width='250px', height='100px'),
        style={'description_width': '80px'},
    )


def create_text_search(description: str) -> widgets.Text:
    """Create a text search widget."""
    return widgets.Text(
        value='',
        placeholder='Search...',
        description=description,
        layout=widgets.Layout(width='250px'),
        style={'description_width': '80px'},
    )


def create_reset_button() -> widgets.Button:
    """Create a reset filters button."""
    return widgets.Button(
        description='Reset Filters',
        button_style='warning',
        icon='refresh',
        layout=widgets.Layout(width='150px'),
    )


def create_export_button() -> widgets.Button:
    """Create an export to CSV button."""
    return widgets.Button(
        description='Export CSV',
        button_style='success',
        icon='download',
        layout=widgets.Layout(width='150px'),
    )


def apply_filters(
    df: pd.DataFrame,
    subjects: tuple,
    modalities: tuple,
    scan_types: tuple,
    session_types: tuple,
    description_search: str,
) -> pd.DataFrame:
    """Apply all filter criteria to the DataFrame."""
    filtered = df.copy()

    if subjects:
        filtered = filtered[filtered['subject'].isin(subjects)]
    if modalities and 'modality' in filtered.columns:
        filtered = filtered[filtered['modality'].isin(modalities)]
    if scan_types and 'scan_type' in filtered.columns:
        filtered = filtered[filtered['scan_type'].isin(scan_types)]
    if session_types and 'session_type' in filtered.columns:
        filtered = filtered[filtered['session_type'].isin(session_types)]
    if description_search and 'series_description' in filtered.columns:
        mask = filtered['series_description'].fillna('').str.contains(
            description_search, case=False, regex=False
        )
        filtered = filtered[mask]

    return filtered


def build_summary_stats(df: pd.DataFrame, filtered: pd.DataFrame) -> widgets.HTML:
    """Generate summary statistics HTML."""
    total_scans = len(df)
    filtered_scans = len(filtered)
    n_subjects = filtered['subject'].nunique()
    n_sessions = filtered['session'].nunique()

    modality_counts = ''
    if 'modality' in filtered.columns and not filtered.empty:
        counts = filtered['modality'].value_counts()
        modality_counts = ' | '.join([f"{k}: {v}" for k, v in counts.items()])

    html = f"""
    <div style="background: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <b>Showing:</b> {filtered_scans} / {total_scans} scans |
        <b>Subjects:</b> {n_subjects} |
        <b>Sessions:</b> {n_sessions}<br>
        <b>Modalities:</b> {modality_counts if modality_counts else 'N/A'}
    </div>
    """
    return widgets.HTML(value=html)


def create_metadata_widget(df: pd.DataFrame, export_path: str = 'filtered_metadata.csv'):
    """
    Main function: Create and display an interactive filtering widget.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from build_project_dataframe() containing XNAT metadata.
    export_path : str
        Default filename for CSV export.

    Returns
    -------
    widgets.VBox
        The widget container (also displayed automatically).
    """
    # Store filtered df for export
    state = {'filtered_df': df.copy()}

    # Create filter widgets
    subject_select = create_multiselect(df, 'subject', 'Subjects:')

    modality_select = None
    if 'modality' in df.columns and df['modality'].notna().any():
        modality_select = create_multiselect(df, 'modality', 'Modality:')

    scan_type_select = None
    if 'scan_type' in df.columns and df['scan_type'].notna().any():
        scan_type_select = create_multiselect(df, 'scan_type', 'Scan Type:')

    session_type_select = None
    if 'session_type' in df.columns and df['session_type'].notna().any():
        session_type_select = create_multiselect(df, 'session_type', 'Session:')

    description_search = create_text_search('Search:')

    reset_btn = create_reset_button()
    export_btn = create_export_button()

    # Output areas
    stats_output = widgets.Output()
    table_output = widgets.Output()

    def update_display(_=None):
        """Refresh the filtered view."""
        filtered = apply_filters(
            df,
            subjects=subject_select.value,
            modalities=modality_select.value if modality_select else (),
            scan_types=scan_type_select.value if scan_type_select else (),
            session_types=session_type_select.value if session_type_select else (),
            description_search=description_search.value,
        )
        state['filtered_df'] = filtered

        with stats_output:
            clear_output(wait=True)
            display(build_summary_stats(df, filtered))

        with table_output:
            clear_output(wait=True)
            if filtered.empty:
                print("No scans match the current filters.")
            else:
                display(filtered)

    def reset_filters(_):
        """Reset all filters to default (all selected)."""
        subject_select.value = list(subject_select.options)
        if modality_select:
            modality_select.value = list(modality_select.options)
        if scan_type_select:
            scan_type_select.value = list(scan_type_select.options)
        if session_type_select:
            session_type_select.value = list(session_type_select.options)
        description_search.value = ''
        update_display()

    def export_csv(_):
        """Export filtered DataFrame to CSV."""
        state['filtered_df'].to_csv(export_path, index=False)
        with table_output:
            print(f"\n✓ Exported {len(state['filtered_df'])} rows to {export_path}")

    # Wire up observers
    subject_select.observe(update_display, names='value')
    description_search.observe(update_display, names='value')
    if modality_select:
        modality_select.observe(update_display, names='value')
    if scan_type_select:
        scan_type_select.observe(update_display, names='value')
    if session_type_select:
        session_type_select.observe(update_display, names='value')

    reset_btn.on_click(reset_filters)
    export_btn.on_click(export_csv)

    # Build layout
    filter_widgets = [subject_select]
    if modality_select:
        filter_widgets.append(modality_select)
    if scan_type_select:
        filter_widgets.append(scan_type_select)
    if session_type_select:
        filter_widgets.append(session_type_select)

    filter_row = widgets.HBox(filter_widgets)
    search_row = widgets.HBox([description_search, reset_btn, export_btn])

    container = widgets.VBox([
        widgets.HTML('<h3>XNAT Metadata Explorer</h3>'),
        filter_row,
        search_row,
        stats_output,
        table_output,
    ])

    # Initial render
    update_display()
    display(container)

    return container