# src/utils/filter_widget.py

"""
Interactive Jupyter dashboard for exploring and filtering XNAT DICOM metadata DataFrames.

This module provides Voila-friendly dashboards with:
- Toggle buttons for categorical attribute filtering
- Range sliders/inputs for numeric metadata values
- Distribution visualizations (histograms, bar charts, counts)
- Export functionality for filtered data

Usage:
    from src.utils.filter_widget import create_metadata_dashboard

    dashboard = create_metadata_dashboard(df)
    dashboard  # Display in Jupyter/Voila
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


# -------------------------------------------------------------------------
# Configuration: Which columns to use for different widget types
# -------------------------------------------------------------------------

# Categorical columns that get toggle buttons
CATEGORICAL_COLUMNS = [
    "Modality",
    "BodyPartExamined",
    "PatientSex",
    "InstitutionName",
    "Manufacturer",
    "xnat_project_id",
]

# Numeric columns that get range inputs
NUMERIC_COLUMNS = [
    "SliceThickness",
    "Rows",
    "Columns",
]

# Date columns for date range filtering
DATE_COLUMNS = [
    "StudyDate",
]


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------

def _get_available_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Return only columns that exist in the DataFrame and have non-null values."""
    return [col for col in candidates if col in df.columns and df[col].notna().any()]


def _get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Get sorted unique non-null values for a column."""
    values = df[column].dropna().unique()
    try:
        return sorted([str(v) for v in values])
    except TypeError:
        return [str(v) for v in values]


# -------------------------------------------------------------------------
# Distribution visualization dashboard
# -------------------------------------------------------------------------

def distribution_dashboard(
    df: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
) -> VBox:
    """
    Dashboard to explore distributions of metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DICOM metadata.
    categorical_columns : List[str], optional
        Columns to show as bar charts. Defaults to CATEGORICAL_COLUMNS.
    numeric_columns : List[str], optional
        Columns to show as histograms. Defaults to NUMERIC_COLUMNS.

    Returns
    -------
    ipywidgets.VBox
    """
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_COLUMNS
    if numeric_columns is None:
        numeric_columns = NUMERIC_COLUMNS

    available_cat = _get_available_columns(df, categorical_columns)
    available_num = _get_available_columns(df, numeric_columns)
    all_columns = available_cat + available_num

    if not all_columns:
        return VBox([widgets.HTML("<p>No plottable columns found in DataFrame.</p>")])

    # Column selector
    column_selector = widgets.Dropdown(
        options=all_columns,
        value=all_columns[0],
        description="Column:",
        style={"description_width": "60px"},
    )

    # Plot type selector
    plot_type = widgets.ToggleButtons(
        options=[("Bar Chart", "bar"), ("Histogram", "hist"), ("Pie Chart", "pie")],
        value="bar",
        description="Plot:",
    )

    # Top N selector for categorical
    top_n = widgets.IntSlider(
        value=10,
        min=5,
        max=50,
        step=5,
        description="Top N:",
        continuous_update=False,
    )

    out = widgets.Output()

    def _update(change=None):
        col = column_selector.value
        ptype = plot_type.value

        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 5))

            if col in available_cat:
                # Categorical: show value counts
                counts = df[col].value_counts().head(top_n.value)

                if ptype == "bar":
                    counts.plot(kind="barh", ax=ax, color="steelblue")
                    ax.set_xlabel("Count")
                    ax.set_ylabel(col)
                    ax.set_title(f"Distribution of {col} (Top {len(counts)})")
                    ax.invert_yaxis()
                elif ptype == "pie":
                    counts.plot(kind="pie", ax=ax, autopct="%1.1f%%")
                    ax.set_ylabel("")
                    ax.set_title(f"Distribution of {col}")
                else:
                    counts.plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_ylabel("Count")
                    ax.set_xlabel(col)
                    ax.set_title(f"Distribution of {col}")
                    plt.xticks(rotation=45, ha="right")

            elif col in available_num:
                # Numeric: show histogram
                data = pd.to_numeric(df[col], errors="coerce").dropna()

                if ptype in ("bar", "hist"):
                    ax.hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"Distribution of {col}")

                    # Add statistics
                    stats_text = f"Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}"
                    ax.text(
                        0.95, 0.95, stats_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )
                else:
                    # For pie, bin the data
                    binned = pd.cut(data, bins=5)
                    binned.value_counts().sort_index().plot(kind="pie", ax=ax, autopct="%1.1f%%")
                    ax.set_ylabel("")
                    ax.set_title(f"Distribution of {col} (binned)")

            plt.tight_layout()
            plt.show()

    column_selector.observe(_update, names="value")
    plot_type.observe(_update, names="value")
    top_n.observe(_update, names="value")

    _update()

    controls = HBox([column_selector, plot_type, top_n])
    return VBox([controls, out])


# -------------------------------------------------------------------------
# Filter dashboard with toggle buttons and range inputs
# -------------------------------------------------------------------------

def create_metadata_dashboard(
    df: pd.DataFrame,
    export_path: str = "filtered_metadata.csv",
) -> VBox:
    """
    Create an interactive dashboard for exploring and filtering DICOM metadata.

    Features:
    - Toggle buttons to filter by categorical attributes (Modality, BodyPart, etc.)
    - Text input for searching descriptions
    - Distribution visualizations that update with filters
    - Summary statistics panel
    - Export filtered data to CSV

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from collect_dicom_metadata() containing DICOM metadata.
    export_path : str
        Default filename for CSV export.

    Returns
    -------
    ipywidgets.VBox
        The dashboard widget container.
    """
    if df.empty:
        return VBox([widgets.HTML("<p>No data available. DataFrame is empty.</p>")])

    # State to track filtered DataFrame
    state = {"filtered_df": df.copy()}

    # Identify available columns
    available_cat = _get_available_columns(df, CATEGORICAL_COLUMNS)
    available_date = _get_available_columns(df, DATE_COLUMNS)

    # -------------------------------------------------------------------------
    # Create filter widgets
    # -------------------------------------------------------------------------

    filter_widgets: Dict[str, Any] = {}

    # Toggle buttons for categorical columns
    for col in available_cat:
        unique_vals = _get_unique_values(df, col)
        if len(unique_vals) > 20:
            # Too many values - use a SelectMultiple instead
            filter_widgets[col] = widgets.SelectMultiple(
                options=unique_vals,
                value=tuple(unique_vals),
                description=f"{col}:",
                layout=Layout(width="200px", height="100px"),
                style={"description_width": "80px"},
            )
        else:
            # Use checkboxes wrapped in a container
            checkboxes = []
            for val in unique_vals:
                cb = widgets.Checkbox(
                    value=True,
                    description=str(val),
                    indent=False,
                    layout=Layout(width="auto"),
                )
                checkboxes.append(cb)
            filter_widgets[col] = {
                "type": "checkboxes",
                "widgets": checkboxes,
                "values": unique_vals,
            }

    # Text search for SeriesDescription
    description_search = widgets.Text(
        value="",
        placeholder="Search series description...",
        description="Search:",
        layout=Layout(width="300px"),
        style={"description_width": "60px"},
    )

    # -------------------------------------------------------------------------
    # Action buttons
    # -------------------------------------------------------------------------

    reset_btn = widgets.Button(
        description="Reset Filters",
        button_style="warning",
        icon="refresh",
        layout=Layout(width="140px"),
    )

    export_btn = widgets.Button(
        description="Export CSV",
        button_style="success",
        icon="download",
        layout=Layout(width="140px"),
    )

    # -------------------------------------------------------------------------
    # Output areas
    # -------------------------------------------------------------------------

    stats_output = widgets.Output()
    viz_output = widgets.Output()
    table_output = widgets.Output()

    # Visualization controls
    viz_column = widgets.Dropdown(
        options=available_cat if available_cat else ["(none)"],
        value=available_cat[0] if available_cat else "(none)",
        description="Show:",
        style={"description_width": "50px"},
        layout=Layout(width="200px"),
    )

    viz_type = widgets.ToggleButtons(
        options=[("Count", "count"), ("Proportion", "prop")],
        value="count",
        description="Y-axis:",
    )

    # -------------------------------------------------------------------------
    # Update functions
    # -------------------------------------------------------------------------

    def apply_filters() -> pd.DataFrame:
        """Apply all filters and return filtered DataFrame."""
        filtered = df.copy()

        # Apply categorical filters
        for col, widget_info in filter_widgets.items():
            if col not in filtered.columns:
                continue

            if isinstance(widget_info, dict) and widget_info["type"] == "checkboxes":
                # Checkbox-based filter
                selected = [
                    val for cb, val in zip(widget_info["widgets"], widget_info["values"])
                    if cb.value
                ]
                if selected and len(selected) < len(widget_info["values"]):
                    filtered = filtered[filtered[col].astype(str).isin(selected)]
            elif isinstance(widget_info, widgets.SelectMultiple):
                # SelectMultiple filter
                if widget_info.value:
                    filtered = filtered[filtered[col].astype(str).isin(widget_info.value)]

        # Apply text search
        if description_search.value and "SeriesDescription" in filtered.columns:
            mask = filtered["SeriesDescription"].fillna("").str.contains(
                description_search.value, case=False, regex=False
            )
            filtered = filtered[mask]

        return filtered

    def update_stats(filtered: pd.DataFrame):
        """Update the statistics panel."""
        total = len(df)
        shown = len(filtered)
        n_projects = filtered["xnat_project_id"].nunique() if "xnat_project_id" in filtered.columns else 0
        n_experiments = filtered["xnat_experiment_label"].nunique() if "xnat_experiment_label" in filtered.columns else 0

        modality_summary = ""
        if "Modality" in filtered.columns and not filtered.empty:
            counts = filtered["Modality"].value_counts()
            modality_summary = " | ".join([f"{k}: {v}" for k, v in counts.head(5).items()])

        html = f"""
        <div style="background: #f8f9fa; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <h4 style="margin-top: 0; color: #333;">Summary Statistics</h4>
            <table style="width: 100%;">
                <tr>
                    <td><b>Showing:</b></td>
                    <td>{shown:,} / {total:,} files ({100*shown/total:.1f}%)</td>
                </tr>
                <tr>
                    <td><b>Projects:</b></td>
                    <td>{n_projects}</td>
                </tr>
                <tr>
                    <td><b>Experiments:</b></td>
                    <td>{n_experiments}</td>
                </tr>
                <tr>
                    <td><b>Modalities:</b></td>
                    <td>{modality_summary if modality_summary else 'N/A'}</td>
                </tr>
            </table>
        </div>
        """
        with stats_output:
            stats_output.clear_output(wait=True)
            display(widgets.HTML(html))

    def update_visualization(filtered: pd.DataFrame):
        """Update the distribution visualization."""
        col = viz_column.value
        if col == "(none)" or col not in filtered.columns:
            return

        with viz_output:
            viz_output.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8, 4))

            counts = filtered[col].value_counts().head(15)

            if viz_type.value == "prop":
                counts = counts / counts.sum()
                ylabel = "Proportion"
            else:
                ylabel = "Count"

            colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
            counts.plot(kind="barh", ax=ax, color=colors)

            ax.set_xlabel(ylabel)
            ax.set_ylabel(col)
            ax.set_title(f"Distribution of {col} (filtered data)")
            ax.invert_yaxis()

            plt.tight_layout()
            plt.show()

    def update_table(filtered: pd.DataFrame):
        """Update the data table preview."""
        with table_output:
            table_output.clear_output(wait=True)
            if filtered.empty:
                print("No data matches the current filters.")
            else:
                # Show a subset of columns for readability
                display_cols = [
                    "PatientID", "Modality", "SeriesDescription",
                    "BodyPartExamined", "xnat_experiment_label", "xnat_scan_id",
                ]
                available_display = [c for c in display_cols if c in filtered.columns]
                display(filtered[available_display].head(20))

    def update_all(change=None):
        """Refresh all dashboard components."""
        filtered = apply_filters()
        state["filtered_df"] = filtered
        update_stats(filtered)
        update_visualization(filtered)
        update_table(filtered)

    def reset_filters(btn):
        """Reset all filters to default state."""
        for col, widget_info in filter_widgets.items():
            if isinstance(widget_info, dict) and widget_info["type"] == "checkboxes":
                for cb in widget_info["widgets"]:
                    cb.value = True
            elif isinstance(widget_info, widgets.SelectMultiple):
                widget_info.value = tuple(widget_info.options)
        description_search.value = ""
        update_all()

    def export_csv(btn):
        """Export filtered DataFrame to CSV."""
        state["filtered_df"].to_csv(export_path, index=False)
        with table_output:
            print(f"\nExported {len(state['filtered_df'])} rows to {export_path}")

    # -------------------------------------------------------------------------
    # Wire up observers
    # -------------------------------------------------------------------------

    description_search.observe(update_all, names="value")
    viz_column.observe(update_all, names="value")
    viz_type.observe(update_all, names="value")
    reset_btn.on_click(reset_filters)
    export_btn.on_click(export_csv)

    for col, widget_info in filter_widgets.items():
        if isinstance(widget_info, dict) and widget_info["type"] == "checkboxes":
            for cb in widget_info["widgets"]:
                cb.observe(update_all, names="value")
        elif isinstance(widget_info, widgets.SelectMultiple):
            widget_info.observe(update_all, names="value")

    # -------------------------------------------------------------------------
    # Build layout
    # -------------------------------------------------------------------------

    # Build filter section
    filter_rows = []
    for col, widget_info in filter_widgets.items():
        if isinstance(widget_info, dict) and widget_info["type"] == "checkboxes":
            label = widgets.HTML(f"<b>{col}:</b>")
            checkbox_row = HBox(
                widget_info["widgets"],
                layout=Layout(flex_flow="row wrap", gap="5px"),
            )
            filter_rows.append(VBox([label, checkbox_row]))
        elif isinstance(widget_info, widgets.SelectMultiple):
            filter_rows.append(widget_info)

    filters_container = VBox(
        filter_rows,
        layout=Layout(padding="10px", border="1px solid #ddd", border_radius="5px"),
    )

    search_row = HBox([description_search, reset_btn, export_btn])
    viz_controls = HBox([viz_column, viz_type])

    # Main dashboard layout
    dashboard = VBox([
        widgets.HTML("<h3>XNAT Metadata Explorer</h3>"),
        widgets.HTML("<p><i>Use the filters below to explore your DICOM metadata</i></p>"),
        filters_container,
        search_row,
        widgets.HTML("<hr>"),
        stats_output,
        viz_controls,
        viz_output,
        widgets.HTML("<h4>Data Preview (first 20 rows)</h4>"),
        table_output,
    ])

    # Initial render
    update_all()

    return dashboard


# -------------------------------------------------------------------------
# Counts summary dashboard
# -------------------------------------------------------------------------

def counts_summary_dashboard(df: pd.DataFrame) -> VBox:
    """
    Simple dashboard showing counts/summaries for quick data overview.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing DICOM metadata.

    Returns
    -------
    ipywidgets.VBox
    """
    if df.empty:
        return VBox([widgets.HTML("<p>No data available.</p>")])

    out = widgets.Output()

    with out:
        # Overall stats
        html = f"""
        <div style="background: #e7f3fe; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3 style="margin-top: 0;">Dataset Overview</h3>
            <table style="width: 100%; font-size: 14px;">
                <tr><td><b>Total DICOM files:</b></td><td>{len(df):,}</td></tr>
        """

        if "xnat_project_id" in df.columns:
            html += f"<tr><td><b>Projects:</b></td><td>{df['xnat_project_id'].nunique()}</td></tr>"
        if "xnat_experiment_label" in df.columns:
            html += f"<tr><td><b>Experiments/Sessions:</b></td><td>{df['xnat_experiment_label'].nunique()}</td></tr>"
        if "PatientID" in df.columns:
            html += f"<tr><td><b>Unique Patients:</b></td><td>{df['PatientID'].nunique()}</td></tr>"
        if "Modality" in df.columns:
            html += f"<tr><td><b>Modalities:</b></td><td>{', '.join(df['Modality'].dropna().unique())}</td></tr>"

        html += "</table></div>"
        display(widgets.HTML(html))

        # Modality breakdown
        if "Modality" in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Bar chart
            modality_counts = df["Modality"].value_counts()
            modality_counts.plot(kind="bar", ax=axes[0], color="steelblue")
            axes[0].set_title("Files by Modality")
            axes[0].set_ylabel("Count")
            axes[0].tick_params(axis="x", rotation=45)

            # Pie chart
            modality_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%")
            axes[1].set_ylabel("")
            axes[1].set_title("Modality Distribution")

            plt.tight_layout()
            plt.show()

        # Body part breakdown if available
        if "BodyPartExamined" in df.columns and df["BodyPartExamined"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 4))
            body_counts = df["BodyPartExamined"].value_counts().head(15)
            body_counts.plot(kind="barh", ax=ax, color="teal")
            ax.set_xlabel("Count")
            ax.set_title("Files by Body Part Examined (Top 15)")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.show()

    return VBox([out])
