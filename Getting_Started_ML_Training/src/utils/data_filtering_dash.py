"""
Data filtering dashboard for XNAT metadata.

This module provides an interactive dashboard for filtering and exploring
XNAT metadata with visualizations.
"""

from typing import Optional, List, Callable

import numpy as np
import pandas as pd
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import matplotlib.pyplot as plt


class DataFilterDashboard:
    """
    Container class for the data filter dashboard that provides access
    to the filtered/exported DataFrame.

    Attributes
    ----------
    widget : VBox
        The dashboard widget to display.
    exported_df : pd.DataFrame or None
        The DataFrame that was last exported via the Export CSV button.

    Methods
    -------
    get_filtered_df()
        Returns the currently filtered DataFrame.
    get_exported_df()
        Returns the last exported DataFrame.
    """

    def __init__(self):
        self.widget = None
        self.exported_df = None
        self._state = None

    def get_filtered_df(self) -> pd.DataFrame:
        """Return the currently filtered DataFrame."""
        if self._state is not None:
            return self._state.get("filtered_df", pd.DataFrame())
        return pd.DataFrame()

    def get_exported_df(self) -> pd.DataFrame:
        """Return the last exported DataFrame."""
        if self.exported_df is not None:
            return self.exported_df
        return pd.DataFrame()

    def _repr_mimebundle_(self, **kwargs):
        """Allow the dashboard to be displayed directly in Jupyter."""
        if self.widget is not None:
            return self.widget._repr_mimebundle_(**kwargs)
        return {"text/plain": "Dashboard not initialized"}

    def _ipython_display_(self):
        """Display the dashboard widget in IPython/Jupyter."""
        if self.widget is not None:
            from IPython.display import display
            display(self.widget)


def data_filter_dashboard(
    df: pd.DataFrame,
    filter_columns: Optional[List[str]] = None,
    on_filter_callback: Optional[Callable[[pd.DataFrame], None]] = None,
) -> DataFilterDashboard:
    """
    Interactive dashboard to filter XNAT metadata and create data subsets.

    Features a professional layout with:
    - Header with progress/summary statistics
    - Visualizations and data preview in the main area
    - Collapsible control panel for filtering on the left side
    - User-selectable filter columns

    Parameters
    ----------
    df : pd.DataFrame
        The metadata DataFrame from fetch_xnat_metadata().
    filter_columns : list of str, optional
        Initial columns to use for filtering. Users can add/remove columns
        dynamically through the interface.
    on_filter_callback : callable, optional
        A callback function that receives the filtered DataFrame whenever
        the filters change. Useful for chaining with other processing.

    Returns
    -------
    DataFilterDashboard
        A dashboard object that can be displayed and provides access to filtered data.
        - Display the dashboard by putting it as the last line in a cell or using display()
        - Access the current filtered DataFrame via dashboard.get_filtered_df()
        - Access the last exported DataFrame via dashboard.get_exported_df()

    Examples
    --------
    >>> from src.utils.data_filtering_dash import data_filter_dashboard
    >>> dashboard = data_filter_dashboard(meta_df)
    >>> dashboard  # displays the dashboard

    Access filtered data for downstream use:
    >>> dashboard = data_filter_dashboard(meta_df)
    >>> # ... user interacts with filters ...
    >>> filtered_df = dashboard.get_filtered_df()  # get current filtered data
    >>> exported_df = dashboard.get_exported_df()  # get last exported data

    With callback to capture filtered data on every change:
    >>> filtered_data = {}
    >>> def capture_filter(df):
    ...     filtered_data['current'] = df
    >>> dashboard = data_filter_dashboard(meta_df, on_filter_callback=capture_filter)
    """
    # Create the dashboard container object
    dashboard_obj = DataFilterDashboard()

    if df is None or df.empty:
        dashboard_obj.widget = VBox([
            widgets.HTML(
                "<p style='color: orange;'>No data provided. "
                "Please pass a valid DataFrame.</p>"
            )
        ])
        return dashboard_obj

    # Preprocess: compute derived columns for filtering
    df = df.copy()

    # Compute voxel volume if pixel spacing and slice thickness are available
    has_pixel_spacing = ("pixel_spacing_row" in df.columns and "pixel_spacing_col" in df.columns)
    has_slice_thickness = "slice_thickness" in df.columns

    if has_pixel_spacing and has_slice_thickness:
        df["voxel_volume_mm3"] = (
            df["pixel_spacing_row"] * df["pixel_spacing_col"] * df["slice_thickness"]
        )
    elif has_pixel_spacing:
        # Compute pixel area if no slice thickness
        df["pixel_area_mm2"] = df["pixel_spacing_row"] * df["pixel_spacing_col"]

    # State to hold current filtered DataFrame and active filters
    state = {
        "filtered_df": df.copy(),
        "original_df": df.copy(),
        "active_filter_columns": [],
        "plot_visible_columns": set(),  # Columns that should show in plots
    }

    # Link state to dashboard object for external access
    dashboard_obj._state = state

    # -------------------------------------------------------------------------
    # Helper: Get all filterable columns
    # -------------------------------------------------------------------------
    def _get_all_filterable_columns(dataframe: pd.DataFrame) -> List[str]:
        """Get all columns that could potentially be used for filtering."""
        # Columns to always exclude from filtering
        excluded_columns = [
            "dicom_files_sample", "file_path", "image_orientation_patient",
            "project_id",  # Exclude project_id - use project_name instead
        ]

        # Columns to always include if they exist (even if all null in some datasets)
        # This list prioritizes commonly useful imaging columns, but the dashboard
        # will dynamically include any column that exists in the DataFrame
        always_include = [
            # Core identifiers
            "project_name", "modality", "body_part_examined",
            # Core imaging metrics
            "slice_thickness", "pixel_spacing_row", "pixel_spacing_col",
            "voxel_volume_mm3", "pixel_area_mm2", "num_slices", "number_of_frames",
            "rows", "columns", "bits_stored", "bits_allocated",
            # Scanner info
            "manufacturer", "manufacturer_model_name",
            # Demographics
            "gender", "patient_sex",
            # MRI specific
            "magnetic_field_strength", "repetition_time", "echo_time",
            "flip_angle", "mr_acquisition_type", "scanning_sequence",
            "inversion_time", "echo_train_length",
            # CT specific
            "kvp", "convolution_kernel", "reconstruction_diameter",
            "gantry_detector_tilt", "exposure", "x_ray_tube_current",
            # PET specific
            "radiopharmaceutical", "decay_correction",
            "attenuation_correction_method", "reconstruction_method",
            # General series/study
            "series_description", "study_description", "protocol_name",
            "image_type", "contrast_bolus_agent",
            # Window settings
            "window_center", "window_width",
            # Temporal
            "temporal_position_identifier", "number_of_temporal_positions",
        ]

        candidates = []
        for col in dataframe.columns:
            # Skip excluded columns
            if col in excluded_columns:
                continue

            # Always include priority columns if they exist
            if col in always_include:
                candidates.append(col)
                continue

            # Skip columns with all null values for non-priority columns
            if dataframe[col].isna().all():
                continue

            # For object/string columns, check cardinality
            if dataframe[col].dtype == "object" or str(dataframe[col].dtype) == "category":
                n_unique = dataframe[col].nunique()
                if 1 < n_unique <= 100:
                    candidates.append(col)
            # For numeric columns
            elif np.issubdtype(dataframe[col].dtype, np.number):
                n_unique = dataframe[col].nunique()
                if 1 < n_unique <= 500:
                    candidates.append(col)
        return candidates

    # -------------------------------------------------------------------------
    # Helper: Get default filter columns
    # -------------------------------------------------------------------------
    def _get_default_filter_columns(dataframe: pd.DataFrame) -> List[str]:
        """Select default columns for filtering."""
        if filter_columns is not None:
            return [c for c in filter_columns if c in dataframe.columns]

        all_cols = _get_all_filterable_columns(dataframe)

        # Prioritize certain columns (removed project_id, added imaging-specific filters)
        priority = [
            "project_name", "modality", "slice_thickness",
            "pixel_spacing_row", "pixel_spacing_col", "voxel_volume_mm3",
            "pixel_area_mm2", "gender", "scan_type",
            "manufacturer", "body_part_examined", "photometric_interpretation",
            "bits_stored", "rows", "columns", "num_slices"
        ]
        ordered = [c for c in priority if c in all_cols]
        ordered += [c for c in all_cols if c not in ordered]

        return ordered[:6]  # Start with 6 default filters

    all_filterable_columns = _get_all_filterable_columns(df)
    state["active_filter_columns"] = _get_default_filter_columns(df)

    # Set default plot visible columns (first 2-3 that would make good plots)
    default_plot_cols = []
    for col in state["active_filter_columns"]:
        if col in df.columns:
            # Good for distributions/pie: categorical or low cardinality
            if df[col].dtype == "object" or df[col].nunique() <= 10:
                default_plot_cols.append(col)
            # Good for histograms: numeric with some variation
            elif np.issubdtype(df[col].dtype, np.number) and df[col].nunique() > 5:
                default_plot_cols.append(col)
        if len(default_plot_cols) >= 3:
            break
    state["plot_visible_columns"] = set(default_plot_cols)

    if not all_filterable_columns:
        dashboard_obj.widget = VBox([
            widgets.HTML(
                "<p style='color: orange;'>No suitable columns found for filtering. "
                "The DataFrame may have too few categorical columns.</p>"
            )
        ])
        return dashboard_obj

    # -------------------------------------------------------------------------
    # Filter widget creation and management
    # -------------------------------------------------------------------------
    filter_widgets = {}
    plot_toggle_widgets = {}  # Checkboxes for toggling plot visibility
    filter_container = VBox()

    def _create_filter_widget(col: str) -> Optional[widgets.Widget]:
        """Create appropriate filter widget based on column type."""
        col_data = df[col].dropna()

        # Handle columns with no non-null values - show placeholder
        if col_data.empty:
            widget = widgets.HTML(
                value="<div style='padding: 10px; color: #666; font-style: italic;'>"
                      "No data available for this filter</div>"
            )
            return widget

        # For categorical/object columns, use SelectMultiple
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            unique_vals = sorted(col_data.unique().astype(str))
            widget = widgets.SelectMultiple(
                options=["(All)"] + unique_vals,
                value=["(All)"],
                description="",
                layout=widgets.Layout(width="100%", height="100px"),
            )
            return widget

        # For numeric columns with few unique values, use SelectMultiple
        elif np.issubdtype(df[col].dtype, np.number):
            n_unique = col_data.nunique()
            if n_unique == 0:
                # No valid numeric values
                widget = widgets.HTML(
                    value="<div style='padding: 10px; color: #666; font-style: italic;'>"
                          "No data available for this filter</div>"
                )
                return widget
            elif n_unique <= 20:
                unique_vals = sorted(col_data.unique())
                unique_vals_str = [str(v) for v in unique_vals]
                widget = widgets.SelectMultiple(
                    options=["(All)"] + unique_vals_str,
                    value=["(All)"],
                    description="",
                    layout=widgets.Layout(width="100%", height="100px"),
                )
                return widget
            else:
                # Use range slider for continuous numeric
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                step = (max_val - min_val) / 100 if max_val > min_val else 1
                widget = widgets.FloatRangeSlider(
                    value=[min_val, max_val],
                    min=min_val,
                    max=max_val,
                    step=step,
                    description="",
                    layout=widgets.Layout(width="100%"),
                    continuous_update=False,
                )
                return widget
        return None

    def _move_filter(col: str, direction: int):
        """Move a filter up (-1) or down (+1) in the list."""
        cols = state["active_filter_columns"]
        idx = cols.index(col)
        new_idx = idx + direction
        if 0 <= new_idx < len(cols):
            cols[idx], cols[new_idx] = cols[new_idx], cols[idx]
            _rebuild_filter_widgets()
            _update()

    def _remove_filter(col: str):
        """Remove a filter from the active list."""
        if col in state["active_filter_columns"]:
            state["active_filter_columns"].remove(col)
            state["plot_visible_columns"].discard(col)
            _rebuild_filter_widgets()
            _update()
            # Refresh search results to show the newly available filter
            _update_search_results(filter_search_input.value)

    def _rebuild_filter_widgets():
        """Rebuild filter widgets based on active columns."""
        # Clear existing
        filter_widgets.clear()
        plot_toggle_widgets.clear()

        filter_boxes = []
        num_filters = len(state["active_filter_columns"])

        for idx, col in enumerate(state["active_filter_columns"]):
            widget = _create_filter_widget(col)
            if widget is not None:
                filter_widgets[col] = widget

                # Only attach observers to interactive widgets (not HTML placeholders)
                if hasattr(widget, "observe"):
                    widget.observe(_update, names="value")

                # Create plot visibility toggle checkbox
                plot_toggle = widgets.Checkbox(
                    value=col in state["plot_visible_columns"],
                    description="Plot",
                    indent=False,
                    layout=widgets.Layout(width="60px"),
                    style={"description_width": "initial"},
                )

                def _make_toggle_handler(column):
                    def handler(change):
                        if change["new"]:
                            state["plot_visible_columns"].add(column)
                        else:
                            state["plot_visible_columns"].discard(column)
                        _update()
                    return handler

                plot_toggle.observe(_make_toggle_handler(col), names="value")
                plot_toggle_widgets[col] = plot_toggle

                # Clean header with just the title (no buttons)
                header_container = widgets.HTML(
                    f"<div style='font-weight: bold; font-size: 11px; color: white; "
                    f"background: #2c3e50; padding: 6px 10px; border-radius: 4px 4px 0 0;'>"
                    f"{col.replace('_', ' ').title()}</div>"
                )

                # Toggle row below the filter
                toggle_row = HBox([
                    plot_toggle,
                ], layout=widgets.Layout(
                    padding="2px 10px",
                    background="#ecf0f1",
                ))

                filter_box = VBox(
                    [header_container, widget, toggle_row],
                    layout=widgets.Layout(
                        border="1px solid #bdc3c7",
                        border_radius="4px",
                        margin="5px 0",
                    )
                )
                filter_boxes.append(filter_box)

        filter_container.children = filter_boxes

    # -------------------------------------------------------------------------
    # Column selector for adding/removing filters (with search)
    # -------------------------------------------------------------------------

    # State for search visibility
    search_state = {"is_open": False, "panel": None}

    # Search input for filtering available columns
    filter_search_input = widgets.Text(
        placeholder="Type to search...",
        layout=widgets.Layout(width="100%"),
    )

    # Container for search results (initially hidden)
    filter_search_results = VBox(layout=widgets.Layout(
        max_height="250px",
        overflow_y="auto",
        width="100%",
    ))

    def _get_available_filters():
        """Get list of filters not currently active."""
        return [c for c in all_filterable_columns if c not in state["active_filter_columns"]]

    def _hide_search_panel():
        """Hide the search panel."""
        search_state["is_open"] = False
        if search_state["panel"] is not None:
            search_state["panel"].layout.display = "none"

    def _show_search_panel():
        """Show the search panel."""
        search_state["is_open"] = True
        if search_state["panel"] is not None:
            search_state["panel"].layout.display = "block"
        _update_search_results(filter_search_input.value)

    def _update_search_results(search_text: str = ""):
        """Update the search results based on search text."""
        available = _get_available_filters()
        search_lower = search_text.lower().strip()

        if search_lower:
            # Filter by search text
            matches = [c for c in available if search_lower in c.lower().replace("_", " ")]
        else:
            # Show all available when no search
            matches = available

        # Create clickable buttons for each match
        result_buttons = []
        for col in matches[:20]:  # Limit to 20 results
            btn = widgets.Button(
                description=col.replace("_", " ").title(),
                layout=widgets.Layout(width="100%", margin="1px 0"),
                button_style="",
            )

            def _make_add_handler(column):
                def handler(b):
                    if column not in state["active_filter_columns"]:
                        state["active_filter_columns"].append(column)
                        filter_search_input.value = ""
                        _hide_search_panel()
                        _rebuild_filter_widgets()
                        _update()
                return handler

            btn.on_click(_make_add_handler(col))
            result_buttons.append(btn)

        if not result_buttons and search_text:
            result_buttons = [widgets.HTML(
                "<div style='padding: 8px; color: #666; font-style: italic;'>"
                "No matching filters found</div>"
            )]
        elif not result_buttons:
            result_buttons = [widgets.HTML(
                "<div style='padding: 8px; color: #666; font-style: italic;'>"
                "All filters are active</div>"
            )]

        # Add count indicator and close button
        header_items = []
        if available:
            header_items.append(widgets.HTML(
                f"<div style='padding: 4px 0; font-size: 10px; color: #888;'>"
                f"Showing {len(matches[:20])} of {len(available)} available</div>"
            ))

        close_btn = widgets.Button(
            description="Close",
            button_style="",
            layout=widgets.Layout(width="60px", height="22px", padding="0"),
        )
        close_btn.on_click(lambda b: _hide_search_panel())

        header_row = HBox([
            header_items[0] if header_items else widgets.HTML(""),
            close_btn,
        ], layout=widgets.Layout(justify_content="space-between", align_items="center"))

        filter_search_results.children = [header_row] + result_buttons

    def _on_search_change(change):
        if search_state["is_open"]:
            _update_search_results(change["new"])

    filter_search_input.observe(_on_search_change, names="value")

    # Button to open/toggle search
    open_search_btn = widgets.Button(
        description="+ Add Filter",
        button_style="info",
        layout=widgets.Layout(width="100%", margin="0"),
    )

    # Search panel container (initially hidden)
    search_panel_container = VBox([
        filter_search_input,
        filter_search_results,
    ], layout=widgets.Layout(
        border="1px solid #ddd",
        border_radius="4px",
        padding="8px",
        margin="5px 0 0 0",
        background="#fafafa",
        display="none",
    ))

    # Store reference in state for show/hide functions
    search_state["panel"] = search_panel_container

    def _toggle_search_panel(b):
        if search_state["is_open"]:
            _hide_search_panel()
        else:
            _show_search_panel()

    open_search_btn.on_click(_toggle_search_panel)

    # Combined search widget
    add_filter_container = VBox([
        open_search_btn,
        search_panel_container,
    ])

    # -------------------------------------------------------------------------
    # Manage Filters section (reorder and remove)
    # -------------------------------------------------------------------------
    manage_state = {"is_open": False, "panel": None}
    manage_filters_list = VBox()

    def _rebuild_manage_filters_list():
        """Rebuild the list of filters in the manage panel."""
        items = []
        num_filters = len(state["active_filter_columns"])

        for idx, col in enumerate(state["active_filter_columns"]):
            # Create compact row with filter name and controls
            name_label = widgets.HTML(
                f"<span style='font-size: 11px;'>{idx + 1}. {col.replace('_', ' ').title()}</span>"
            )

            up_btn = widgets.Button(
                description="↑",
                layout=widgets.Layout(width="24px", height="20px", padding="0"),
                disabled=(idx == 0),
            )
            down_btn = widgets.Button(
                description="↓",
                layout=widgets.Layout(width="24px", height="20px", padding="0"),
                disabled=(idx == num_filters - 1),
            )
            remove_btn = widgets.Button(
                description="×",
                button_style="danger",
                layout=widgets.Layout(width="24px", height="20px", padding="0"),
            )

            def _make_move_handler(column, direction):
                def handler(btn):
                    _move_filter(column, direction)
                    _rebuild_manage_filters_list()
                return handler

            def _make_remove_handler(column):
                def handler(btn):
                    _remove_filter(column)
                    _rebuild_manage_filters_list()
                return handler

            up_btn.on_click(_make_move_handler(col, -1))
            down_btn.on_click(_make_move_handler(col, 1))
            remove_btn.on_click(_make_remove_handler(col))

            row = HBox([
                name_label,
                HBox([up_btn, down_btn, remove_btn], layout=widgets.Layout(gap="2px")),
            ], layout=widgets.Layout(
                justify_content="space-between",
                align_items="center",
                padding="3px 5px",
                border_bottom="1px solid #eee",
            ))
            items.append(row)

        if not items:
            items = [widgets.HTML(
                "<div style='padding: 10px; color: #666; font-style: italic;'>"
                "No active filters</div>"
            )]

        manage_filters_list.children = items

    def _hide_manage_panel():
        """Hide the manage filters panel."""
        manage_state["is_open"] = False
        if manage_state["panel"] is not None:
            manage_state["panel"].layout.display = "none"

    def _show_manage_panel():
        """Show the manage filters panel."""
        manage_state["is_open"] = True
        if manage_state["panel"] is not None:
            manage_state["panel"].layout.display = "block"
        _rebuild_manage_filters_list()

    def _toggle_manage_panel(b):
        if manage_state["is_open"]:
            _hide_manage_panel()
        else:
            _show_manage_panel()

    manage_filters_btn = widgets.Button(
        description="⚙ Manage Filters",
        button_style="",
        layout=widgets.Layout(width="100%", margin="5px 0 0 0"),
    )
    manage_filters_btn.on_click(_toggle_manage_panel)

    manage_close_btn = widgets.Button(
        description="Close",
        button_style="",
        layout=widgets.Layout(width="50px", height="20px", padding="0"),
    )
    manage_close_btn.on_click(lambda b: _hide_manage_panel())

    manage_panel_header = HBox([
        widgets.HTML("<span style='font-size: 11px; font-weight: bold;'>Reorder / Remove</span>"),
        manage_close_btn,
    ], layout=widgets.Layout(justify_content="space-between", align_items="center", padding="4px"))

    manage_panel_container = VBox([
        manage_panel_header,
        manage_filters_list,
    ], layout=widgets.Layout(
        border="1px solid #ddd",
        border_radius="4px",
        padding="5px",
        margin="5px 0 0 0",
        background="#fafafa",
        max_height="200px",
        overflow_y="auto",
        display="none",
    ))

    manage_state["panel"] = manage_panel_container

    # Combined manage filters widget
    manage_filter_container = VBox([
        manage_filters_btn,
        manage_panel_container,
    ])

    # -------------------------------------------------------------------------
    # Summary statistics cards (header area)
    # -------------------------------------------------------------------------
    summary_cards_html = widgets.HTML(value="")

    def _update_summary_cards(filtered: pd.DataFrame):
        """Update the summary statistics cards at the top."""
        total = len(state["original_df"])
        filtered_count = len(filtered)
        pct = (filtered_count / total * 100) if total > 0 else 0

        # Count unique values
        n_projects = filtered["project_name"].nunique() if "project_name" in filtered.columns else 0
        n_subjects = filtered["subject_id"].nunique() if "subject_id" in filtered.columns else 0
        n_experiments = filtered["experiment_id"].nunique() if "experiment_id" in filtered.columns else 0
        n_scans = filtered["scan_id"].nunique() if "scan_id" in filtered.columns else 0

        # Progress bar width
        progress_pct = min(100, pct)

        html = f"""
        <div style="background: linear-gradient(to right, #1a365d 0%, #2c5282 25%, #d69e2e 50%, #48bb78 75%, #276749 100%);
                    padding: 20px; border-radius: 8px; margin-bottom: 15px; color: white;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <div>
                    <h2 style="margin: 0; font-size: 24px;">XNAT Metadata Filter Dashboard</h2>
                    <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 14px;">
                        Filter and explore imaging metadata across projects
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 32px; font-weight: bold;">{filtered_count:,} / {total:,}</div>
                    <div style="font-size: 12px; opacity: 0.8;">Records Selected ({pct:.1f}%)</div>
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.2); border-radius: 4px; height: 8px; margin-bottom: 15px;">
                <div style="background: #48bb78; height: 100%; border-radius: 4px; width: {progress_pct}%;"></div>
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_projects}</div>
                    <div style="font-size: 12px; opacity: 0.8;">PROJECTS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_subjects}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SUBJECTS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_experiments}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SESSIONS</div>
                </div>
                <div style="background: rgba(255,255,255,0.15); padding: 15px 25px; border-radius: 6px; text-align: center; flex: 1; min-width: 120px;">
                    <div style="font-size: 28px; font-weight: bold;">{n_scans}</div>
                    <div style="font-size: 12px; opacity: 0.8;">SCANS</div>
                </div>
            </div>
        </div>
        """
        summary_cards_html.value = html

    # -------------------------------------------------------------------------
    # Visualization outputs
    # -------------------------------------------------------------------------
    out_plots = widgets.Output()
    out_table = widgets.Output()

    # Plot type selector
    plot_selector = widgets.ToggleButtons(
        options=[
            ("Distributions", "dist"),
            ("Pie Charts", "pie"),
            ("Histograms", "hist"),
            ("Data Table", "table"),
        ],
        value="dist",
        layout=widgets.Layout(margin="0 0 10px 0"),
    )

    def _plot_distributions(filtered: pd.DataFrame):
        """Plot distribution bar charts for categorical columns with plot visibility enabled."""
        # Only plot columns that have plot visibility toggled ON
        plot_cols = [c for c in state["plot_visible_columns"]
                    if c in filtered.columns and
                    (filtered[c].dtype == "object" or filtered[c].nunique() <= 20)][:6]

        if not plot_cols:
            print("No columns selected for plotting. Use 'Show in plots' checkbox to enable plots for filters.")
            return

        n_cols = min(3, len(plot_cols))
        n_rows = (len(plot_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        colors_palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
                         "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b"]

        for idx, col in enumerate(plot_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]

            value_counts = filtered[col].value_counts().head(8)
            colors = [colors_palette[i % len(colors_palette)] for i in range(len(value_counts))]

            bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels([str(v)[:25] for v in value_counts.index], fontsize=9)
            ax.set_xlabel("Count", fontsize=10)
            ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
            ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for bar, count in zip(bars, value_counts.values):
                ax.text(bar.get_width() + max(value_counts) * 0.02,
                       bar.get_y() + bar.get_height()/2,
                       f"{count:,}", va="center", fontsize=9)

        for idx in range(len(plot_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    def _plot_pie_charts(filtered: pd.DataFrame):
        """Plot pie charts for columns with plot visibility enabled."""
        # Only plot columns that have plot visibility toggled ON and are suitable for pie charts
        pie_cols = [c for c in state["plot_visible_columns"]
                   if c in filtered.columns and 1 < filtered[c].nunique() <= 10][:4]

        if not pie_cols:
            print("No columns selected for pie charts. Use 'Show in plots' checkbox to enable plots for filters.")
            print("(Pie charts work best with columns having 2-10 unique values)")
            return

        n_cols = min(2, len(pie_cols))
        n_rows = (len(pie_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        colors_palette = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
                         "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b"]

        for idx, col in enumerate(pie_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]

            value_counts = filtered[col].value_counts()
            colors = [colors_palette[i % len(colors_palette)] for i in range(len(value_counts))]

            wedges, texts, autotexts = ax.pie(
                value_counts.values,
                labels=None,
                autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
                colors=colors,
                startangle=90,
                pctdistance=0.75,
            )
            ax.set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")

            # Add legend
            ax.legend(
                wedges,
                [f"{str(v)[:20]}: {c:,}" for v, c in zip(value_counts.index, value_counts.values)],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=9,
            )

        for idx in range(len(pie_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    def _plot_numeric_histograms(filtered: pd.DataFrame):
        """Plot histograms for numeric columns with plot visibility enabled."""
        # Only plot columns that have plot visibility toggled ON and are numeric
        numeric_cols = [c for c in state["plot_visible_columns"]
                       if c in filtered.columns and
                       np.issubdtype(filtered[c].dtype, np.number) and
                       filtered[c].nunique() > 2][:6]

        if not numeric_cols:
            print("No numeric columns selected for histograms. Use 'Show in plots' checkbox to enable plots for filters.")
            return

        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, col in enumerate(numeric_cols):
            row, col_idx = divmod(idx, n_cols)
            ax = axes[row, col_idx]

            data = filtered[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
                ax.axvline(data.mean(), color="#e74c3c", linestyle="--", linewidth=2,
                          label=f"Mean: {data.mean():.2f}")
                ax.axvline(data.median(), color="#2ecc71", linestyle="--", linewidth=2,
                          label=f"Median: {data.median():.2f}")
                ax.set_xlabel(col.replace("_", " ").title(), fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                ax.set_title(f"{col.replace('_', ' ').title()} (n={len(data):,})",
                           fontsize=11, fontweight="bold")
                ax.legend(fontsize=8, loc="upper right")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
                ax.set_title(col.replace("_", " ").title(), fontweight="bold")
                ax.axis("off")

        for idx in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = divmod(idx, n_cols)
            axes[row, col_idx].axis("off")

        plt.tight_layout()
        plt.show()

    def _update_table(filtered: pd.DataFrame):
        """Display a sample of the filtered DataFrame."""
        with out_table:
            out_table.clear_output(wait=True)

            display_cols = [c for c in filtered.columns
                          if c not in ["dicom_files_sample", "image_orientation_patient"]][:15]

            if len(filtered) > 0:
                sample = filtered[display_cols].head(15)
                # Style the dataframe
                print(f"Showing {len(sample)} of {len(filtered):,} filtered records\n")
                display(sample)
            else:
                print("No records match the current filters.")

    # -------------------------------------------------------------------------
    # Apply filters
    # -------------------------------------------------------------------------
    def _apply_filters() -> pd.DataFrame:
        """Apply all current filter selections to the DataFrame."""
        filtered = state["original_df"].copy()

        for col, widget in filter_widgets.items():
            if col not in filtered.columns:
                continue

            if isinstance(widget, widgets.SelectMultiple):
                selected = list(widget.value)
                if "(All)" not in selected and selected:
                    if np.issubdtype(df[col].dtype, np.number):
                        selected_vals = [float(v) for v in selected]
                        filtered = filtered[filtered[col].isin(selected_vals) | filtered[col].isna()]
                    else:
                        filtered = filtered[filtered[col].astype(str).isin(selected) | filtered[col].isna()]

            elif isinstance(widget, widgets.FloatRangeSlider):
                min_val, max_val = widget.value
                filtered = filtered[
                    ((filtered[col] >= min_val) & (filtered[col] <= max_val)) |
                    filtered[col].isna()
                ]

        return filtered

    # -------------------------------------------------------------------------
    # Main update function
    # -------------------------------------------------------------------------
    def _update(change=None):
        """Main update function triggered by any filter change."""
        filtered = _apply_filters()
        state["filtered_df"] = filtered

        _update_summary_cards(filtered)

        with out_plots:
            out_plots.clear_output(wait=True)

            if len(filtered) == 0:
                print("No data matches the current filters.")
            else:
                plot_type = plot_selector.value
                if plot_type == "dist":
                    _plot_distributions(filtered)
                elif plot_type == "pie":
                    _plot_pie_charts(filtered)
                elif plot_type == "hist":
                    _plot_numeric_histograms(filtered)
                elif plot_type == "table":
                    _update_table(filtered)

        # Always update table in background for callback
        if plot_selector.value != "table":
            _update_table(filtered)

        if on_filter_callback is not None:
            on_filter_callback(filtered)

    # -------------------------------------------------------------------------
    # Button handlers
    # -------------------------------------------------------------------------
    reset_btn = widgets.Button(
        description="Reset Filters",
        button_style="warning",
        icon="refresh",
        layout=widgets.Layout(width="100%", margin="5px 0"),
    )

    export_btn = widgets.Button(
        description="Export CSV",
        button_style="success",
        icon="download",
        layout=widgets.Layout(width="100%", margin="5px 0"),
    )

    def _reset_filters(btn):
        """Reset all filters to default values."""
        for col, widget in filter_widgets.items():
            if isinstance(widget, widgets.SelectMultiple):
                widget.value = ["(All)"]
            elif isinstance(widget, widgets.FloatRangeSlider):
                widget.value = [widget.min, widget.max]
        _update()

    def _export_csv(btn):
        """Export the filtered DataFrame to CSV and store for downstream use."""
        from datetime import datetime
        from pathlib import Path

        filtered = state["filtered_df"]
        if len(filtered) == 0:
            with out_table:
                print("No data to export.")
            return

        export_dir = Path("logs/exported_data")
        export_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = export_dir / f"filtered_metadata_{timestamp}.csv"

        export_cols = [c for c in filtered.columns if c != "dicom_files_sample"]
        exported_df = filtered[export_cols].copy()
        exported_df.to_csv(filename, index=False)

        # Store the exported DataFrame in the dashboard object for downstream access
        dashboard_obj.exported_df = exported_df

        with out_table:
            print(f"\n✓ Exported {len(exported_df):,} records to: {filename}")
            print(f"   Access via: dashboard.get_exported_df()")

    reset_btn.on_click(_reset_filters)
    export_btn.on_click(_export_csv)

    # -------------------------------------------------------------------------
    # Wire up observers
    # -------------------------------------------------------------------------
    plot_selector.observe(_update, names="value")

    # -------------------------------------------------------------------------
    # Build initial filter widgets
    # -------------------------------------------------------------------------
    _rebuild_filter_widgets()

    # Initial render
    _update()

    # -------------------------------------------------------------------------
    # Layout: Control Panel (left) + Main Content (right)
    # -------------------------------------------------------------------------

    # Control panel header
    control_panel_header = widgets.HTML(
        """<div style="background: #34495e; color: white; padding: 12px;
           border-radius: 6px 6px 0 0; font-weight: bold;">
           <span style="font-size: 14px;">Control Panel</span>
        </div>"""
    )

    # Filter section
    filter_section_header = widgets.HTML(
        """<div style="background: #ecf0f1; padding: 8px 12px; border-bottom: 1px solid #bdc3c7;">
           <div style="font-weight: bold; font-size: 12px; color: #2c3e50;">Filters</div>
           <div style="font-size: 10px; color: #7f8c8d; margin-top: 2px;">
               Check 'Plot' to show in visualizations
           </div>
        </div>"""
    )

    # Control panel content
    control_panel_content = VBox([
        filter_section_header,
        filter_container,
        add_filter_container,
        manage_filter_container,
        widgets.HTML("<hr style='margin: 10px 0; border-color: #ecf0f1;'>"),
        reset_btn,
        export_btn,
    ], layout=widgets.Layout(
        padding="0 10px 10px 10px",
        background="#f8f9fa",
    ))

    control_panel = VBox([
        control_panel_header,
        control_panel_content,
    ], layout=widgets.Layout(
        width="280px",
        border="1px solid #bdc3c7",
        border_radius="6px",
        margin="0 15px 0 0",
    ))

    # Main content area
    viz_header = widgets.HTML(
        """<div style="background: #ecf0f1; padding: 10px 15px; border-radius: 6px;
           margin-bottom: 10px;">
           <span style="font-weight: bold; color: #2c3e50;">Visualization</span>
        </div>"""
    )

    main_content = VBox([
        plot_selector,
        out_plots,
        widgets.HTML("<hr style='margin: 15px 0;'>"),
        widgets.HTML("<div style='font-weight: bold; margin-bottom: 10px;'>Data Preview</div>"),
        out_table,
    ], layout=widgets.Layout(
        flex="1",
    ))

    # Main layout: header on top, then control panel + content side by side
    content_row = HBox([
        control_panel,
        main_content,
    ], layout=widgets.Layout(
        width="100%",
    ))

    dashboard_widget = VBox([
        summary_cards_html,
        content_row,
    ], layout=widgets.Layout(
        width="100%",
    ))

    # Attach the widget to the dashboard object
    dashboard_obj.widget = dashboard_widget

    return dashboard_obj
