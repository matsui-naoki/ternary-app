"""
Ternary Plot Visualizer - Streamlit Web Application
Three-component composition diagram visualization tool for academic papers and presentations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from typing import Dict
from scipy.interpolate import griddata

# Import utility modules
from utils import (
    subscript_num_html,
    factorize_composition,
    parse_uploaded_file,
    get_sample_data,
    PYMATGEN_AVAILABLE,
    SYMPY_AVAILABLE,
    INTERPOLATION_METHODS,
    HEATMAP_MARKER_MODES,
    LOG_SCALE_HELP,
    DISCRETE_COLORS_HELP,
)

# Import Composition for data table (need it directly)
if PYMATGEN_AVAILABLE:
    from pymatgen.core import Composition


# Page configuration
st.set_page_config(
    page_title="Ternary Plot Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply Arial font globally
st.markdown("""
<style>
    * {
        font-family: Arial, sans-serif !important;
    }
    .stMarkdown, .stText, .stDataFrame, .stTable {
        font-family: Arial, sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame({
            'A': pd.Series(dtype='float64'),
            'B': pd.Series(dtype='float64'),
            'C': pd.Series(dtype='float64'),
            'Z': pd.Series(dtype='float64'),
            'Name': pd.Series(dtype='str')
        })

    if 'labels' not in st.session_state:
        st.session_state.labels = {'A': 'A', 'B': 'B', 'C': 'C', 'Z': 'Z'}

    if 'z_label_preset' not in st.session_state:
        st.session_state.z_label_preset = 'custom'

    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {}

    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

    if 'loaded_file_data' not in st.session_state:
        st.session_state.loaded_file_data = None

    # Initialize data version for tracking changes
    if 'data_version' not in st.session_state:
        st.session_state.data_version = 0

    # Initialize widget defaults for plot settings
    widget_defaults = {
        'ps_fig_width': 700,
        'ps_fig_height': 600,
        'ps_colorscale': 'Turbo',
        'ps_reverse_colorscale': False,
        'ps_auto_z_range': True,
        'ps_log_scale': False,
        'ps_marker_size': 8,
        'ps_marker_symbol': 'circle',
        'ps_marker_line_width': 1,
        'ps_marker_opacity': 0.8,
        'ps_use_single_color': False,
        'ps_single_color': '#1f77b4',
        'ps_axis_line_width': 2,
        'ps_show_grid': True,
        'ps_grid_color': '#808080',
        'ps_grid_line_width': 1,
        'ps_show_tick_labels': False,
        'ps_auto_subscript': True,
        'ps_axis_font_size': 24,
        'ps_tick_font_size': 14,
        'ps_tick_step': 0.1,
        'ps_show_colorbar': True,
        'ps_colorbar_len': 0.6,
        'ps_colorbar_thickness': 20,
        'ps_colorbar_x': 1.02,
        'ps_colorbar_y': 0.5,
        'ps_colorbar_ticks': True,
        'ps_colorbar_title_side': 'right',
        'ps_discrete_colors': True,
        'ps_discrete_steps': 10,
        'ps_heatmap_enabled': False,
        'ps_heatmap_resolution': 100,
        'ps_heatmap_method': 'linear',
        'ps_heatmap_marker_mode': 'fill',
        'ps_heatmap_marker_size': 8,
        'ps_heatmap_symbol': 'hexagon2',
        'ps_heatmap_opacity': 1.0,
        'ps_margin_top': 40,
        'ps_margin_bottom': 60,
        'ps_margin_left': 60,
        'ps_margin_right': 60,
    }
    for key, default in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def create_ternary_plot(data: pd.DataFrame, labels: Dict[str, str], settings: Dict) -> go.Figure:
    """Create ternary plot with given data and settings."""
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'ternary'}]])

    font_family = "Arial, sans-serif"
    font_color = "black"

    if settings.get('auto_subscript', True):
        a_label = subscript_num_html(labels.get('A', 'A'))
        b_label = subscript_num_html(labels.get('B', 'B'))
        c_label = subscript_num_html(labels.get('C', 'C'))
    else:
        a_label = labels.get('A', 'A')
        b_label = labels.get('B', 'B')
        c_label = labels.get('C', 'C')

    if settings.get('auto_z_range', True) or settings.get('z_min') is None or settings.get('z_max') is None:
        if 'Z' in data.columns and len(data) > 0 and data['Z'].notna().any():
            z_min = float(data['Z'].min())
            z_max = float(data['Z'].max())
        else:
            z_min = 0
            z_max = 1
    else:
        z_min = settings.get('z_min', 0)
        z_max = settings.get('z_max', 1)

    colorscale_name = settings.get('colorscale', 'Turbo')
    reverse_colorscale = settings.get('reverse_colorscale', False)

    # Build colorscale - always start with the base colorscale name
    colorscale = colorscale_name

    # Create discrete colorscale if enabled
    if settings.get('discrete_colors', True):
        n_steps = settings.get('discrete_steps', 10)
        try:
            # Sample the colorscale at n_steps points
            sample_points = [(i + 0.5) / n_steps for i in range(n_steps)]
            if reverse_colorscale:
                sample_points = sample_points[::-1]
            sampled_colors = pc.sample_colorscale(colorscale_name, sample_points)

            discrete_colorscale = []
            for i in range(n_steps):
                low = i / n_steps
                high = (i + 1) / n_steps
                discrete_colorscale.append([low, sampled_colors[i]])
                discrete_colorscale.append([high, sampled_colors[i]])
            colorscale = discrete_colorscale
        except Exception:
            # Fallback to simple colorscale name
            colorscale = colorscale_name
    elif reverse_colorscale:
        # For continuous colorscale with reverse
        try:
            base_colors = pc.get_colorscale(colorscale_name)
            colorscale = [[1 - pos, color] for pos, color in base_colors][::-1]
        except Exception:
            # Fallback - just use the name, Plotly might handle it
            colorscale = colorscale_name

    # Build colorbar dict with all settings
    cb_title_side = settings.get('colorbar_title_side', 'right')
    colorbar_dict = dict(
        title=dict(
            text=settings.get('colorbar_title', '') or labels.get('Z', 'Z'),
            font=dict(size=settings.get('axis_font_size', 24), family=font_family, color=font_color),
            side=cb_title_side,
        ),
        tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
        len=settings.get('colorbar_len', 0.6),
        thickness=settings.get('colorbar_thickness', 20),
        x=settings.get('colorbar_x', 1.02),
        y=settings.get('colorbar_y', 0.5),
        ticks='outside' if settings.get('colorbar_ticks', True) else '',
        showticklabels=settings.get('colorbar_ticks', True),
    )

    # Grid settings
    grid_color = settings.get('grid_color', '#808080')
    grid_width = settings.get('grid_line_width', 1)

    if len(data) > 0:
        valid_data = data.dropna(subset=['A', 'B', 'C'])

        if len(valid_data) > 0:
            a_vals = valid_data['A'].values.astype(float)
            b_vals = valid_data['B'].values.astype(float)
            c_vals = valid_data['C'].values.astype(float)

            total = a_vals + b_vals + c_vals
            total = np.where(total == 0, 1, total)

            a_norm = a_vals / total
            b_norm = b_vals / total
            c_norm = c_vals / total

            z_vals = valid_data['Z'].values if 'Z' in valid_data.columns else None
            name_vals = valid_data['Name'].values if 'Name' in valid_data.columns else None

            # Apply log scale if enabled
            if settings.get('log_scale', False) and z_vals is not None:
                z_vals_plot = np.log10(np.clip(z_vals, 1e-10, None))
                z_min_plot = np.log10(max(z_min, 1e-10)) if z_min > 0 else -10
                z_max_plot = np.log10(max(z_max, 1e-10)) if z_max > 0 else 0
            else:
                z_vals_plot = z_vals
                z_min_plot = z_min
                z_max_plot = z_max

            heatmap_enabled = settings.get('heatmap_enabled', False)
            heatmap_marker_mode = settings.get('heatmap_marker_mode', 'fill')

            if heatmap_enabled and z_vals is not None and len(z_vals) > 3 and not np.all(np.isnan(z_vals)):
                resolution = settings.get('heatmap_resolution', 100)
                method = settings.get('heatmap_method', 'linear')

                grid_a, grid_b, grid_c = [], [], []
                for i in range(resolution + 1):
                    for j in range(resolution + 1 - i):
                        a = i / resolution
                        b = j / resolution
                        c = 1 - a - b
                        if c >= 0:
                            grid_a.append(a)
                            grid_b.append(b)
                            grid_c.append(c)

                grid_a = np.array(grid_a)
                grid_b = np.array(grid_b)
                grid_c = np.array(grid_c)

                try:
                    x_data = 0.5 * (2 * b_norm + c_norm)
                    y_data = (np.sqrt(3) / 2) * c_norm
                    x_grid = 0.5 * (2 * grid_b + grid_c)
                    y_grid = (np.sqrt(3) / 2) * grid_c

                    valid_z_mask = ~np.isnan(z_vals_plot)
                    if np.sum(valid_z_mask) >= 3:
                        z_interp = griddata(
                            (x_data[valid_z_mask], y_data[valid_z_mask]),
                            z_vals_plot[valid_z_mask],
                            (x_grid, y_grid),
                            method=method
                        )

                        valid_mask = ~np.isnan(z_interp)

                        # Heatmap layer - use heatmap-specific marker size and opacity
                        fig.add_trace(go.Scatterternary(
                            a=grid_a[valid_mask], b=grid_b[valid_mask], c=grid_c[valid_mask],
                            mode='markers',
                            marker=dict(
                                size=settings.get('heatmap_marker_size', 8),
                                color=z_interp[valid_mask],
                                colorscale=colorscale,
                                cmin=z_min_plot, cmax=z_max_plot,
                                showscale=settings.get('show_colorbar', True),
                                colorbar=colorbar_dict,
                                opacity=settings.get('heatmap_opacity', 1.0),
                                symbol=settings.get('heatmap_symbol', 'hexagon2'),
                            ),
                            hoverinfo='text',
                            text=[f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}"
                                  for a, b, c, z in zip(grid_a[valid_mask], grid_b[valid_mask], grid_c[valid_mask], z_interp[valid_mask])],
                            showlegend=False
                        ))

                        # Data points layer based on mode
                        if heatmap_marker_mode != 'hide':
                            if heatmap_marker_mode == 'fill':
                                # Color-filled markers
                                marker_dict = dict(
                                    size=settings.get('marker_size', 8) + 2,
                                    color=z_vals_plot,
                                    colorscale=colorscale,
                                    cmin=z_min_plot, cmax=z_max_plot,
                                    showscale=False,
                                    line=dict(color=settings.get('marker_line_color', '#000000'), width=settings.get('marker_line_width', 1)),
                                )
                            else:  # white
                                marker_dict = dict(
                                    size=settings.get('marker_size', 8) + 2,
                                    color='white',
                                    line=dict(color=settings.get('marker_line_color', '#000000'), width=settings.get('marker_line_width', 1)),
                                )

                            fig.add_trace(go.Scatterternary(
                                a=a_norm, b=b_norm, c=c_norm,
                                mode='markers',
                                marker=marker_dict,
                                hoverinfo='text',
                                text=[f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}" if not np.isnan(z) else
                                      f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                                      for a, b, c, z in zip(a_norm, b_norm, c_norm, z_vals)],
                                showlegend=False
                            ))
                    else:
                        heatmap_enabled = False
                except Exception:
                    heatmap_enabled = False

            if not heatmap_enabled:
                # Single color mode
                use_single_color = settings.get('use_single_color', False)
                single_color = settings.get('single_color', '#1f77b4')

                marker_dict = dict(
                    size=settings.get('marker_size', 8),
                    symbol=settings.get('marker_symbol', 'circle'),
                    line=dict(color=settings.get('marker_line_color', '#000000'), width=settings.get('marker_line_width', 1)),
                    opacity=settings.get('marker_opacity', 0.8),
                )

                # Helper function to build hover text with optional Name
                def build_hover_text(a, b, c, z=None, name=None):
                    text = f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                    if z is not None and not np.isnan(z):
                        text += f"<br>{labels['Z']}: {z:.4g}"
                    if name is not None and pd.notna(name) and str(name).strip():
                        text += f"<br>Name: {name}"
                    return text

                if use_single_color:
                    marker_dict['color'] = single_color
                    if name_vals is not None:
                        hover_text = [build_hover_text(a, b, c, name=n) for a, b, c, n in zip(a_norm, b_norm, c_norm, name_vals)]
                    else:
                        hover_text = [build_hover_text(a, b, c) for a, b, c in zip(a_norm, b_norm, c_norm)]
                elif z_vals_plot is not None and len(z_vals_plot) > 0 and not np.all(np.isnan(z_vals_plot)):
                    marker_dict['color'] = z_vals_plot
                    marker_dict['colorscale'] = colorscale
                    marker_dict['cmin'] = z_min_plot
                    marker_dict['cmax'] = z_max_plot
                    marker_dict['showscale'] = settings.get('show_colorbar', True)
                    marker_dict['colorbar'] = colorbar_dict
                    if name_vals is not None:
                        hover_text = [build_hover_text(a, b, c, z, n) for a, b, c, z, n in zip(a_norm, b_norm, c_norm, z_vals, name_vals)]
                    else:
                        hover_text = [build_hover_text(a, b, c, z) for a, b, c, z in zip(a_norm, b_norm, c_norm, z_vals)]
                else:
                    marker_dict['color'] = '#1f77b4'
                    if name_vals is not None:
                        hover_text = [build_hover_text(a, b, c, name=n) for a, b, c, n in zip(a_norm, b_norm, c_norm, name_vals)]
                    else:
                        hover_text = [build_hover_text(a, b, c) for a, b, c in zip(a_norm, b_norm, c_norm)]

                fig.add_trace(go.Scatterternary(
                    a=a_norm, b=b_norm, c=c_norm,
                    mode='markers',
                    marker=marker_dict,
                    hoverinfo='text',
                    text=hover_text,
                    showlegend=False
                ))

    tick_step = settings.get('tick_step', 0.1)

    fig.update_layout(
        ternary=dict(
            sum=1,
            aaxis=dict(
                title=dict(text=a_label, font=dict(size=settings.get('axis_font_size', 24), family=font_family, color=font_color)),
                tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor=grid_color if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=grid_width,
                tick0=0, dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', False),
                ticks='outside' if settings.get('show_tick_labels', False) else '',
            ),
            baxis=dict(
                title=dict(text=b_label, font=dict(size=settings.get('axis_font_size', 24), family=font_family, color=font_color)),
                tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor=grid_color if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=grid_width,
                tick0=0, dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', False),
                ticks='outside' if settings.get('show_tick_labels', False) else '',
            ),
            caxis=dict(
                title=dict(text=c_label, font=dict(size=settings.get('axis_font_size', 24), family=font_family, color=font_color)),
                tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor=grid_color if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=grid_width,
                tick0=0, dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', False),
                ticks='outside' if settings.get('show_tick_labels', False) else '',
            ),
            bgcolor=settings.get('bgcolor', 'white'),
        ),
        font=dict(family=font_family, color=font_color),
        width=settings.get('fig_width', 700),
        height=settings.get('fig_height', 600),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(
            l=settings.get('margin_left', 60),
            r=settings.get('margin_right', 60),
            t=settings.get('margin_top', 40),
            b=settings.get('margin_bottom', 60)
        ),
    )

    return fig


def render_data_loader():
    """Render data loader section."""
    st.markdown("#### Data Loader")

    c1, c2 = st.columns([3, 1])
    with c1:
        uploaded_file = st.file_uploader(
            "Upload CSV/TXT",
            type=['csv', 'txt', 'tsv'],
            help="Upload a file with A, B, C (and optionally Z) columns.",
            key='file_uploader',
            label_visibility="collapsed"
        )
    with c2:
        if st.button("Load Sample", key='load_sample_btn', help="Load sample data from test_data/comp_sigma.csv"):
            # Load from test_data/comp_sigma.csv
            import os
            sample_path = os.path.join(os.path.dirname(__file__), 'test_data', 'comp_sigma.csv')
            try:
                sample_df = pd.read_csv(sample_path)
                new_data = pd.DataFrame()
                new_data['A'] = sample_df['Li2S']
                new_data['B'] = sample_df['P2S5']
                new_data['C'] = sample_df['LiI']
                new_data['Z'] = sample_df['sigma']
                new_data['Name'] = ''
                st.session_state.data = new_data
                st.session_state.labels = {'A': 'Li2S', 'B': 'P2S5', 'C': 'LiI', 'Z': 'σ / S cm⁻¹'}
                st.session_state.label_a = 'Li2S'
                st.session_state.label_b = 'P2S5'
                st.session_state.label_c = 'LiI'
                st.session_state.label_z = 'σ / S cm⁻¹'
                # Enable heatmap since Z values exist
                st.session_state.ps_heatmap_enabled = True
                st.session_state.data_version += 1
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {e}")

    if uploaded_file is not None:
        if st.session_state.uploaded_file_name != uploaded_file.name:
            df, headers, error = parse_uploaded_file(uploaded_file)
            if error:
                st.error(f"Error: {error}")
            else:
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.loaded_file_data = {'df': df, 'headers': headers}

        if st.session_state.loaded_file_data is not None:
            df = st.session_state.loaded_file_data['df']
            headers = st.session_state.loaded_file_data['headers']
            available_cols = list(df.columns)
            n_cols = len(available_cols)

            # Auto-detect column mapping based on number of columns
            # 3 cols: ABC, 4 cols: ABCZ, 5+ cols: ABCZName
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                a_col = st.selectbox("A", options=available_cols, index=0, key='a_col_select')
            with c2:
                b_col = st.selectbox("B", options=available_cols, index=min(1, n_cols-1), key='b_col_select')
            with c3:
                c_col = st.selectbox("C", options=available_cols, index=min(2, n_cols-1), key='c_col_select')
            with c4:
                z_options = ['(None)'] + available_cols
                # Default Z index: column 4 if exists, else (None)
                z_default_idx = min(4, len(z_options)-1) if n_cols >= 4 else 0
                z_col = st.selectbox("Z", options=z_options, index=z_default_idx, key='z_col_select')
            with c5:
                name_options = ['(None)'] + available_cols
                # Default Name index: column 5 if exists, else (None)
                name_default_idx = min(5, len(name_options)-1) if n_cols >= 5 else 0
                name_col = st.selectbox("Name", options=name_options, index=name_default_idx, key='name_col_select')

            use_headers = headers and st.checkbox("Use headers as labels", value=True, key='use_headers')

            if st.button("Load Data", type="primary", key='load_data_btn'):
                new_data = pd.DataFrame()
                new_data['A'] = pd.to_numeric(df[a_col], errors='coerce')
                new_data['B'] = pd.to_numeric(df[b_col], errors='coerce')
                new_data['C'] = pd.to_numeric(df[c_col], errors='coerce')
                new_data['Z'] = pd.to_numeric(df[z_col], errors='coerce') if z_col != '(None)' else np.nan
                new_data['Name'] = df[name_col].astype(str) if name_col != '(None)' else ''

                if use_headers:
                    if not str(a_col).isdigit():
                        st.session_state.labels['A'] = str(a_col)
                        st.session_state.label_a = str(a_col)
                    if not str(b_col).isdigit():
                        st.session_state.labels['B'] = str(b_col)
                        st.session_state.label_b = str(b_col)
                    if not str(c_col).isdigit():
                        st.session_state.labels['C'] = str(c_col)
                        st.session_state.label_c = str(c_col)
                    if z_col != '(None)' and not str(z_col).isdigit():
                        st.session_state.labels['Z'] = str(z_col)
                        st.session_state.label_z = str(z_col)
                        st.session_state.z_label_preset = 'custom'

                st.session_state.data = new_data
                # Enable heatmap if Z values exist
                if z_col != '(None)' and new_data['Z'].notna().any():
                    st.session_state.ps_heatmap_enabled = True
                st.session_state.data_version += 1
                st.rerun()


def render_data_labels():
    """Render data labels section."""
    st.markdown("#### Data Labels")

    z_presets = {
        'custom': 'Custom',
        'sigma': '<i>σ</i><sub>298K</sub> / S cm<sup>–1</sup>',
        'ea': '<i>E</i><sub>a</sub> / kJ mol<sup>–1</sup>',
        'capacity': 'Capacity / mAh g<sup>–1</sup>'
    }

    if 'label_a' not in st.session_state:
        st.session_state.label_a = st.session_state.labels.get('A', 'A')
    if 'label_b' not in st.session_state:
        st.session_state.label_b = st.session_state.labels.get('B', 'B')
    if 'label_c' not in st.session_state:
        st.session_state.label_c = st.session_state.labels.get('C', 'C')
    if 'label_z' not in st.session_state:
        st.session_state.label_z = st.session_state.labels.get('Z', 'Z')

    is_custom = st.session_state.z_label_preset == 'custom'

    # Always use 4 columns for A, B, C, Z preset
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        new_a = st.text_input("A", key='label_a')
        st.session_state.labels['A'] = new_a

    with c2:
        new_b = st.text_input("B", key='label_b')
        st.session_state.labels['B'] = new_b

    with c3:
        new_c = st.text_input("C", key='label_c')
        st.session_state.labels['C'] = new_c

    with c4:
        z_preset = st.selectbox("Z preset", options=list(z_presets.keys()), format_func=lambda x: z_presets[x],
                                 index=list(z_presets.keys()).index(st.session_state.z_label_preset), key='z_preset')

        if z_preset != st.session_state.z_label_preset:
            st.session_state.z_label_preset = z_preset
            if z_preset != 'custom':
                st.session_state.labels['Z'] = z_presets[z_preset]
                st.session_state.label_z = z_presets[z_preset]
            st.rerun()

    # Show Z input on new line when Custom is selected
    if is_custom:
        new_z = st.text_input("Z label", key='label_z')
        st.session_state.labels['Z'] = new_z


def render_data_table():
    """Render data table section."""
    st.markdown("#### Data Table")

    # Use data_version to force refresh when data changes externally
    data_version = st.session_state.get('data_version', 0)

    display_df = st.session_state.data.copy()
    for col in ['A', 'B', 'C', 'Z']:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

    if PYMATGEN_AVAILABLE and len(display_df) > 0:
        sum_formulas = []
        for idx, row in display_df.iterrows():
            try:
                a_val = float(row['A']) if pd.notna(row['A']) else 0
                b_val = float(row['B']) if pd.notna(row['B']) else 0
                c_val = float(row['C']) if pd.notna(row['C']) else 0

                if a_val > 0 or b_val > 0 or c_val > 0:
                    comp_str = ""
                    if a_val > 0: comp_str += f"({st.session_state.labels['A']}){a_val}"
                    if b_val > 0: comp_str += f"({st.session_state.labels['B']}){b_val}"
                    if c_val > 0: comp_str += f"({st.session_state.labels['C']}){c_val}"
                    sum_formulas.append(Composition(comp_str).reduced_formula)
                else:
                    sum_formulas.append("")
            except Exception:
                sum_formulas.append("")
        display_df['SUM'] = sum_formulas
    else:
        display_df['SUM'] = ""

    # Use unique key with data_version to force refresh
    edited_df = st.data_editor(
        display_df,
        column_config={
            "A": st.column_config.NumberColumn(st.session_state.labels.get('A', 'A'), min_value=0, format="%.4f"),
            "B": st.column_config.NumberColumn(st.session_state.labels.get('B', 'B'), min_value=0, format="%.4f"),
            "C": st.column_config.NumberColumn(st.session_state.labels.get('C', 'C'), min_value=0, format="%.4f"),
            "Z": st.column_config.NumberColumn(st.session_state.labels.get('Z', 'Z'), format="%.4g"),
            "Name": st.column_config.TextColumn("Name"),
            "SUM": st.column_config.TextColumn("Formula", disabled=True),
        },
        num_rows="dynamic",
        key=f'data_editor_{data_version}',
        height=300
    )

    if 'SUM' in edited_df.columns:
        edited_df = edited_df.drop(columns=['SUM'])

    # Check if data actually changed to avoid infinite loop
    if not edited_df.equals(st.session_state.data):
        st.session_state.data = edited_df

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Clear", key='clear_btn', help="Clear all data from the table"):
            st.session_state.data = pd.DataFrame({
                'A': pd.Series(dtype='float64'), 'B': pd.Series(dtype='float64'),
                'C': pd.Series(dtype='float64'), 'Z': pd.Series(dtype='float64'),
                'Name': pd.Series(dtype='str')
            })
            st.session_state.data_version += 1
            st.rerun()

    # Composition Factorizer
    if PYMATGEN_AVAILABLE and SYMPY_AVAILABLE:
        st.markdown("##### Composition Factorizer")
        st.caption(f"Input formula to factorize into {st.session_state.labels['A']}, {st.session_state.labels['B']}, {st.session_state.labels['C']}")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            formula_input = st.text_input("Formula", key='factorize_input', placeholder="e.g., Li3PS4", help="Chemical formula to factorize")
        with c2:
            z_value_str = st.text_input("Z value", key='factorize_z', placeholder="(optional)", help="Optional value for Z column")
        with c3:
            name_value = st.text_input("Name", key='factorize_name', placeholder="(optional)", help="Optional sample name")
        with c4:
            if st.button("Add", key='factorize_btn', type="primary", help="Add factorized composition to data"):
                if formula_input:
                    basis = [st.session_state.labels['A'], st.session_state.labels['B'], st.session_state.labels['C']]
                    result = factorize_composition(formula_input, basis)
                    if result:
                        total = sum(result.values())
                        if total > 0:
                            # Parse Z value (optional)
                            try:
                                z_value = float(z_value_str) if z_value_str.strip() else np.nan
                            except ValueError:
                                z_value = np.nan
                            new_row = pd.DataFrame({
                                'A': [result.get(basis[0], 0) / total],
                                'B': [result.get(basis[1], 0) / total],
                                'C': [result.get(basis[2], 0) / total],
                                'Z': [z_value],
                                'Name': [name_value]
                            })
                            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                            st.session_state.data_version += 1
                            st.rerun()
                        else:
                            st.error("Factorization result is zero")
                    else:
                        st.error(f"Cannot factorize '{formula_input}' into {basis}")


def render_plot_settings():
    """Render plot settings panel - all settings visible."""
    st.markdown("#### Plot Settings")

    # Widget defaults - updated per requirements
    widget_defaults = {
        'ps_fig_width': 700,
        'ps_fig_height': 600,
        'ps_colorscale': 'Turbo',
        'ps_reverse_colorscale': False,
        'ps_auto_z_range': True,
        'ps_log_scale': False,
        'ps_marker_size': 8,
        'ps_marker_symbol': 'circle',
        'ps_marker_line_width': 1,
        'ps_marker_opacity': 0.8,
        'ps_use_single_color': False,
        'ps_single_color': '#1f77b4',
        'ps_axis_line_width': 2,
        'ps_show_grid': True,
        'ps_grid_color': '#808080',
        'ps_grid_line_width': 1,
        'ps_show_tick_labels': False,  # Default OFF
        'ps_auto_subscript': True,
        'ps_axis_font_size': 24,  # Default 24
        'ps_tick_font_size': 14,
        'ps_tick_step': 0.1,
        'ps_show_colorbar': True,
        'ps_colorbar_len': 0.6,
        'ps_colorbar_thickness': 20,
        'ps_colorbar_x': 1.02,
        'ps_colorbar_y': 0.5,
        'ps_colorbar_ticks': True,
        'ps_colorbar_title_side': 'right',
        'ps_discrete_colors': True,
        'ps_discrete_steps': 10,
        'ps_heatmap_enabled': False,
        'ps_heatmap_resolution': 100,
        'ps_heatmap_method': 'linear',
        'ps_heatmap_marker_mode': 'fill',
        'ps_heatmap_marker_size': 8,
        'ps_heatmap_symbol': 'hexagon2',
        'ps_heatmap_opacity': 1.0,
        'ps_margin_top': 40,
        'ps_margin_bottom': 60,
        'ps_margin_left': 60,  # Default 60
        'ps_margin_right': 60,  # Default 60
    }

    for key, default in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ===== General Settings =====
    st.markdown("##### General")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Width", 400, 1500, step=50, key='ps_fig_width', help="Figure width in pixels")
    with c2:
        st.number_input("Height", 400, 1500, step=50, key='ps_fig_height', help="Figure height in pixels")
    with c3:
        st.number_input("Axis font", 8, 40, key='ps_axis_font_size', help="Font size for axis labels (A, B, C)")
    with c4:
        st.number_input("Tick font", 8, 30, key='ps_tick_font_size', help="Font size for tick labels")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.slider("Axis width", 1, 5, value=st.session_state.get('ps_axis_line_width', 2), key='ps_axis_line_width', help="Line width of triangle axes")
    with c2:
        st.checkbox("Grid", key='ps_show_grid', help="Show grid lines inside triangle")
    with c3:
        st.checkbox("Ticks", key='ps_show_tick_labels', help="Show tick labels on axes")
    with c4:
        st.checkbox("Subscript", key='ps_auto_subscript', help="Auto-convert numbers to subscript in labels")

    if st.session_state.ps_show_grid:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.color_picker("Grid color", key='ps_grid_color', help="Color of grid lines")
        with c2:
            st.slider("Grid width", 1, 5, value=st.session_state.get('ps_grid_line_width', 1), key='ps_grid_line_width', help="Line width of grid")
        with c3:
            tick_options = [0.05, 0.1, 0.2, 0.25, 0.5]
            st.selectbox("Tick step", tick_options, key='ps_tick_step', help="Spacing between grid lines")

    # Margins
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Margin Top", 0, 200, step=10, key='ps_margin_top', help="Top margin in pixels")
    with c2:
        st.number_input("Margin Bottom", 0, 200, step=10, key='ps_margin_bottom', help="Bottom margin in pixels")
    with c3:
        st.number_input("Margin Left", 0, 200, step=10, key='ps_margin_left', help="Left margin in pixels")
    with c4:
        st.number_input("Margin Right", 0, 200, step=10, key='ps_margin_right', help="Right margin in pixels")

    st.markdown("---")

    # ===== Marker Settings =====
    st.markdown("##### Markers")

    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'hexagon', 'star']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.slider("Marker size", 2, 30, value=st.session_state.get('ps_marker_size', 8), key='ps_marker_size', help="Size of data point markers")
    with c2:
        st.selectbox("Symbol", symbols, key='ps_marker_symbol', help="Shape of data point markers")
    with c3:
        st.slider("Edge width", 0, 5, value=st.session_state.get('ps_marker_line_width', 1), key='ps_marker_line_width', help="Width of marker edge/outline")
    with c4:
        st.slider("Opacity", 0.0, 1.0, value=st.session_state.get('ps_marker_opacity', 0.8), step=0.1, key='ps_marker_opacity', help="Transparency of markers (1.0 = opaque)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.checkbox("Single color", key='ps_use_single_color', help="Use a single color for all markers instead of Z-value colorscale")
    if st.session_state.ps_use_single_color:
        with c2:
            st.color_picker("Color", key='ps_single_color')

    st.markdown("---")

    # ===== Colorbar Settings =====
    st.markdown("##### Colorbar")

    colorscales = ['Turbo', 'Jet', 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Hot', 'RdBu', 'RdYlBu', 'Blues', 'Reds']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.selectbox("Colorscale", colorscales, key='ps_colorscale', help="Color palette for Z-value mapping")
    with c2:
        st.checkbox("Reverse", key='ps_reverse_colorscale', help="Reverse the colorscale direction")
    with c3:
        st.checkbox("Log scale", key='ps_log_scale', help=LOG_SCALE_HELP)
    with c4:
        st.checkbox("Colorbar", key='ps_show_colorbar', help="Show colorbar legend for Z-values")

    # Z Range
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.checkbox("Auto Z", key='ps_auto_z_range', help="Automatically calculate Z range from data")
    if not st.session_state.ps_auto_z_range:
        data = st.session_state.data
        if 'Z' in data.columns and len(data) > 0 and data['Z'].notna().any():
            data_z_min = float(data['Z'].min())
            data_z_max = float(data['Z'].max())
        else:
            data_z_min = 0.0
            data_z_max = 1.0

        if 'ps_z_min' not in st.session_state:
            st.session_state.ps_z_min = data_z_min
        if 'ps_z_max' not in st.session_state:
            st.session_state.ps_z_max = data_z_max

        with c2:
            st.number_input("Z min", key='ps_z_min')
        with c3:
            st.number_input("Z max", key='ps_z_max')

    if st.session_state.ps_show_colorbar:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.slider("CB length", 0.2, 1.0, value=st.session_state.get('ps_colorbar_len', 0.6), step=0.1, key='ps_colorbar_len', help="Length of colorbar (fraction of plot)")
        with c2:
            st.slider("CB thickness", 10, 40, value=st.session_state.get('ps_colorbar_thickness', 20), key='ps_colorbar_thickness', help="Thickness of colorbar in pixels")
        with c3:
            st.checkbox("CB ticks", key='ps_colorbar_ticks', help="Show tick marks on colorbar")
        with c4:
            title_sides = ['right', 'top']
            st.selectbox("CB title side", title_sides, key='ps_colorbar_title_side', help="Position of colorbar title")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.number_input("CB x pos", 0.8, 1.2, step=0.02, key='ps_colorbar_x', help="Horizontal position of colorbar")
        with c2:
            st.number_input("CB y pos", 0.0, 1.0, step=0.1, key='ps_colorbar_y', help="Vertical position of colorbar")
        with c3:
            st.checkbox("Discrete colors", key='ps_discrete_colors', help=DISCRETE_COLORS_HELP)

    if st.session_state.get('ps_discrete_colors', False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.slider("Color steps", 2, 20, value=st.session_state.get('ps_discrete_steps', 10), key='ps_discrete_steps', help="Number of discrete color bands")

    st.markdown("---")

    # ===== Heatmap Settings =====
    st.markdown("##### Heatmap")

    methods = ['linear', 'cubic', 'nearest']
    marker_modes = ['fill', 'white', 'hide']  # 'fill' first as default

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.checkbox("Enable heatmap", key='ps_heatmap_enabled', help="Enable heatmap interpolation for Z-values")

    if st.session_state.ps_heatmap_enabled:
        with c2:
            st.slider("Resolution", 10, 500, value=st.session_state.get('ps_heatmap_resolution', 100), key='ps_heatmap_resolution', help="Number of grid points for interpolation (higher = smoother)")
        with c3:
            method_help = "\n".join([f"**{m}**: {INTERPOLATION_METHODS[m]}" for m in methods])
            current_method = st.session_state.get('ps_heatmap_method', 'linear')
            method_index = methods.index(current_method) if current_method in methods else 0
            st.selectbox("Method", methods, index=method_index, key='ps_heatmap_method', help=method_help)
        with c4:
            mode_help = "\n".join([f"**{m}**: {HEATMAP_MARKER_MODES[m]}" for m in marker_modes])
            current_mode = st.session_state.get('ps_heatmap_marker_mode', 'fill')
            mode_index = marker_modes.index(current_mode) if current_mode in marker_modes else 0
            st.selectbox("Markers", marker_modes, index=mode_index, key='ps_heatmap_marker_mode',
                        format_func=lambda x: {'white': 'White fill', 'fill': 'Color fill', 'hide': 'Hide'}[x],
                        help=mode_help)

        # Heatmap-specific marker settings
        hm_symbols = ['hexagon2', 'circle', 'square', 'diamond', 'triangle-up', 'triangle-down']
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.slider("HM marker size", 2, 30, value=st.session_state.get('ps_heatmap_marker_size', 8), key='ps_heatmap_marker_size', help="Marker size for heatmap interpolation layer")
        with c2:
            current_hm_symbol = st.session_state.get('ps_heatmap_symbol', 'hexagon2')
            hm_symbol_index = hm_symbols.index(current_hm_symbol) if current_hm_symbol in hm_symbols else 0
            st.selectbox("HM symbol", hm_symbols, index=hm_symbol_index, key='ps_heatmap_symbol',
                        format_func=lambda x: {'hexagon2': 'Hexagon', 'circle': 'Circle', 'square': 'Square',
                                               'diamond': 'Diamond', 'triangle-up': 'Triangle Up', 'triangle-down': 'Triangle Down'}[x],
                        help="Marker symbol for heatmap fill")
        with c3:
            st.slider("HM opacity", 0.1, 1.0, value=st.session_state.get('ps_heatmap_opacity', 1.0), step=0.1, key='ps_heatmap_opacity', help="Opacity for heatmap interpolation layer")


def main():
    """Main application function."""
    initialize_session_state()

    st.markdown("### Ternary Plot Visualizer")

    col_loader, col_labels = st.columns([1, 1])
    with col_loader:
        render_data_loader()
    with col_labels:
        render_data_labels()

    st.markdown("---")

    col_plot, col_table = st.columns([6, 4])

    with col_plot:
        if len(st.session_state.data) > 0:
            settings = {
                'fig_width': st.session_state.get('ps_fig_width', 700),
                'fig_height': st.session_state.get('ps_fig_height', 600),
                'colorscale': st.session_state.get('ps_colorscale', 'Turbo'),
                'reverse_colorscale': st.session_state.get('ps_reverse_colorscale', False),
                'auto_z_range': st.session_state.get('ps_auto_z_range', True),
                'z_min': st.session_state.get('ps_z_min'),
                'z_max': st.session_state.get('ps_z_max'),
                'log_scale': st.session_state.get('ps_log_scale', False),
                'marker_size': st.session_state.get('ps_marker_size', 8),
                'marker_symbol': st.session_state.get('ps_marker_symbol', 'circle'),
                'marker_line_width': st.session_state.get('ps_marker_line_width', 1),
                'marker_line_color': '#000000',
                'marker_opacity': st.session_state.get('ps_marker_opacity', 0.8),
                'use_single_color': st.session_state.get('ps_use_single_color', False),
                'single_color': st.session_state.get('ps_single_color', '#1f77b4'),
                'axis_line_width': st.session_state.get('ps_axis_line_width', 2),
                'show_grid': st.session_state.get('ps_show_grid', True),
                'grid_color': st.session_state.get('ps_grid_color', '#808080'),
                'grid_line_width': st.session_state.get('ps_grid_line_width', 1),
                'show_tick_labels': st.session_state.get('ps_show_tick_labels', False),
                'auto_subscript': st.session_state.get('ps_auto_subscript', True),
                'axis_font_size': st.session_state.get('ps_axis_font_size', 24),
                'tick_font_size': st.session_state.get('ps_tick_font_size', 14),
                'tick_step': st.session_state.get('ps_tick_step', 0.1),
                'show_colorbar': st.session_state.get('ps_show_colorbar', True),
                'colorbar_len': st.session_state.get('ps_colorbar_len', 0.6),
                'colorbar_thickness': st.session_state.get('ps_colorbar_thickness', 20),
                'colorbar_x': st.session_state.get('ps_colorbar_x', 1.02),
                'colorbar_y': st.session_state.get('ps_colorbar_y', 0.5),
                'colorbar_ticks': st.session_state.get('ps_colorbar_ticks', True),
                'colorbar_title_side': st.session_state.get('ps_colorbar_title_side', 'right'),
                'discrete_colors': st.session_state.get('ps_discrete_colors', True),
                'discrete_steps': st.session_state.get('ps_discrete_steps', 10),
                'heatmap_enabled': st.session_state.get('ps_heatmap_enabled', False),
                'heatmap_resolution': st.session_state.get('ps_heatmap_resolution', 100),
                'heatmap_method': st.session_state.get('ps_heatmap_method', 'linear'),
                'heatmap_marker_mode': st.session_state.get('ps_heatmap_marker_mode', 'fill'),
                'heatmap_marker_size': st.session_state.get('ps_heatmap_marker_size', 8),
                'heatmap_symbol': st.session_state.get('ps_heatmap_symbol', 'hexagon2'),
                'heatmap_opacity': st.session_state.get('ps_heatmap_opacity', 1.0),
                'margin_top': st.session_state.get('ps_margin_top', 40),
                'margin_bottom': st.session_state.get('ps_margin_bottom', 60),
                'margin_left': st.session_state.get('ps_margin_left', 60),
                'margin_right': st.session_state.get('ps_margin_right', 60),
                'bgcolor': 'white',
                'colorbar_title': '',
            }
            fig = create_ternary_plot(st.session_state.data, st.session_state.labels, settings)

            # Config for high-quality PNG export via toolbar camera button
            plotly_config = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'ternary_plot',
                    'width': settings.get('fig_width', 700),
                    'height': settings.get('fig_height', 600),
                    'scale': 2
                },
                'displayModeBar': True,
                'displaylogo': False,
            }
            st.plotly_chart(fig, key='ternary_plot', config=plotly_config)

            # Export buttons: HTML and CSV only
            c1, c2 = st.columns(2)
            with c1:
                html_data = fig.to_html(include_plotlyjs='cdn')
                st.download_button("HTML", html_data, "ternary.html", "text/html", key='dl_html')
            with c2:
                st.download_button("CSV", st.session_state.data.to_csv(index=False), "data.csv", "text/csv", key='dl_csv')
        else:
            st.info("No data. Load a file or add data manually.")

    with col_table:
        render_data_table()

    st.markdown("---")

    render_plot_settings()


if __name__ == "__main__":
    main()
