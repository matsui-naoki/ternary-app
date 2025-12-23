"""
Ternary Plot Visualizer - Streamlit Web Application
Three-component composition diagram visualization tool for academic papers and presentations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
            'Z': pd.Series(dtype='float64')
        })

    if 'labels' not in st.session_state:
        st.session_state.labels = {'A': 'A', 'B': 'B', 'C': 'C', 'Z': 'Z'}

    if 'z_label_preset' not in st.session_state:
        st.session_state.z_label_preset = 'custom'

    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {
            'fig_width': 700,
            'fig_height': 600,
            'colorscale': 'Viridis',
            'reverse_colorscale': False,
            'z_min': None,
            'z_max': None,
            'auto_z_range': True,
            'interpolate': False,
            'interpolation_resolution': 50,
            'interpolation_method': 'linear',
            'marker_size': 12,
            'marker_symbol': 'circle',
            'marker_line_color': '#000000',
            'marker_line_width': 1,
            'marker_opacity': 0.8,
            'axis_line_width': 2,
            'grid_line_width': 1,
            'show_grid': True,
            'tick_step': 0.1,
            'show_tick_labels': True,
            'title_font_size': 24,
            'axis_font_size': 20,
            'tick_font_size': 14,
            'auto_subscript': True,
            'bgcolor': 'white',
            'show_colorbar': True,
            'colorbar_title': '',
            'margin_top': 40,
            'margin_bottom': 60,
            'margin_left': 40,
            'margin_right': 40,
        }

    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

    if 'loaded_file_data' not in st.session_state:
        st.session_state.loaded_file_data = None


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

    colorscale = settings.get('colorscale', 'Viridis')
    if settings.get('reverse_colorscale', False):
        colorscale = colorscale + '_r'

    # Create discrete colorscale if enabled
    if settings.get('discrete_colors', False):
        import plotly.colors as pc
        n_steps = settings.get('discrete_steps', 10)
        try:
            base_colors = pc.get_colorscale(colorscale)
            discrete_colorscale = []
            for i in range(n_steps):
                # Create discrete bands
                low = i / n_steps
                high = (i + 1) / n_steps
                # Get color at midpoint
                mid = (low + high) / 2
                color = pc.sample_colorscale(colorscale, [mid])[0]
                discrete_colorscale.append([low, color])
                discrete_colorscale.append([high, color])
            colorscale = discrete_colorscale
        except Exception:
            pass  # Fall back to continuous colorscale

    # Build colorbar dict with all settings
    cb_title_side = settings.get('colorbar_title_side', 'right')
    colorbar_dict = dict(
        title=dict(
            text=settings.get('colorbar_title', '') or labels.get('Z', 'Z'),
            font=dict(size=settings.get('axis_font_size', 20), family=font_family, color=font_color),
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
    # For 'top' title side, rotate title 90 degrees
    if cb_title_side == 'top':
        colorbar_dict['title']['side'] = 'top'

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

            if settings.get('interpolate', False) and z_vals is not None and len(z_vals) > 3 and not np.all(np.isnan(z_vals)):
                resolution = settings.get('interpolation_resolution', 50)
                method = settings.get('interpolation_method', 'linear')

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

                    valid_z_mask = ~np.isnan(z_vals)
                    if np.sum(valid_z_mask) >= 3:
                        z_interp = griddata(
                            (x_data[valid_z_mask], y_data[valid_z_mask]),
                            z_vals[valid_z_mask],
                            (x_grid, y_grid),
                            method=method
                        )

                        valid_mask = ~np.isnan(z_interp)

                        fig.add_trace(go.Scatterternary(
                            a=grid_a[valid_mask], b=grid_b[valid_mask], c=grid_c[valid_mask],
                            mode='markers',
                            marker=dict(
                                size=settings.get('marker_size', 12),
                                color=z_interp[valid_mask],
                                colorscale=colorscale,
                                cmin=z_min, cmax=z_max,
                                showscale=settings.get('show_colorbar', True),
                                colorbar=colorbar_dict,
                                opacity=settings.get('marker_opacity', 0.6),
                                symbol='hexagon2',
                            ),
                            hoverinfo='text',
                            text=[f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}"
                                  for a, b, c, z in zip(grid_a[valid_mask], grid_b[valid_mask], grid_c[valid_mask], z_interp[valid_mask])],
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatterternary(
                            a=a_norm, b=b_norm, c=c_norm,
                            mode='markers',
                            marker=dict(
                                size=settings.get('marker_size', 12) + 2,
                                color='white',
                                line=dict(color=settings.get('marker_line_color', '#000000'), width=settings.get('marker_line_width', 2)),
                            ),
                            hoverinfo='text',
                            text=[f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}" if not np.isnan(z) else
                                  f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                                  for a, b, c, z in zip(a_norm, b_norm, c_norm, z_vals)],
                            showlegend=False
                        ))
                    else:
                        settings['interpolate'] = False
                except Exception:
                    settings['interpolate'] = False

            if not settings.get('interpolate', False):
                marker_dict = dict(
                    size=settings.get('marker_size', 12),
                    symbol=settings.get('marker_symbol', 'circle'),
                    line=dict(color=settings.get('marker_line_color', '#000000'), width=settings.get('marker_line_width', 1)),
                    opacity=settings.get('marker_opacity', 0.8),
                )

                if z_vals is not None and len(z_vals) > 0 and not np.all(np.isnan(z_vals)):
                    marker_dict['color'] = z_vals
                    marker_dict['colorscale'] = colorscale
                    marker_dict['cmin'] = z_min
                    marker_dict['cmax'] = z_max
                    marker_dict['showscale'] = settings.get('show_colorbar', True)
                    marker_dict['colorbar'] = colorbar_dict
                    hover_text = [f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}" if not np.isnan(z) else
                                  f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                                  for a, b, c, z in zip(a_norm, b_norm, c_norm, z_vals)]
                else:
                    marker_dict['color'] = '#1f77b4'
                    hover_text = [f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                                  for a, b, c in zip(a_norm, b_norm, c_norm)]

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
                title=dict(text=a_label, font=dict(size=settings.get('axis_font_size', 20), family=font_family, color=font_color)),
                tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor='gray' if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=settings.get('grid_line_width', 1),
                tick0=0, dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', True),
                ticks='outside' if settings.get('show_tick_labels', True) else '',
            ),
            baxis=dict(
                title=dict(text=b_label, font=dict(size=settings.get('axis_font_size', 20), family=font_family, color=font_color)),
                tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor='gray' if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=settings.get('grid_line_width', 1),
                tick0=0, dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', True),
                ticks='outside' if settings.get('show_tick_labels', True) else '',
            ),
            caxis=dict(
                title=dict(text=c_label, font=dict(size=settings.get('axis_font_size', 20), family=font_family, color=font_color)),
                tickfont=dict(size=settings.get('tick_font_size', 14), family=font_family, color=font_color),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor='gray' if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=settings.get('grid_line_width', 1),
                tick0=0, dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', True),
                ticks='outside' if settings.get('show_tick_labels', True) else '',
            ),
            bgcolor=settings.get('bgcolor', 'white'),
        ),
        font=dict(family=font_family, color=font_color),
        width=settings.get('fig_width', 700),
        height=settings.get('fig_height', 600),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(
            l=settings.get('margin_left', 40),
            r=settings.get('margin_right', 40),
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
        if st.button("Load Sample", key='load_sample_btn'):
            st.session_state.data = get_sample_data()
            st.session_state.labels = {'A': 'Li2S', 'B': 'P2S5', 'C': 'LiI', 'Z': 'σ / mS cm⁻¹'}
            st.session_state.label_a = 'Li2S'
            st.session_state.label_b = 'P2S5'
            st.session_state.label_c = 'LiI'
            st.session_state.label_z = 'σ / mS cm⁻¹'
            st.rerun()

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

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                a_col = st.selectbox("A", options=available_cols, index=0, key='a_col_select')
            with c2:
                b_col = st.selectbox("B", options=available_cols, index=min(1, len(available_cols)-1), key='b_col_select')
            with c3:
                c_col = st.selectbox("C", options=available_cols, index=min(2, len(available_cols)-1), key='c_col_select')
            with c4:
                z_options = ['(None)'] + available_cols
                z_col = st.selectbox("Z", options=z_options, index=min(4, len(z_options)-1), key='z_col_select')

            use_headers = headers and st.checkbox("Use headers as labels", value=True, key='use_headers')

            if st.button("Load Data", type="primary", key='load_data_btn'):
                new_data = pd.DataFrame()
                new_data['A'] = pd.to_numeric(df[a_col], errors='coerce')
                new_data['B'] = pd.to_numeric(df[b_col], errors='coerce')
                new_data['C'] = pd.to_numeric(df[c_col], errors='coerce')
                new_data['Z'] = pd.to_numeric(df[z_col], errors='coerce') if z_col != '(None)' else np.nan

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
                st.rerun()


def render_data_labels():
    """Render data labels section."""
    st.markdown("#### Data Labels")

    z_presets = {
        'custom': 'Custom',
        'sigma': '<i>σ</i><sub>298K</sub> / mS cm<sup>–1</sup>',
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

    if is_custom:
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    else:
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

    if is_custom:
        with c5:
            new_z = st.text_input("Z", key='label_z')
            st.session_state.labels['Z'] = new_z


def render_data_table():
    """Render data table section."""
    st.markdown("#### Data Table")

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

    edited_df = st.data_editor(
        display_df,
        column_config={
            "A": st.column_config.NumberColumn(st.session_state.labels.get('A', 'A'), min_value=0, format="%.4f"),
            "B": st.column_config.NumberColumn(st.session_state.labels.get('B', 'B'), min_value=0, format="%.4f"),
            "C": st.column_config.NumberColumn(st.session_state.labels.get('C', 'C'), min_value=0, format="%.4f"),
            "Z": st.column_config.NumberColumn(st.session_state.labels.get('Z', 'Z'), format="%.4g"),
            "SUM": st.column_config.TextColumn("Formula", disabled=True),
        },
        num_rows="dynamic",
        key='data_editor',
        height=300
    )

    if 'SUM' in edited_df.columns:
        edited_df = edited_df.drop(columns=['SUM'])
    st.session_state.data = edited_df

    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Clear", key='clear_btn'):
            st.session_state.data = pd.DataFrame({
                'A': pd.Series(dtype='float64'), 'B': pd.Series(dtype='float64'),
                'C': pd.Series(dtype='float64'), 'Z': pd.Series(dtype='float64')
            })
            st.rerun()

    # Composition Factorizer
    if PYMATGEN_AVAILABLE and SYMPY_AVAILABLE:
        st.markdown("##### Composition Factorizer")
        st.caption(f"Input formula to factorize into {st.session_state.labels['A']}, {st.session_state.labels['B']}, {st.session_state.labels['C']}")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            formula_input = st.text_input("Formula", key='factorize_input', placeholder="e.g., Li3PS4")
        with c2:
            z_value = st.number_input("Z value", key='factorize_z', value=0.0)
        with c3:
            if st.button("Add", key='factorize_btn', type="primary"):
                if formula_input:
                    basis = [st.session_state.labels['A'], st.session_state.labels['B'], st.session_state.labels['C']]
                    result = factorize_composition(formula_input, basis)
                    if result:
                        total = sum(result.values())
                        if total > 0:
                            new_row = pd.DataFrame({
                                'A': [result.get(basis[0], 0) / total],
                                'B': [result.get(basis[1], 0) / total],
                                'C': [result.get(basis[2], 0) / total],
                                'Z': [z_value]
                            })
                            st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                            st.rerun()
                        else:
                            st.error("Factorization result is zero")
                    else:
                        st.error(f"Cannot factorize '{formula_input}' into {basis}")


def render_plot_settings():
    """Render plot settings panel - all settings visible."""
    st.markdown("#### Plot Settings")

    s = st.session_state.plot_settings
    widget_defaults = {
        'ps_fig_width': s.get('fig_width', 700),
        'ps_fig_height': s.get('fig_height', 600),
        'ps_colorscale': s.get('colorscale', 'Viridis'),
        'ps_reverse_colorscale': s.get('reverse_colorscale', False),
        'ps_auto_z_range': s.get('auto_z_range', True),
        'ps_marker_size': s.get('marker_size', 12),
        'ps_marker_symbol': s.get('marker_symbol', 'circle'),
        'ps_marker_line_width': s.get('marker_line_width', 1),
        'ps_marker_opacity': s.get('marker_opacity', 0.8),
        'ps_axis_line_width': s.get('axis_line_width', 2),
        'ps_show_grid': s.get('show_grid', True),
        'ps_show_tick_labels': s.get('show_tick_labels', False),  # Default OFF
        'ps_auto_subscript': s.get('auto_subscript', True),
        'ps_axis_font_size': s.get('axis_font_size', 20),
        'ps_tick_font_size': s.get('tick_font_size', 14),
        'ps_tick_step': s.get('tick_step', 0.1),
        'ps_show_colorbar': s.get('show_colorbar', True),
        'ps_colorbar_len': s.get('colorbar_len', 0.6),
        'ps_colorbar_thickness': s.get('colorbar_thickness', 20),
        'ps_colorbar_x': s.get('colorbar_x', 1.02),
        'ps_colorbar_y': s.get('colorbar_y', 0.5),
        'ps_colorbar_ticks': s.get('colorbar_ticks', True),
        'ps_colorbar_title_side': s.get('colorbar_title_side', 'right'),
        'ps_discrete_colors': s.get('discrete_colors', False),
        'ps_discrete_steps': s.get('discrete_steps', 10),
        'ps_interpolate': s.get('interpolate', False),
        'ps_interpolation_resolution': s.get('interpolation_resolution', 50),
        'ps_interpolation_method': s.get('interpolation_method', 'linear'),
        'ps_margin_top': s.get('margin_top', 40),
        'ps_margin_bottom': s.get('margin_bottom', 60),
        'ps_margin_left': s.get('margin_left', 40),
        'ps_margin_right': s.get('margin_right', 40),
    }

    for key, default in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Size & Color
    c1, c2, c3, c4 = st.columns(4)
    colorscales = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Jet', 'Hot', 'RdBu', 'RdYlBu', 'Blues', 'Reds']
    with c1:
        st.number_input("Width", 400, 1500, step=50, key='ps_fig_width')
    with c2:
        st.number_input("Height", 400, 1500, step=50, key='ps_fig_height')
    with c3:
        st.selectbox("Colorscale", colorscales, key='ps_colorscale')
    with c4:
        st.checkbox("Reverse", key='ps_reverse_colorscale')

    # Z Range
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.checkbox("Auto Z", key='ps_auto_z_range')
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

    # Markers
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'hexagon', 'star']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.slider("Marker size", 2, 30, key='ps_marker_size')
    with c2:
        st.selectbox("Symbol", symbols, key='ps_marker_symbol')
    with c3:
        st.slider("Edge width", 0, 5, key='ps_marker_line_width')
    with c4:
        st.slider("Opacity", 0.0, 1.0, step=0.1, key='ps_marker_opacity')

    # Axis
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.slider("Axis width", 1, 5, key='ps_axis_line_width')
    with c2:
        st.checkbox("Grid", key='ps_show_grid')
    with c3:
        st.checkbox("Ticks", key='ps_show_tick_labels')
    with c4:
        st.checkbox("Subscript", key='ps_auto_subscript')

    # Fonts & Tick step
    tick_options = [0.05, 0.1, 0.2, 0.25, 0.5]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Axis font", 8, 40, key='ps_axis_font_size')
    with c2:
        st.number_input("Tick font", 8, 30, key='ps_tick_font_size')
    with c3:
        st.selectbox("Tick step", tick_options, key='ps_tick_step')

    # Colorbar settings
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.checkbox("Colorbar", key='ps_show_colorbar')
    if st.session_state.ps_show_colorbar:
        with c2:
            st.slider("CB length", 0.2, 1.0, step=0.1, key='ps_colorbar_len')
        with c3:
            st.slider("CB thickness", 10, 40, key='ps_colorbar_thickness')
        with c4:
            st.checkbox("CB ticks", key='ps_colorbar_ticks')

    if st.session_state.ps_show_colorbar:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.number_input("CB x pos", 0.8, 1.2, step=0.02, key='ps_colorbar_x')
        with c2:
            st.number_input("CB y pos", 0.0, 1.0, step=0.1, key='ps_colorbar_y')
        with c3:
            title_sides = ['right', 'top']
            st.selectbox("CB title side", title_sides, key='ps_colorbar_title_side')
        with c4:
            st.checkbox("Discrete colors", key='ps_discrete_colors')

    if st.session_state.get('ps_show_colorbar', True) and st.session_state.get('ps_discrete_colors', False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.slider("Color steps", 3, 20, key='ps_discrete_steps')

    # Interpolation
    c1, c2, c3 = st.columns(3)
    methods = ['linear', 'cubic', 'nearest']
    with c1:
        st.checkbox("Interpolate", key='ps_interpolate')
    if st.session_state.ps_interpolate:
        with c2:
            st.slider("Resolution", 10, 100, key='ps_interpolation_resolution')
        with c3:
            st.selectbox("Method", methods, key='ps_interpolation_method')

    # Margins
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.number_input("Margin Top", 0, 200, step=10, key='ps_margin_top')
    with c2:
        st.number_input("Margin Bottom", 0, 200, step=10, key='ps_margin_bottom')
    with c3:
        st.number_input("Margin Left", 0, 200, step=10, key='ps_margin_left')
    with c4:
        st.number_input("Margin Right", 0, 200, step=10, key='ps_margin_right')


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
                'colorscale': st.session_state.get('ps_colorscale', 'Viridis'),
                'reverse_colorscale': st.session_state.get('ps_reverse_colorscale', False),
                'auto_z_range': st.session_state.get('ps_auto_z_range', True),
                'z_min': st.session_state.get('ps_z_min'),
                'z_max': st.session_state.get('ps_z_max'),
                'marker_size': st.session_state.get('ps_marker_size', 12),
                'marker_symbol': st.session_state.get('ps_marker_symbol', 'circle'),
                'marker_line_width': st.session_state.get('ps_marker_line_width', 1),
                'marker_line_color': '#000000',
                'marker_opacity': st.session_state.get('ps_marker_opacity', 0.8),
                'axis_line_width': st.session_state.get('ps_axis_line_width', 2),
                'show_grid': st.session_state.get('ps_show_grid', True),
                'show_tick_labels': st.session_state.get('ps_show_tick_labels', False),
                'auto_subscript': st.session_state.get('ps_auto_subscript', True),
                'axis_font_size': st.session_state.get('ps_axis_font_size', 20),
                'tick_font_size': st.session_state.get('ps_tick_font_size', 14),
                'tick_step': st.session_state.get('ps_tick_step', 0.1),
                'show_colorbar': st.session_state.get('ps_show_colorbar', True),
                'colorbar_len': st.session_state.get('ps_colorbar_len', 0.6),
                'colorbar_thickness': st.session_state.get('ps_colorbar_thickness', 20),
                'colorbar_x': st.session_state.get('ps_colorbar_x', 1.02),
                'colorbar_y': st.session_state.get('ps_colorbar_y', 0.5),
                'colorbar_ticks': st.session_state.get('ps_colorbar_ticks', True),
                'colorbar_title_side': st.session_state.get('ps_colorbar_title_side', 'right'),
                'discrete_colors': st.session_state.get('ps_discrete_colors', False),
                'discrete_steps': st.session_state.get('ps_discrete_steps', 10),
                'interpolate': st.session_state.get('ps_interpolate', False),
                'interpolation_resolution': st.session_state.get('ps_interpolation_resolution', 50),
                'interpolation_method': st.session_state.get('ps_interpolation_method', 'linear'),
                'margin_top': st.session_state.get('ps_margin_top', 40),
                'margin_bottom': st.session_state.get('ps_margin_bottom', 60),
                'margin_left': st.session_state.get('ps_margin_left', 40),
                'margin_right': st.session_state.get('ps_margin_right', 40),
                'bgcolor': 'white',
                'colorbar_title': '',
            }
            fig = create_ternary_plot(st.session_state.data, st.session_state.labels, settings)
            st.plotly_chart(fig, key='ternary_plot')

            c1, c2, c3 = st.columns(3)
            with c1:
                try:
                    st.download_button("PNG", fig.to_image(format="png", scale=2), "ternary.png", "image/png", key='dl_png')
                except Exception:
                    st.caption("PNG: install kaleido")
            with c2:
                try:
                    st.download_button("SVG", fig.to_image(format="svg"), "ternary.svg", "image/svg+xml", key='dl_svg')
                except Exception:
                    st.caption("SVG: install kaleido")
            with c3:
                st.download_button("CSV", st.session_state.data.to_csv(index=False), "data.csv", "text/csv", key='dl_csv')
        else:
            st.info("No data. Load a file or add data manually.")

    with col_table:
        render_data_table()

    st.markdown("---")

    render_plot_settings()


if __name__ == "__main__":
    main()
