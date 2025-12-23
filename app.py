"""
Ternary Plot Analyzer - Streamlit Web Application
Three-component composition diagram visualization tool for academic papers and presentations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
from typing import Optional, Dict, List, Tuple
from scipy.interpolate import griddata

# Pymatgen for composition handling
try:
    from pymatgen.core import Composition
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

# Sympy for composition factorization
try:
    from sympy import Matrix
    from fractions import Fraction
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Ternary Plot Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def subscript_num_html(composition: str) -> str:
    """Convert numbers in composition formula to HTML subscript format.

    Example: Li2O -> Li<sub>2</sub>O
    """
    if not composition:
        return composition

    # Simple regex-based approach
    result = re.sub(r'(\d+\.?\d*)', r'<sub>\1</sub>', composition)
    # Remove <sub>1</sub> as it's redundant
    result = result.replace('<sub>1</sub>', '')

    return result


def factorize_composition(target_composition: str, basis_compositions: List[str]) -> Optional[Dict[str, float]]:
    """
    Factorize a target composition into basis compositions.

    Example:
        target = "BaLi2.1MgF6.1"
        basis_compositions = ["BaF2", "LiF", "MgF2"]
        result = {"BaF2": 1.0, "LiF": 2.1, "MgF2": 1.0}
    """
    if not PYMATGEN_AVAILABLE or not SYMPY_AVAILABLE:
        return None

    try:
        temp_basis_compositions = basis_compositions
        target_dict = Composition(target_composition).as_dict()
        basis_dicts = [Composition(bc).as_dict() for bc in basis_compositions]

        # Get unique elements
        elements = sorted(set(
            element
            for composition in basis_dicts + [target_dict]
            for element in composition
        ))

        # Convert to fractions
        target_fractions = {
            element: Fraction(target_dict.get(element, 0)).limit_denominator()
            for element in elements
        }
        basis_fractions = [
            {element: Fraction(bd.get(element, 0)).limit_denominator() for element in elements}
            for bd in basis_dicts
        ]

        # Calculate LCM of denominators
        denominators = [f.denominator for f in target_fractions.values()]
        for bf in basis_fractions:
            denominators.extend([f.denominator for f in bf.values()])
        lcm_denominator = 1
        for d in denominators:
            lcm_denominator = (lcm_denominator * d) // np.gcd(lcm_denominator, d)

        # Create coefficient matrix and target vector
        coefficient_matrix = []
        for bf in basis_fractions:
            row = [int(bf[element] * lcm_denominator) for element in elements]
            coefficient_matrix.append(row)

        target_vector = [int(target_fractions[element] * lcm_denominator) for element in elements]

        coefficient_matrix = Matrix(coefficient_matrix).transpose()
        target_vector = Matrix(target_vector)

        # Solve linear system
        solution = coefficient_matrix.solve(target_vector)

        # Check if solution is valid (non-negative)
        if all(x >= 0 for x in solution):
            factorization = {}
            for bc, coeff in zip(temp_basis_compositions, solution):
                factorization[bc] = float(coeff)
            return factorization
        else:
            return None
    except Exception:
        return None


def parse_uploaded_file(uploaded_file, delimiter: str = 'auto') -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[str]]:
    """
    Parse uploaded CSV/TXT file.

    Returns:
        - DataFrame with data
        - List of column names (if header exists)
        - Error message if any
    """
    try:
        content = uploaded_file.read()

        # Try to decode
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('shift-jis')

        # Reset file position
        uploaded_file.seek(0)

        # Detect delimiter
        if delimiter == 'auto':
            first_line = text.split('\n')[0]
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            else:
                delimiter = r'\s+'

        # Check if first line is header (contains non-numeric characters)
        lines = text.strip().split('\n')
        first_line = lines[0]

        # Try to parse first line as numbers
        try:
            if delimiter == r'\s+':
                values = first_line.split()
            else:
                values = first_line.split(delimiter)
            # Check if all values are numeric
            for v in values:
                float(v.strip())
            has_header = False
        except ValueError:
            has_header = True

        # Read data
        if has_header:
            if delimiter == r'\s+':
                df = pd.read_csv(io.StringIO(text), sep=delimiter, engine='python')
            else:
                df = pd.read_csv(io.StringIO(text), sep=delimiter)
            headers = list(df.columns)
        else:
            if delimiter == r'\s+':
                df = pd.read_csv(io.StringIO(text), sep=delimiter, header=None, engine='python')
            else:
                df = pd.read_csv(io.StringIO(text), sep=delimiter, header=None)
            headers = None

        return df, headers, None

    except Exception as e:
        return None, None, str(e)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=['A', 'B', 'C', 'Z'])

    if 'labels' not in st.session_state:
        st.session_state.labels = {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'Z': 'Z'
        }

    if 'z_label_preset' not in st.session_state:
        st.session_state.z_label_preset = 'custom'

    if 'plot_settings' not in st.session_state:
        st.session_state.plot_settings = {
            # Figure size
            'fig_width': 800,
            'fig_height': 700,

            # Colorscale
            'colorscale': 'Viridis',
            'reverse_colorscale': False,

            # Z range
            'z_min': None,
            'z_max': None,
            'auto_z_range': True,

            # Interpolation
            'interpolate': False,
            'interpolation_resolution': 50,
            'interpolation_method': 'linear',

            # Marker settings
            'marker_size': 12,
            'marker_symbol': 'circle',
            'marker_line_color': '#000000',
            'marker_line_width': 1,
            'marker_opacity': 0.8,

            # Axis settings
            'axis_line_width': 2,
            'grid_line_width': 1,
            'show_grid': True,
            'tick_step': 0.2,
            'show_tick_labels': True,

            # Font settings
            'title_font_size': 24,
            'axis_font_size': 20,
            'tick_font_size': 14,

            # Auto subscript
            'auto_subscript': True,

            # Background
            'bgcolor': 'white',

            # Colorbar
            'show_colorbar': True,
            'colorbar_title': '',
        }

    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

    if 'loaded_file_data' not in st.session_state:
        st.session_state.loaded_file_data = None


def create_ternary_plot(
    data: pd.DataFrame,
    labels: Dict[str, str],
    settings: Dict
) -> go.Figure:
    """Create ternary plot with given data and settings."""

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'ternary'}]])

    # Process labels for display
    if settings.get('auto_subscript', True):
        a_label = subscript_num_html(labels.get('A', 'A'))
        b_label = subscript_num_html(labels.get('B', 'B'))
        c_label = subscript_num_html(labels.get('C', 'C'))
    else:
        a_label = labels.get('A', 'A')
        b_label = labels.get('B', 'B')
        c_label = labels.get('C', 'C')

    # Get Z range
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

    # Colorscale
    colorscale = settings.get('colorscale', 'Viridis')
    if settings.get('reverse_colorscale', False):
        colorscale = colorscale + '_r'

    # Normalize data (A + B + C = 1)
    if len(data) > 0:
        # Filter out rows with NaN in A, B, C
        valid_data = data.dropna(subset=['A', 'B', 'C'])

        if len(valid_data) > 0:
            a_vals = valid_data['A'].values.astype(float)
            b_vals = valid_data['B'].values.astype(float)
            c_vals = valid_data['C'].values.astype(float)

            total = a_vals + b_vals + c_vals
            # Avoid division by zero
            total = np.where(total == 0, 1, total)

            a_norm = a_vals / total
            b_norm = b_vals / total
            c_norm = c_vals / total

            z_vals = valid_data['Z'].values if 'Z' in valid_data.columns else None

            # Interpolation
            if settings.get('interpolate', False) and z_vals is not None and len(z_vals) > 3 and not np.all(np.isnan(z_vals)):
                # Create grid for interpolation
                resolution = settings.get('interpolation_resolution', 50)
                method = settings.get('interpolation_method', 'linear')

                # Generate grid points in ternary space
                grid_a = []
                grid_b = []
                grid_c = []

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

                # Interpolate Z values
                try:
                    # Use 2D interpolation in Cartesian coordinates
                    # Convert ternary to Cartesian for interpolation
                    x_data = 0.5 * (2 * b_norm + c_norm)
                    y_data = (np.sqrt(3) / 2) * c_norm

                    x_grid = 0.5 * (2 * grid_b + grid_c)
                    y_grid = (np.sqrt(3) / 2) * grid_c

                    # Filter NaN from z_vals for interpolation
                    valid_z_mask = ~np.isnan(z_vals)
                    if np.sum(valid_z_mask) >= 3:
                        z_interp = griddata(
                            (x_data[valid_z_mask], y_data[valid_z_mask]),
                            z_vals[valid_z_mask],
                            (x_grid, y_grid),
                            method=method
                        )

                        # Filter out NaN values
                        valid_mask = ~np.isnan(z_interp)

                        # Add interpolated surface
                        fig.add_trace(go.Scatterternary(
                            a=grid_a[valid_mask],
                            b=grid_b[valid_mask],
                            c=grid_c[valid_mask],
                            mode='markers',
                            marker=dict(
                                size=settings.get('marker_size', 12),
                                color=z_interp[valid_mask],
                                colorscale=colorscale,
                                cmin=z_min,
                                cmax=z_max,
                                showscale=settings.get('show_colorbar', True),
                                colorbar=dict(
                                    title=settings.get('colorbar_title', '') or labels.get('Z', 'Z'),
                                    titlefont=dict(size=settings.get('axis_font_size', 20)),
                                    tickfont=dict(size=settings.get('tick_font_size', 14)),
                                ),
                                opacity=settings.get('marker_opacity', 0.6),
                                symbol='hexagon2',
                            ),
                            hoverinfo='text',
                            text=[
                                f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}"
                                for a, b, c, z in zip(grid_a[valid_mask], grid_b[valid_mask], grid_c[valid_mask], z_interp[valid_mask])
                            ],
                            showlegend=False
                        ))

                        # Add original data points as overlay
                        fig.add_trace(go.Scatterternary(
                            a=a_norm,
                            b=b_norm,
                            c=c_norm,
                            mode='markers',
                            marker=dict(
                                size=settings.get('marker_size', 12) + 2,
                                color='white',
                                line=dict(
                                    color=settings.get('marker_line_color', '#000000'),
                                    width=settings.get('marker_line_width', 2),
                                ),
                            ),
                            hoverinfo='text',
                            text=[
                                f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}" if not np.isnan(z) else
                                f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                                for a, b, c, z in zip(a_norm, b_norm, c_norm, z_vals)
                            ],
                            showlegend=False
                        ))
                    else:
                        st.warning("Not enough valid Z values for interpolation. Showing raw data.")
                        settings['interpolate'] = False

                except Exception as e:
                    st.warning(f"Interpolation failed: {e}. Showing raw data only.")
                    settings['interpolate'] = False

            if not settings.get('interpolate', False):
                # No interpolation - just plot the data points
                marker_dict = dict(
                    size=settings.get('marker_size', 12),
                    symbol=settings.get('marker_symbol', 'circle'),
                    line=dict(
                        color=settings.get('marker_line_color', '#000000'),
                        width=settings.get('marker_line_width', 1),
                    ),
                    opacity=settings.get('marker_opacity', 0.8),
                )

                if z_vals is not None and len(z_vals) > 0 and not np.all(np.isnan(z_vals)):
                    marker_dict['color'] = z_vals
                    marker_dict['colorscale'] = colorscale
                    marker_dict['cmin'] = z_min
                    marker_dict['cmax'] = z_max
                    marker_dict['showscale'] = settings.get('show_colorbar', True)
                    marker_dict['colorbar'] = dict(
                        title=settings.get('colorbar_title', '') or labels.get('Z', 'Z'),
                        titlefont=dict(size=settings.get('axis_font_size', 20)),
                        tickfont=dict(size=settings.get('tick_font_size', 14)),
                    )

                    hover_text = [
                        f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}<br>{labels['Z']}: {z:.4g}" if not np.isnan(z) else
                        f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                        for a, b, c, z in zip(a_norm, b_norm, c_norm, z_vals)
                    ]
                else:
                    marker_dict['color'] = '#1f77b4'
                    hover_text = [
                        f"{labels['A']}: {a:.3f}<br>{labels['B']}: {b:.3f}<br>{labels['C']}: {c:.3f}"
                        for a, b, c in zip(a_norm, b_norm, c_norm)
                    ]

                fig.add_trace(go.Scatterternary(
                    a=a_norm,
                    b=b_norm,
                    c=c_norm,
                    mode='markers',
                    marker=marker_dict,
                    hoverinfo='text',
                    text=hover_text,
                    showlegend=False
                ))

    # Update layout
    tick_step = settings.get('tick_step', 0.2)

    fig.update_layout(
        ternary=dict(
            sum=1,
            aaxis=dict(
                title=a_label,
                titlefont=dict(size=settings.get('axis_font_size', 20)),
                tickfont=dict(size=settings.get('tick_font_size', 14)),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor='gray' if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=settings.get('grid_line_width', 1),
                tick0=0,
                dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', True),
                ticks='outside' if settings.get('show_tick_labels', True) else '',
            ),
            baxis=dict(
                title=b_label,
                titlefont=dict(size=settings.get('axis_font_size', 20)),
                tickfont=dict(size=settings.get('tick_font_size', 14)),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor='gray' if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=settings.get('grid_line_width', 1),
                tick0=0,
                dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', True),
                ticks='outside' if settings.get('show_tick_labels', True) else '',
            ),
            caxis=dict(
                title=c_label,
                titlefont=dict(size=settings.get('axis_font_size', 20)),
                tickfont=dict(size=settings.get('tick_font_size', 14)),
                linewidth=settings.get('axis_line_width', 2),
                linecolor='black',
                gridcolor='gray' if settings.get('show_grid', True) else 'rgba(0,0,0,0)',
                gridwidth=settings.get('grid_line_width', 1),
                tick0=0,
                dtick=tick_step,
                showticklabels=settings.get('show_tick_labels', True),
                ticks='outside' if settings.get('show_tick_labels', True) else '',
            ),
            bgcolor=settings.get('bgcolor', 'white'),
        ),
        width=settings.get('fig_width', 800),
        height=settings.get('fig_height', 700),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=60, t=60, b=60),
    )

    return fig


def get_sample_data() -> pd.DataFrame:
    """Generate sample ternary data for demonstration."""
    # Li2S-P2S5-LiI system example
    data = [
        {'A': 0.75, 'B': 0.25, 'C': 0.0, 'Z': 0.5},
        {'A': 0.70, 'B': 0.20, 'C': 0.10, 'Z': 1.2},
        {'A': 0.60, 'B': 0.30, 'C': 0.10, 'Z': 2.5},
        {'A': 0.50, 'B': 0.40, 'C': 0.10, 'Z': 3.8},
        {'A': 0.65, 'B': 0.25, 'C': 0.10, 'Z': 2.1},
        {'A': 0.55, 'B': 0.35, 'C': 0.10, 'Z': 3.2},
        {'A': 0.60, 'B': 0.25, 'C': 0.15, 'Z': 2.8},
        {'A': 0.50, 'B': 0.30, 'C': 0.20, 'Z': 4.5},
        {'A': 0.45, 'B': 0.35, 'C': 0.20, 'Z': 5.2},
        {'A': 0.40, 'B': 0.40, 'C': 0.20, 'Z': 4.8},
        {'A': 0.55, 'B': 0.30, 'C': 0.15, 'Z': 3.5},
        {'A': 0.50, 'B': 0.35, 'C': 0.15, 'Z': 4.0},
    ]
    return pd.DataFrame(data)


def render_data_browser():
    """Render data browser section with editable table."""
    st.markdown("### Data Browser")

    # Labels configuration
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        new_a = st.text_input(
            "A Label",
            value=st.session_state.labels.get('A', 'A'),
            key='label_a_input'
        )
        if new_a != st.session_state.labels.get('A'):
            st.session_state.labels['A'] = new_a

    with col2:
        new_b = st.text_input(
            "B Label",
            value=st.session_state.labels.get('B', 'B'),
            key='label_b_input'
        )
        if new_b != st.session_state.labels.get('B'):
            st.session_state.labels['B'] = new_b

    with col3:
        new_c = st.text_input(
            "C Label",
            value=st.session_state.labels.get('C', 'C'),
            key='label_c_input'
        )
        if new_c != st.session_state.labels.get('C'):
            st.session_state.labels['C'] = new_c

    with col4:
        z_presets = {
            'custom': 'Custom',
            'sigma': 'σ / S cm⁻¹',
            'ea': 'Eₐ / kJ mol⁻¹',
            'capacity': 'Capacity / mAh g⁻¹',
        }

        z_preset = st.selectbox(
            "Z Label Preset",
            options=list(z_presets.keys()),
            format_func=lambda x: z_presets[x],
            index=list(z_presets.keys()).index(st.session_state.z_label_preset),
            key='z_preset_select'
        )

        if z_preset != st.session_state.z_label_preset:
            st.session_state.z_label_preset = z_preset
            if z_preset != 'custom':
                st.session_state.labels['Z'] = z_presets[z_preset]

        if z_preset == 'custom':
            new_z = st.text_input(
                "Z Label",
                value=st.session_state.labels.get('Z', 'Z'),
                key='label_z_input'
            )
            if new_z != st.session_state.labels.get('Z'):
                st.session_state.labels['Z'] = new_z

    # Data table
    st.markdown("#### Composition Data")

    # Prepare display dataframe
    display_df = st.session_state.data.copy()

    # Add SUM column (reduced formula)
    if PYMATGEN_AVAILABLE and len(display_df) > 0:
        sum_formulas = []
        for idx, row in display_df.iterrows():
            try:
                a_label = st.session_state.labels.get('A', 'A')
                b_label = st.session_state.labels.get('B', 'B')
                c_label = st.session_state.labels.get('C', 'C')

                a_val = float(row['A']) if pd.notna(row['A']) else 0
                b_val = float(row['B']) if pd.notna(row['B']) else 0
                c_val = float(row['C']) if pd.notna(row['C']) else 0

                if a_val > 0 or b_val > 0 or c_val > 0:
                    comp_str = ""
                    if a_val > 0:
                        comp_str += f"({a_label}){a_val}"
                    if b_val > 0:
                        comp_str += f"({b_label}){b_val}"
                    if c_val > 0:
                        comp_str += f"({c_label}){c_val}"

                    comp = Composition(comp_str)
                    sum_formulas.append(comp.reduced_formula)
                else:
                    sum_formulas.append("")
            except Exception:
                sum_formulas.append("")

        display_df['SUM'] = sum_formulas
    else:
        display_df['SUM'] = ""

    # Editable data table
    edited_df = st.data_editor(
        display_df,
        column_config={
            "A": st.column_config.NumberColumn(
                st.session_state.labels.get('A', 'A'),
                help="Component A ratio",
                min_value=0,
                format="%.4f",
            ),
            "B": st.column_config.NumberColumn(
                st.session_state.labels.get('B', 'B'),
                help="Component B ratio",
                min_value=0,
                format="%.4f",
            ),
            "C": st.column_config.NumberColumn(
                st.session_state.labels.get('C', 'C'),
                help="Component C ratio",
                min_value=0,
                format="%.4f",
            ),
            "Z": st.column_config.NumberColumn(
                st.session_state.labels.get('Z', 'Z'),
                help="Property value (for heatmap)",
                format="%.4g",
            ),
            "SUM": st.column_config.TextColumn(
                "SUM (Formula)",
                help="Reduced formula calculated from A+B+C",
                disabled=True,
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key='data_editor'
    )

    # Update session state with edited data (excluding SUM column)
    if 'SUM' in edited_df.columns:
        edited_df = edited_df.drop(columns=['SUM'])
    st.session_state.data = edited_df

    # Clear data button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Clear All Data", key='clear_data_btn'):
            st.session_state.data = pd.DataFrame(columns=['A', 'B', 'C', 'Z'])
            st.rerun()

    # Formula to ABC conversion
    if PYMATGEN_AVAILABLE and SYMPY_AVAILABLE:
        st.markdown("#### Formula to Composition")
        st.caption("Enter a formula to factorize into A, B, C components")

        col1, col2 = st.columns([3, 1])

        with col1:
            formula_input = st.text_input(
                "Formula",
                placeholder="e.g., Li3PS4",
                key='formula_input',
                label_visibility="collapsed"
            )

        with col2:
            convert_clicked = st.button("Convert", key='convert_btn')

        if formula_input and convert_clicked:
            basis = [
                st.session_state.labels.get('A', 'A'),
                st.session_state.labels.get('B', 'B'),
                st.session_state.labels.get('C', 'C'),
            ]

            result = factorize_composition(formula_input, basis)

            if result:
                st.success("Factorization successful!")
                result_str = ", ".join([f"{comp}: {coeff:.4f}" for comp, coeff in result.items()])
                st.write(result_str)

                if st.button("Add to Data", key='add_factorized'):
                    new_row = pd.DataFrame({
                        'A': [result.get(basis[0], 0)],
                        'B': [result.get(basis[1], 0)],
                        'C': [result.get(basis[2], 0)],
                        'Z': [np.nan]
                    })
                    st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)
                    st.rerun()
            else:
                st.warning("Could not factorize the formula into the given components.")


def render_data_loader():
    """Render data loader section."""
    st.markdown("### Data Loader")

    # Sample data button
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Load Sample Data", key='load_sample_btn'):
            st.session_state.data = get_sample_data()
            st.session_state.labels = {
                'A': 'Li2S',
                'B': 'P2S5',
                'C': 'LiI',
                'Z': 'σ / mS cm⁻¹'
            }
            st.success("Sample data loaded!")
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload CSV/TXT file",
        type=['csv', 'txt', 'tsv'],
        help="Upload a file with A, B, C (and optionally Z) columns. "
             "First row with text will be treated as header.",
        key='file_uploader'
    )

    if uploaded_file is not None:
        # Parse file if it's new or changed
        if st.session_state.uploaded_file_name != uploaded_file.name:
            df, headers, error = parse_uploaded_file(uploaded_file)
            if error:
                st.error(f"Error loading file: {error}")
            else:
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.loaded_file_data = {'df': df, 'headers': headers}
                st.success(f"Parsed {len(df)} rows from {uploaded_file.name}")

        # Show column mapping if file is loaded
        if st.session_state.loaded_file_data is not None:
            df = st.session_state.loaded_file_data['df']
            headers = st.session_state.loaded_file_data['headers']

            st.markdown("#### Column Mapping")

            available_cols = list(df.columns)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                a_col = st.selectbox(
                    "A Column",
                    options=available_cols,
                    index=0 if len(available_cols) > 0 else None,
                    key='a_col_select'
                )

            with col2:
                b_col = st.selectbox(
                    "B Column",
                    options=available_cols,
                    index=1 if len(available_cols) > 1 else 0,
                    key='b_col_select'
                )

            with col3:
                c_col = st.selectbox(
                    "C Column",
                    options=available_cols,
                    index=2 if len(available_cols) > 2 else 0,
                    key='c_col_select'
                )

            with col4:
                z_options = ['(None)'] + available_cols
                z_default_idx = 4 if len(available_cols) > 3 else 0
                z_col = st.selectbox(
                    "Z Column (optional)",
                    options=z_options,
                    index=z_default_idx,
                    key='z_col_select'
                )

            # Auto-update labels from header
            use_headers = False
            if headers:
                use_headers = st.checkbox("Use column names as labels", value=True, key='use_headers')

            if st.button("Load Data", type="primary", key='load_data_btn'):
                # Create new dataframe with mapped columns
                new_data = pd.DataFrame()
                new_data['A'] = pd.to_numeric(df[a_col], errors='coerce')
                new_data['B'] = pd.to_numeric(df[b_col], errors='coerce')
                new_data['C'] = pd.to_numeric(df[c_col], errors='coerce')

                if z_col != '(None)':
                    new_data['Z'] = pd.to_numeric(df[z_col], errors='coerce')
                else:
                    new_data['Z'] = np.nan

                # Update labels if using headers
                if use_headers and headers:
                    if a_col and not str(a_col).isdigit():
                        st.session_state.labels['A'] = str(a_col)
                    if b_col and not str(b_col).isdigit():
                        st.session_state.labels['B'] = str(b_col)
                    if c_col and not str(c_col).isdigit():
                        st.session_state.labels['C'] = str(c_col)
                    if z_col and z_col != '(None)' and not str(z_col).isdigit():
                        st.session_state.labels['Z'] = str(z_col)

                st.session_state.data = new_data
                st.success(f"Loaded {len(new_data)} data points")
                st.rerun()


def render_plot_settings():
    """Render plot settings panel."""
    st.markdown("### Plot Settings")

    settings = st.session_state.plot_settings

    # Use tabs for organization
    tab1, tab2, tab3, tab4 = st.tabs(["Size & Color", "Markers", "Axis", "Interpolation"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            settings['fig_width'] = st.number_input(
                "Width (px)",
                min_value=400,
                max_value=2000,
                value=settings.get('fig_width', 800),
                step=50,
                key='fig_width_input'
            )

            settings['fig_height'] = st.number_input(
                "Height (px)",
                min_value=400,
                max_value=2000,
                value=settings.get('fig_height', 700),
                step=50,
                key='fig_height_input'
            )

        with col2:
            colorscales = [
                'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                'Turbo', 'Jet', 'Hot', 'Cool', 'Rainbow',
                'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'Blues', 'Reds', 'Greens'
            ]

            settings['colorscale'] = st.selectbox(
                "Colorscale",
                options=colorscales,
                index=colorscales.index(settings.get('colorscale', 'Viridis')),
                key='colorscale_select'
            )

            settings['reverse_colorscale'] = st.checkbox(
                "Reverse colorscale",
                value=settings.get('reverse_colorscale', False),
                key='reverse_colorscale_check'
            )

        st.markdown("#### Z Range")
        settings['auto_z_range'] = st.checkbox(
            "Auto Z range",
            value=settings.get('auto_z_range', True),
            key='auto_z_range_check'
        )

        if not settings['auto_z_range']:
            col1, col2 = st.columns(2)
            with col1:
                settings['z_min'] = st.number_input(
                    "Z Min",
                    value=float(settings.get('z_min', 0.0) or 0.0),
                    format="%.4g",
                    key='z_min_input'
                )
            with col2:
                settings['z_max'] = st.number_input(
                    "Z Max",
                    value=float(settings.get('z_max', 1.0) or 1.0),
                    format="%.4g",
                    key='z_max_input'
                )

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            settings['marker_size'] = st.slider(
                "Marker Size",
                min_value=2,
                max_value=30,
                value=settings.get('marker_size', 12),
                key='marker_size_slider'
            )

            marker_symbols = [
                'circle', 'square', 'diamond', 'cross', 'x',
                'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star'
            ]
            settings['marker_symbol'] = st.selectbox(
                "Marker Symbol",
                options=marker_symbols,
                index=marker_symbols.index(settings.get('marker_symbol', 'circle')),
                key='marker_symbol_select'
            )

        with col2:
            settings['marker_line_color'] = st.color_picker(
                "Marker Edge Color",
                value=settings.get('marker_line_color', '#000000'),
                key='marker_line_color_picker'
            )

            settings['marker_line_width'] = st.slider(
                "Marker Edge Width",
                min_value=0,
                max_value=5,
                value=settings.get('marker_line_width', 1),
                key='marker_line_width_slider'
            )

            settings['marker_opacity'] = st.slider(
                "Marker Opacity",
                min_value=0.0,
                max_value=1.0,
                value=settings.get('marker_opacity', 0.8),
                step=0.05,
                key='marker_opacity_slider'
            )

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            settings['axis_line_width'] = st.slider(
                "Axis Line Width",
                min_value=1,
                max_value=5,
                value=settings.get('axis_line_width', 2),
                key='axis_line_width_slider'
            )

            settings['grid_line_width'] = st.slider(
                "Grid Line Width",
                min_value=0,
                max_value=3,
                value=settings.get('grid_line_width', 1),
                key='grid_line_width_slider'
            )

            settings['show_grid'] = st.checkbox(
                "Show Grid",
                value=settings.get('show_grid', True),
                key='show_grid_check'
            )

        with col2:
            settings['tick_step'] = st.select_slider(
                "Tick Step",
                options=[0.1, 0.2, 0.25, 0.5],
                value=settings.get('tick_step', 0.2),
                key='tick_step_slider'
            )

            settings['show_tick_labels'] = st.checkbox(
                "Show Tick Labels",
                value=settings.get('show_tick_labels', True),
                key='show_tick_labels_check'
            )

            settings['auto_subscript'] = st.checkbox(
                "Auto Subscript Numbers",
                value=settings.get('auto_subscript', True),
                help="Automatically convert numbers in labels to subscript (e.g., Li2O -> Li₂O)",
                key='auto_subscript_check'
            )

        st.markdown("#### Font Sizes")
        col1, col2, col3 = st.columns(3)

        with col1:
            settings['title_font_size'] = st.number_input(
                "Title Font",
                min_value=8,
                max_value=48,
                value=settings.get('title_font_size', 24),
                key='title_font_size_input'
            )

        with col2:
            settings['axis_font_size'] = st.number_input(
                "Axis Font",
                min_value=8,
                max_value=48,
                value=settings.get('axis_font_size', 20),
                key='axis_font_size_input'
            )

        with col3:
            settings['tick_font_size'] = st.number_input(
                "Tick Font",
                min_value=8,
                max_value=48,
                value=settings.get('tick_font_size', 14),
                key='tick_font_size_input'
            )

    with tab4:
        settings['interpolate'] = st.checkbox(
            "Enable Interpolation",
            value=settings.get('interpolate', False),
            help="Interpolate between data points to create a smooth heatmap",
            key='interpolate_check'
        )

        if settings['interpolate']:
            col1, col2 = st.columns(2)

            with col1:
                settings['interpolation_resolution'] = st.slider(
                    "Resolution",
                    min_value=10,
                    max_value=100,
                    value=settings.get('interpolation_resolution', 50),
                    help="Higher values create smoother interpolation but may be slower",
                    key='interpolation_resolution_slider'
                )

            with col2:
                methods = ['linear', 'cubic', 'nearest']
                settings['interpolation_method'] = st.selectbox(
                    "Method",
                    options=methods,
                    index=methods.index(settings.get('interpolation_method', 'linear')),
                    key='interpolation_method_select'
                )

        st.markdown("#### Colorbar")
        settings['show_colorbar'] = st.checkbox(
            "Show Colorbar",
            value=settings.get('show_colorbar', True),
            key='show_colorbar_check'
        )

        if settings['show_colorbar']:
            settings['colorbar_title'] = st.text_input(
                "Colorbar Title",
                value=settings.get('colorbar_title', ''),
                placeholder="Leave empty to use Z label",
                key='colorbar_title_input'
            )

    # Update session state
    st.session_state.plot_settings = settings


def main():
    """Main application function."""
    initialize_session_state()

    # Title
    st.markdown("# Ternary Plot Analyzer")
    st.markdown("Three-component composition diagram visualization for publications")

    # Main layout
    st.markdown("---")

    # Upper section - Ternary plot
    plot_col, spacer = st.columns([3, 1])

    with plot_col:
        st.markdown("### Ternary Plot")

        # Create and display plot
        if len(st.session_state.data) > 0:
            fig = create_ternary_plot(
                st.session_state.data,
                st.session_state.labels,
                st.session_state.plot_settings
            )
            st.plotly_chart(fig, key='ternary_plot')

            # Export buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                try:
                    img_bytes = fig.to_image(format="png", scale=2)
                    st.download_button(
                        label="Download PNG",
                        data=img_bytes,
                        file_name="ternary_plot.png",
                        mime="image/png",
                        key='download_png'
                    )
                except Exception:
                    st.caption("PNG export requires kaleido package")

            with col2:
                try:
                    svg_bytes = fig.to_image(format="svg")
                    st.download_button(
                        label="Download SVG",
                        data=svg_bytes,
                        file_name="ternary_plot.svg",
                        mime="image/svg+xml",
                        key='download_svg'
                    )
                except Exception:
                    st.caption("SVG export requires kaleido package")

            with col3:
                export_df = st.session_state.data.copy()
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Data (CSV)",
                    data=csv,
                    file_name="ternary_data.csv",
                    mime="text/csv",
                    key='download_csv'
                )
        else:
            st.info("No data to display. Upload a file or add data manually below.")

    st.markdown("---")

    # Lower section - Data and Settings
    left_col, right_col = st.columns([1, 1])

    with left_col:
        render_data_loader()
        st.markdown("---")
        render_data_browser()

    with right_col:
        render_plot_settings()


if __name__ == "__main__":
    main()
