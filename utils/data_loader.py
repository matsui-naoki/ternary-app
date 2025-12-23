"""Data loading and parsing utilities."""

import io
from typing import Optional, List, Tuple
import pandas as pd


def parse_uploaded_file(uploaded_file, delimiter: str = 'auto') -> Tuple[Optional[pd.DataFrame], Optional[List[str]], Optional[str]]:
    """Parse uploaded CSV/TXT file.

    Args:
        uploaded_file: Streamlit uploaded file object
        delimiter: Delimiter to use ('auto', ',', '\\t', or regex pattern)

    Returns:
        Tuple of (DataFrame, headers list or None, error message or None)
    """
    try:
        content = uploaded_file.read()

        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('shift-jis')

        uploaded_file.seek(0)

        if delimiter == 'auto':
            first_line = text.split('\n')[0]
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            else:
                delimiter = r'\s+'

        lines = text.strip().split('\n')
        first_line = lines[0]

        try:
            if delimiter == r'\s+':
                values = first_line.split()
            else:
                values = first_line.split(delimiter)
            for v in values:
                float(v.strip())
            has_header = False
        except ValueError:
            has_header = True

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


def get_sample_data() -> pd.DataFrame:
    """Generate sample ternary data for demonstration.

    Returns:
        DataFrame with sample Li2S-P2S5-LiI composition data
    """
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
