# Ternary Plot Visualizer

Three-component composition diagram visualization tool for academic papers and presentations.

## Features

- **Interactive Ternary Plot**: Visualize three-component composition data with Plotly
- **Data Loading**: Support for CSV, TXT, TSV files with automatic delimiter detection
- **Editable Data Table**: Add, edit, and delete data points directly in the browser
- **Composition Formula Support**:
  - Automatic reduced formula calculation (A+B+C → reduced formula) using pymatgen
  - Formula factorization (input a formula like Li3PS4 to auto-calculate A/B/C coefficients)
- **Publication-Ready Figures**:
  - Customizable figure size, fonts, and colors
  - Multiple colorscales (Viridis, Plasma, Jet, etc.) with discrete/continuous options
  - Auto-subscript for chemical formulas (Li2O → Li₂O)
  - Colorbar customization (size, position, ticks, title placement)
  - PNG/SVG export at high resolution
- **Interpolation**: Smooth heatmap interpolation between data points
- **Z Value Heatmap**: Color-coded property values with preset labels (σ, Ea, capacity) or custom

## Installation

```bash
# Clone the repository
git clone https://github.com/matsui-naoki/ternary-app.git
cd ternary-app

# Install dependencies
pip install streamlit numpy pandas plotly scipy pymatgen sympy
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Data Format

The application accepts CSV/TXT files with:
- **Columns**: A, B, C (composition ratios) and optionally Z (property value)
- **Header**: If the first row contains text, it's treated as column headers
- **Delimiter**: Auto-detected (comma, tab, or space)

Example:
```csv
Li2S,P2S5,LiI,sigma
0.75,0.25,0.0,0.5
0.70,0.20,0.10,1.2
0.60,0.30,0.10,2.5
```

## Requirements

- Python 3.8+
- streamlit
- numpy
- pandas
- plotly
- scipy
- pymatgen
- sympy

## License

MIT License
