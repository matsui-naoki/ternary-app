"""Help text strings for UI tooltips."""

INTERPOLATION_METHODS = {
    'linear': 'Linear interpolation: Creates smooth surfaces by linearly interpolating between data points. Best for evenly distributed data.',
    'cubic': 'Cubic interpolation: Uses cubic splines for smoother results. Better for sparse data but may overshoot near edges.',
    'nearest': 'Nearest neighbor: Uses the value of the closest data point. Creates stepped/blocky appearance. Fast and robust.',
}

HEATMAP_MARKER_MODES = {
    'white': 'White fill: Data points shown as white circles with colored border',
    'fill': 'Color fill: Data points filled with their Z-value color',
    'hide': 'Hide markers: Only show interpolated heatmap, hide original data points',
}

LOG_SCALE_HELP = 'Apply logarithmic transformation to Z values for colorscale. Useful for data spanning multiple orders of magnitude.'

DISCRETE_COLORS_HELP = 'Divide the colorscale into discrete steps instead of continuous gradient. Useful for categorical visualization.'
