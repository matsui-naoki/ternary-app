"""Text formatting utilities for chemical formulas."""

import re


def subscript_num_html(composition: str) -> str:
    """Convert numbers in composition formula to HTML subscript format.

    Args:
        composition: Chemical formula string (e.g., "Li2S", "P2S5")

    Returns:
        HTML string with subscript numbers (e.g., "Li<sub>2</sub>S")
    """
    if not composition:
        return composition
    result = re.sub(r'(\d+\.?\d*)', r'<sub>\1</sub>', composition)
    result = result.replace('<sub>1</sub>', '')
    return result
