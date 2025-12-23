"""Composition handling utilities using pymatgen and sympy."""

from typing import Optional, Dict, List
import numpy as np

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


def factorize_composition(target_composition: str, basis_compositions: List[str]) -> Optional[Dict[str, float]]:
    """Factorize a target composition into basis compositions.

    Args:
        target_composition: Target composition formula (e.g., "Li3PS4")
        basis_compositions: List of basis composition formulas (e.g., ["Li2S", "P2S5", "LiI"])

    Returns:
        Dictionary mapping basis compositions to their coefficients, or None if factorization fails
    """
    if not PYMATGEN_AVAILABLE or not SYMPY_AVAILABLE:
        return None

    try:
        temp_basis_compositions = basis_compositions
        target_dict = Composition(target_composition).as_dict()
        basis_dicts = [Composition(bc).as_dict() for bc in basis_compositions]

        elements = sorted(set(
            element
            for composition in basis_dicts + [target_dict]
            for element in composition
        ))

        target_fractions = {
            element: Fraction(target_dict.get(element, 0)).limit_denominator()
            for element in elements
        }
        basis_fractions = [
            {element: Fraction(bd.get(element, 0)).limit_denominator() for element in elements}
            for bd in basis_dicts
        ]

        denominators = [f.denominator for f in target_fractions.values()]
        for bf in basis_fractions:
            denominators.extend([f.denominator for f in bf.values()])
        lcm_denominator = 1
        for d in denominators:
            lcm_denominator = (lcm_denominator * d) // np.gcd(lcm_denominator, d)

        coefficient_matrix = []
        for bf in basis_fractions:
            row = [int(bf[element] * lcm_denominator) for element in elements]
            coefficient_matrix.append(row)

        target_vector = [int(target_fractions[element] * lcm_denominator) for element in elements]

        coefficient_matrix = Matrix(coefficient_matrix).transpose()
        target_vector = Matrix(target_vector)

        solution = coefficient_matrix.solve(target_vector)

        if all(x >= 0 for x in solution):
            factorization = {}
            for bc, coeff in zip(temp_basis_compositions, solution):
                factorization[bc] = float(coeff)
            return factorization
        else:
            return None
    except Exception:
        return None
