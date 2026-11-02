import numpy as np

ELEMENTS = {
    'W': {'mass': 183.84, 'a0': 3.165},
    'Mo': {'mass': 95.95, 'a0': 3.147},
    'V': {'mass': 50.94, 'a0': 3.027}}

def strip_split(s: str, sep=None, item_type=None):
    """Strip a string of whitespace and split it apart given a separator character."""
    s = s.strip()
    s = s.split(sep)
    if item_type is int:
        return [int(x) for x in s]
    elif item_type is float:
        return [float(x) for x in s]
    elif item_type is None:
        return s
    else:
        raise ValueError(f'[{item_type}] Invalid item type. Choose None, int, or float')

def product(x: list[int|float]):
    """Compute the product of items in a list, similar to the built-in sum()."""
    prod = 1
    for v in x:
        prod *= v
    return prod

def is_hnf(H: np.ndarray):
    """Check if a 3x3 integer matrix is in Hermite normal form."""
    if H.shape != (3,3):
        return False

    if np.any(H < 0):
        return False
    
    if not np.all(H == np.floor(H)):
        return False
    
    rowc, colc = False, False
    if np.any(H[np.tril_indices(3, k=-1)]):
        rowc = True
    if np.any(H[np.triu_indices(3, k=1)]):
        colc = True
    if all([rowc, colc]):
        return False

    if rowc:
        for el1, el2 in zip([(1,0), (2,0), (2,1)], [(1,1), (2,2), (2,2)]):
            if H[el1] >= H[el2]:
                return False
    
    if colc:
        for el1, el2 in zip([(0,1), (0,2), (1,2)], [(0,0), (0,0), (1,1)]):
            if H[el1] >= H[el2]:
                return False

    return True

def composition_distance(comp1: tuple[float], comp2: tuple[float]):
    """Compute the Euclidean distance between two compositions of the same rank."""
    if len(comp1) != len(comp2):
        raise ValueError(f"Compositions have different rank (C1={len(comp1)}, C2={len(comp2)})")
    
    distance = 0
    for i in range(len(comp1)):
        distance += (comp1[i] - comp2[i])**2
    
    return distance**(1/2)

def replicate(A: np.ndarray, unit_cell_positions: list[np.ndarray], size: tuple[int]):
    """Replicate the unit cell of a lattice an integer number of times in each direction."""
    supercell_positions: list[tuple[float]] = []

    for bp in unit_cell_positions:
        for tr_vector in list(np.ndindex(size)):
            pos = bp + A @ tr_vector
            pos = tuple(pos.tolist())
            supercell_positions.append(pos)

    return supercell_positions

def alloy_string(elements: tuple[str]):
    alloy_string = ''
    for el in elements:
        alloy_string += el +'-'
    return alloy_string[:-1]