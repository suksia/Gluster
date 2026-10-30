import numpy as np
from math import floor
import itertools, more_itertools
from copy import deepcopy


class Sublattice:
    """A sublattice or quotient group of a parent lattice defined by a transformation matrix H."""
    def __init__(self, H: np.ndarray, A: np.ndarray, rotations: list[np.ndarray]):
        if type(H) != np.ndarray:
            raise ValueError('H must be a numpy array')
        
        self.H = H  # HNF
        self.basis_size = round(np.linalg.det(self.H))

        self.A, self.A_inv = A, np.linalg.inv(A)
        self.rotations = rotations

        self.D, self.U, self.V = None, None, None   # SNF matrices
        self.snf_positions, self.frac_positions = None, None

        self.all_decorations = None
        self.decorations_no_trans, self.decorations_no_super, self.decorations_no_rot = None, None, None
        self.decorations = None

    def compute_SNF(self):
        """Compute the Smith normal form of the transformation matrix H."""
        def reduce(d, uni, coord):
            pivot = (min(coord), min(coord))
            q = floor(d[coord] / d[pivot])
            e = np.eye(3)
            e[coord] = -q
            # row operation
            if coord[0] > coord[1]: 
                return e @ d, e @ uni
            # col operation
            else:
                return d @ e, uni @ e

        def exchange(d, uni, coord):
            e = np.eye(3)
            # row operation
            if coord[0] > coord[1]: 
                e[[coord[0], coord[1]]] = e[[coord[1], coord[0]]]
                return e @ d, e @ uni
            # col operation
            else:                   
                e[[[coord[0]], [coord[1]]]] = e[[[coord[1]], [coord[0]]]]
                return d @ e, uni @ e

        def reduce_coords(d, u, v, coords):
            pivot = None
            for c in coords:
                new_pivot = (min(c), min(c))
                if pivot is None:
                    pivot = new_pivot
                elif pivot != new_pivot:
                    raise ValueError(f'Coordinates have different pivots {pivot} vs. {new_pivot}')

            for c in coords:
                # row operations
                if c[0] > c[1]:
                    uni = u.copy()
                # col operations
                else:
                    uni = v.copy()

                while d[c]:
                    # add or subtract (reduce) current coord
                    if abs(d[pivot]) <= abs(d[c]):
                        d, uni = reduce(d, uni, c)
                    # exchange with pivot
                    else:
                        d, uni = exchange(d, uni, c)

                # row operations
                if c[0] > c[1]:
                    u = uni.copy()
                # col operations
                else:
                    v = uni.copy()

            return d, u, v

        D = self.H.copy()
        U, V = np.eye(3), np.eye(3)

        i, snf_found = 0, False
        while not snf_found:
            # check if diagonal
            if np.all(D * [[0, 1, 1], [1, 0, 1], [1, 1, 0]] == 0):
                # flip sign on negative diagonal entries
                for dc in [(0,0), (1,1), (2,2)]:
                    if D[dc] < 0:
                        e = np.eye(3)
                        e[dc] = -1
                        D, U = e @ D, e @ U

                # check divisibility; if not, mix rows and re-reduce
                if D[1,1] % D[0,0] == 0:
                    snf_found = True
                else:
                    snf_found = False
                    e = np.eye(3)
                    e[0,1] = 1
                    D, U = e @ D, e @ U

                if D[2,2] % D[1,1] == 0:
                    snf_found = True
                else:
                    snf_found = False
                    e = np.eye(3)
                    e[1,2] = 1
                    D, U = e @ D, e @ U

            # outer rows and columns first
            outer_coords = [(1,0), (2,0), (0,1), (0,2)]
            D, U, V = reduce_coords(D, U, V, outer_coords)

            # inner rows and columns next
            if np.all(D * [[0, 1, 1], [1, 0, 0], [1, 0, 0]] == 0):
                inner_coords = [(2,1), (1,2)]
                D, U, V = reduce_coords(D, U, V, inner_coords)

        self.D, self.U, self.V = np.rint(D.copy()).astype('int'), np.rint(U.copy()).astype('int'), np.rint(V.copy()).astype('int')

        self.snf_positions = list(itertools.product(range(self.D[0,0]), range(self.D[1,1]), range(self.D[2,2])))
        self.frac_positions = []
                    
        for pos in self.snf_positions:
            frac_pos = (np.linalg.inv(self.H) @ np.linalg.inv(self.U) @ np.array(pos)) % 1
            frac_pos = np.where(np.isclose(frac_pos, 1.0), 0.0, frac_pos)
            frac_pos = tuple(np.round(frac_pos, 8).tolist())
            self.frac_positions.append(frac_pos)

    def enumerate_bases(self, composition: dict[str, float]):
        if self.D is None:
            self.compute_SNF()

        decoration = []
        for el, conc in composition.items():
            decoration += round(conc*self.basis_size)*[el]

        if len(decoration) != self.basis_size:
            raise ValueError(f'Decoration incompatible with basis size ({len(decoration)} should be {self.basis_size})')

        # non-duplicate permutations of the decoration
        decorations = list(more_itertools.distinct_permutations(decoration))
        self.all_decorations = deepcopy(decorations)

        # define classes of equivalent decorations by translational symmetry
        equiv_classes = []
        for decoration in decorations:
            duplicate = False

            for existing_eq_cls in equiv_classes:
                if decoration in existing_eq_cls:
                    duplicate = True
                    break

            if not duplicate:
                chem_map = {}
                for p, c in zip(self.snf_positions, decoration):
                    chem_map[p] = c

                positions = np.array(self.snf_positions)
                translations = deepcopy(positions)[1:]

                eq_cls = [decoration]
                for t in translations:
                    new_positions = []
                    for new_pos in (positions + t) % np.diag(self.D):
                        new_positions.append(tuple([int(v) for v in new_pos]))

                    new_decoration = tuple([chem_map[new_pos] for new_pos in new_positions])
                    eq_cls.append(new_decoration)
                equiv_classes.append(eq_cls)

        # keep canonical decorations (first in each class)
        decorations = [eq_cls[0] for eq_cls in equiv_classes]
        self.decorations_no_trans = deepcopy(decorations)

        # check each decoration for super-periodicity (translation produces itself)
        new_decorations = []
        for decoration in decorations:
            super_periodic = False

            chem_map = {}
            for p, c in zip(self.snf_positions, decoration):
                chem_map[p] = c

            positions = np.array(self.snf_positions)
            translations = deepcopy(positions)[1:]

            for t in translations:
                new_positions = []
                for new_pos in (positions + t) % np.diag(self.D):
                    new_positions.append(tuple([int(v) for v in new_pos]))

                new_decoration = tuple([chem_map[new_pos] for new_pos in new_positions])
                if new_decoration == decoration:
                    super_periodic = True
                    break

            if not super_periodic:
                new_decorations.append(decoration)

        decorations = new_decorations
        self.decorations_no_super = deepcopy(decorations)
        
        # define classes of equivalent decorations by rotational symmetry
        equiv_classes = []
        for decoration in decorations:
            duplicate = False

            for existing_eq_cls in equiv_classes:
                if decoration in existing_eq_cls:
                    duplicate = True
                    break

            if not duplicate:
                chem_map = {}
                for p, c in zip(self.snf_positions, decoration):
                    chem_map[p] = c

                G = np.array(self.snf_positions).T

                eq_cls = [decoration]
                for R in self.rotations:
                    R_snf = self.U @ self.A_inv @ R @ self.A @ np.linalg.inv(self.U)
                    
                    # SNF rotation should be a valid rotation of the group
                    Z = np.round(np.linalg.inv(self.D) @ R_snf @ self.D, 8)
                    Z_mod = Z % 1

                    if np.all(Z_mod == 0):
                        continue

                    new_G = np.rint(R_snf @ G).astype(int)

                    new_positions = []
                    for new_pos in new_G.T:
                        new_pos %= np.diag(self.D)
                        new_positions.append(tuple([int(v) for v in new_pos]))

                    new_decoration = tuple([chem_map[new_pos] for new_pos in new_positions])
                    eq_cls.append(new_decoration)
                    
                equiv_classes.append(eq_cls)

        decorations = [eq_cls[0] for eq_cls in equiv_classes]
        self.decorations_no_rot = deepcopy(decorations)
        self.decorations = deepcopy(decorations)


class Lattice:
    """A parent lattice which contains all possible sublattices up to a maximum basis size."""
    def __init__(self, L: int):
        self.L = L
        self.sublattices: dict[int, list[Sublattice]] = None

        # defined in subclass (e.g., BCC, FCC)
        self.A = None
        self.rotations = None

    def enumerate_HNF(self, print_info = False):
        """Construct all possible HNF matrices and de-duplicate them using the symmetry of the subclass."""
        non_unique_hnfs = {l: [] for l in range(1, self.L+1)}
        
        for order in range(1, self.L+1):
            H = np.eye(3)
            for d1 in range(1, order+1):
                for d2 in range(1, order+1):
                    if d1*d2 <= order:
                        H[0,0] = d1
                        H[1,1] = d2
                        for l21 in range(d2):
                            H[1,0] = l21
                            for d3 in range(1, order+1):
                                if d1*d2*d3 == order:
                                    H[2,2] = d3
                                    for l31 in range(d3):
                                        H[2,0] = l31
                                        for l32 in range(d3):
                                            H[2,1] = l32
                                            non_unique_hnfs[order].append(H.copy())
        
        hnfs_equiv_classes: dict[int, list[list]] = {l: [] for l in range(1, self.L+1)}

        hnf_count, unique_hnf_count = 0, 0

        for l, hnf_list in non_unique_hnfs.items():
            for H in hnf_list:
                # create first equivalence class
                if len(hnfs_equiv_classes[l]) == 0:
                    hnfs_equiv_classes[l].append([H])
                    continue

                B = self.A @ H

                # check if H is equivalent to any in an existing equivalence class
                duplicate = False
                for j, existing_H_eq_list in enumerate(hnfs_equiv_classes[l]):
                    if duplicate:
                        break

                    for H_ref in existing_H_eq_list:
                        if duplicate:
                            break

                        B_ref = self.A @ H_ref
                        B_ref_inv = np.linalg.inv(B_ref)

                        for R in self.rotations:
                            R_inv = np.linalg.inv(R)
                            Z = np.round(B_ref_inv @ R_inv @ B, 8)
                            Z_mod = Z % 1

                            if np.all(Z_mod == 0): # z is matrix of integers = duplicate
                                hnfs_equiv_classes[l][j].append(H)
                                duplicate = True
                                break

                # create new equivalence class if it is not a duplicate
                if not duplicate:
                    hnfs_equiv_classes[l].append([H])

            hnf_count += len(hnf_list)
            unique_hnf_count += len(hnfs_equiv_classes[l])

            if print_info:
                print(f'l = {l}\t total HNFs = {len(hnf_list)} ({hnf_count})\t unique HNFs = {len(hnfs_equiv_classes[l])} ({unique_hnf_count})')

        # only save canonical (first) H in each equivalence class
        self.sublattices = {l: [] for l in range(1, self.L+1)}

        for l, all_eq_cls in hnfs_equiv_classes.items():
            for eq_cls in all_eq_cls:
                self.sublattices[l].append(Sublattice(eq_cls[0].copy(), self.A, self.rotations))


class BCC(Lattice):
    def __init__(self, max_basis_size: int):
        super().__init__(max_basis_size)

        self.A = 0.5*np.array([[1, -1, 1], [-1, 1, 1], [1, 1, -1]])

        # proper rotation matrices with theta=90
        self.rotations = []
        for perm in itertools.permutations(range(3)): # permutations of rows/columns
            for signs in itertools.product([-1, 1], repeat=3): # all possible signs for each row/column
                rot = np.zeros((3,3))
                
                for i in range(3):
                    rot[i, perm[i]] = signs[i]

                if np.linalg.det(rot) > 0: # proper rotations only
                    self.rotations.append(rot.copy())