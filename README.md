_Gluster_ is Pythonic glue code connecting VASP and the [icet](https://icet.materialsmodeling.org/index.html) cluster expansion package. The One Input File (OIF) philosophy is a core tenet, which dictates that the execution of a computational task (e.g., fitting a cluster expansion) should be completely described by a single user-created input file called "the OIF."

# Stage 1: Generating Configurations

Configurations with arbitrary order can be described by a _chemical basis_, which forms a superlattice of the underlying lattice (i.e., bcc, fcc, hcp). It contains $Q$ lattice points along with a mapping between said lattice points and a set of occupancy probabilities $\{p_{\sigma_i}\}$, where $\sigma_i=0,1,...,M$ are the allowed species on site $i$ and $M$ is the alloy order (binary, ternary, ...).

A chemical basis $\mathbf{A}'$ is constructed by applying a transformation matrix $\mathbf{P}$ in Hermite normal form (HNF) to the primitive basis $\mathbf{A}$. HNF requires $$\mathbf{P} = \begin{pmatrix} a & 0 & 0 \\ b & c & 0 \\ d & f & g \end{pmatrix}, \qquad 0\leq b < c, \qquad 0 \leq d,\,f < g,$$ using the row-convention form. Column-convention form would be written in the same way as $\mathbf{P}^{\mathsf{T}}$ and produces a different basis. If $\mathbf{A}$ is written with the lattice vectors as column vectors, $$\mathbf{A}' = \mathbf{A}\mathbf{P}.$$ Since $Q=|\det\mathbf{P}|$, the chemical basis usually contains additional lattice points. These points can be determined algorithmically by enumerating a sufficient set of integer vectors $\mathbf{n}=(n_x, n_y, n_z)$ with $n_x,n_y,n_z=0,1,2,\dots$ and computing $$\mathbf{f}_j = \mathbf{P}^{-1}\mathbf{n}_j \mod 1,$$ where the modulo operator is applied element-wise. The vectors $\mathbf{f}_j$ are the fractional coordinates of all possible lattice points $\mathbf{A}'$ which have been reduced to within the first unit cell of this new basis. All unique vectors in this list form the basis points of $\mathbf{A}'$ represented as $\mathbf{b}_\alpha$. 

### Specifying a Basis

Often $\mathbf{b}_\alpha$ is not known ahead of time, so they are always printed to file named `basis.out` along with the transformed lattice vectors $\mathbf{A}'/a_0$. For each basis point, the chemical mapping must be provided as a dictionary of occupancy probabilities or as species name. A minimal input file is shown below, which contains two configurations: one is a disordered $\text{W}_{50} \text{V}_{50}$ alloy, while the other is an ordered version. 

In configuration (0), the primitive lattice vectors are used and every lattice point in the conventional $4\times 4\times 4$ supercell (128 atoms) is randomly decorated with either $\text{W}$ or $\text{V}$. In configuration (1), a 2-atom basis with a fixed ordering is used. Since (1) is fully ordered, the composition can be determined directly from the basis occupancy and therefore does not need to be provided. However, (0) is disordered and the composition would vary randomly if the species were sampled without replacement. The composition, expressed in terms of the number of each species in the conventional supercell, provides counters which are decremented during decoration, thus fixing it.

```YAML
name: example_01
dir: <working directory path>

configurations:
    0: 
        size: [4, 4, 4]
        transform: [[1, 0, 0], 
                    [0, 1, 0], 
                    [0, 0, 1]]
        basis:
            "0, 0, 0": {W: 0.5, V: 0.5}
        comp: {W: 64, V: 64}
    1: 
        size: [3, 3, 3]
        transform: [[1, 0, 0], 
                    [1, 2, 0], 
                    [0, 0, 1]]
        basis:
            "0, 0, 0": W
            "0, 0.5, 0": {V: 1.0}
```

At the very least, a configuration <u>must</u> have a transformation matrix given by the `transform` kwarg. The basis positions can then be copied from `basis.out` and given chemical mappings. If a configuration is not fully defined (i.e., necessary kwargs are missing), _Gluster_ does not attempt to run VASP  on this configuration or include it in the cluster expansion fitting dataset. To prevent _Gluster_ from running VASP and fitting a cluster expansion entirely, the `check-basis` flag can be set.
```bash
python Gluster/src/gluster.py in.yml --check-basis
```
If a configuration is fully defined, _Gluster_ will write an XYZ file for the bases and decorated conventional supercells in the `xyz/` sub-directory. Since a disordered configuration may change between runs, a `POSCAR` can be provided using the `poscar` kwarg to lock-in a specific arrangement, though the other kwargs used to generate it should still be included as it fingerprints the configuration.

### OIF Template

```YAML
name: <name of directory>
dir: <working directory path>
lattice: <bcc, fcc, hcp; ONLY bcc is IMPLEMENTED>
potential: <path to directory containing POTCAR files>

configurations:
    <id>:
        run: <boolean value forcing VASP files to be regenerated and VASP to be run (see section 2); default = True unless POSCAR and energy.out exist>
        size: <list of three positive integers [Nx, Ny, Nz] for generating the conventional supercell> 
        transform: <3x3 matrix in Hermite normal form used to transform the primitive lattice vectors>
        basis: 
            <fractional coordinates of basis point>: <dictionary mapping species to occupancy probabilities, or a species symbol>
            ...
        comp: <dictionary mapping species to percentage or number of atoms in the conventional supercell>
        ncores: <number of cores (MPI ranks) to launch VASP with for this configuration>
        incar: |
            <INCAR lines for this configuration>
        poscar: |
            <POSCAR lines for this configuration>
        kpoints: |
            <KPOINTS lines for this configuration>
        potcar: <list of directory names (e.g., [W_sv, Mo, V_pv]) to load each POTCAR file from>
    ...
```

# Stage 2: Running VASP