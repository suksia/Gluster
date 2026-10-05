from ase import build
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from dataset import Dataset
from configuration import Configuration
from pathlib import Path
import logging, warnings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('Gluster')

class Expansion:
    def __init__(self, dataset: Dataset, a0: float, cutoffs: list[float], output_dir: Path):
        self.dataset = Dataset()
        self.update_dataset(dataset)

        self.a0 = a0
        self.cutoffs = cutoffs
        self.output_dir = output_dir

        self.cluster_space: ClusterSpace = None
        self.structure_container: StructureContainer = None
        self.cluster_expansion: ClusterExpansion = None
        self.optimizer: CrossValidationEstimator = None

    def update_dataset(self, dataset: Dataset):
        # check if config should be included
        new_configs: dict[int, Configuration] = {}

        for cid, conf in dataset.configs.items():
            if cid not in self.dataset.configs.keys():
                conf.load_energy()
                if all([conf.energy, conf.Atoms, conf.include_fit]):
                    new_configs[cid] = conf

        self.dataset.update(new_configs)

    def init_cluster_space(self, a0=None, cutoffs=None):
        if a0:
            self.a0 = a0
        if cutoffs:
            self.cutoffs = cutoffs

        self.dataset.compute_mixing_energies()

        self.prim = build.bulk('W', 'bcc', self.a0)
        self.cluster_space = ClusterSpace(
            structure=self.prim, 
            cutoffs=self.cutoffs, 
            chemical_symbols=[em for em in self.dataset.end_members.keys()])

        with open(self.output_dir / 'cluster_space.out', 'w') as f:
            print(self.cluster_space, file=f)

        logger.debug(f"Initialized cluster space")

    def init_structure_container(self):
        if self.cluster_space is None:
            raise RuntimeError(f"Tried to initialize structure container, but cluster space does not exist")
        self.structure_container = StructureContainer(cluster_space=self.cluster_space)

        for conf in self.dataset.configs.values():
            vol_scale_factor = (self.a0**3 / conf.a0**3)**(1/3)
            conf.Atoms.set_cell(conf.Atoms.cell*vol_scale_factor, scale_atoms=True)

            self.structure_container.add_structure(
                structure=conf.Atoms, 
                user_tag=str(conf.name), 
                properties={'mixing_energy': conf.mixing_energy})

        with open(self.output_dir / 'structure_container.out', 'w') as f:
            print(self.structure_container, file=f)

        logger.debug(f"Initialized structure container")

    def fit(self):
        if self.structure_container is None:
            raise RuntimeError(f"Tried to fit cluster expansion, but structure container does not exist")
        
        # fit cluster expansion
        self.optimizer = CrossValidationEstimator(
            fit_data= self.structure_container.get_fit_data(key='mixing_energy'), 
            fit_method='ardr')
        self.optimizer.validate()
        self.optimizer.train()

        self.cluster_expansion = ClusterExpansion(
            cluster_space=self.cluster_space, 
            parameters=self.optimizer.parameters, 
            metadata=self.optimizer.summary)
        
        with open(self.output_dir / 'cross_validation.out', 'w') as f:
            print(self.optimizer, file=f)

        with open(self.output_dir / 'expansion.out', 'w') as f:
            print(self.cluster_expansion, file=f)

        self.cluster_expansion.write(self.output_dir / 'expansion.ce')
        logger.debug(f"Initialized and fit cluster expansion")