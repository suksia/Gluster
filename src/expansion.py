from ase import build
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from dataset import Dataset
from configuration import Configuration
from pathlib import Path

class Expansion:
    def __init__(self, dataset: Dataset, a0: float, cutoffs: list[float]):
        self.dataset = Dataset()
        self.update_dataset(dataset)

        self.a0 = a0
        self.cutoffs = cutoffs

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

        self.prim = build.bulk('W', 'bcc', a0)
        self.cluster_space = ClusterSpace(
            structure=self.prim, 
            cutoffs=cutoffs, 
            chemical_symbols=[em for em in self.dataset.end_members.keys()])

    def init_structure_container(self):
        if self.cluster_space is None:
            raise RuntimeError(f"Tried to initialize structure container, but cluster space does not exist")
        self.structure_container = StructureContainer(cluster_space=self.cluster_space)

        self.dataset.compute_mixing_energies()
        for conf in self.dataset.configs.values():
            self.structure_container.add_structure(
                structure=conf.Atoms, 
                user_tag=conf.name, 
                properties={'mixing_energy': conf.mixing_energy})

    def fit(self, write_path: Path):
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
        
        self.cluster_expansion.write(write_path)