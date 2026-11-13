from ase import build
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from copy import copy
import warnings, logging

from dataset import Dataset

logger = logging.getLogger('Gluster')


class Expansion:
    def __init__(self, dataset: Dataset, cutoffs: list[float], a0: float):
        self.dataset = dataset

        self.cutoffs = cutoffs
        self.a0 = a0

        self.cluster_space: ClusterSpace = None
        self.structure_container: StructureContainer = None
        self.cluster_expansion: ClusterExpansion = None
        self.optimizer: CrossValidationEstimator = None

        self.rmse_train, self.rmse_val = None, None # root mean-squared error
        self.bic = None  # bayesian information criterion
        self.num_params, self.num_nonzero_params = None, None

    def init_cluster_space(self):
        self.prim = build.bulk('W', 'bcc', self.a0)
        self.cluster_space = ClusterSpace(
            structure=self.prim, 
            cutoffs=self.cutoffs, 
            chemical_symbols=list(self.dataset.end_members))

    def init_structure_container(self):
        if self.cluster_space is None:
            raise RuntimeError(f"Tried to initialize structure container, but cluster space does not exist")
        self.structure_container = StructureContainer(cluster_space=self.cluster_space)

        target_vol = self.a0**3 / 2

        for conf in self.dataset.configurations:
            conf_vol = conf.Atoms.get_volume() / len(conf.Atoms)

            if conf_vol != target_vol:
                vol_scale_factor = (target_vol / conf_vol)**(1/3)
                conf.Atoms.set_cell(conf.Atoms.cell*vol_scale_factor, scale_atoms=True)

            self.structure_container.add_structure(
                structure=conf.Atoms, 
                user_tag=f"config {conf.cid}", 
                properties={'mixing_energy': conf.mixing_energy})

    def fit(self):
        if self.structure_container is None:
            raise RuntimeError(f"Tried to fit cluster expansion, but structure container does not exist")
        
        # fit cluster expansion
        self.optimizer = CrossValidationEstimator(
            fit_data = self.structure_container.get_fit_data(key='mixing_energy'), 
            fit_method='ardr')
        self.optimizer.validate()
        self.optimizer.train()

        self.rmse_train = self.optimizer.rmse_train
        self.rmse_val = self.optimizer.rmse_validation
        self.bic = self.optimizer.BIC
        self.num_params = self.optimizer.n_parameters
        self.num_nonzero_params = self.optimizer.n_nonzero_parameters

        self.cluster_expansion = ClusterExpansion(
            cluster_space=self.cluster_space, 
            parameters=self.optimizer.parameters, 
            metadata=self.optimizer.summary)

    def write(self, write_dir: Path):
        with open(write_dir / 'cluster_space.out', 'w') as f:
            f.write(str(self.cluster_space))

        with open(write_dir / 'structure_container.out', 'w') as f:
            f.write(str(self.structure_container))

        with open(write_dir / 'cluster_expansion.out', 'w') as f:
            f.write(str(self.cluster_expansion))

        self.cluster_expansion.write(write_dir / 'cluster_expansion.ce')
        

def scan_cutoffs(order: int, cutoff_values: list[float], expansion: Expansion, write_dir: Path):
    if len(expansion.cutoffs) != order-2:
        raise ValueError(f"Need {order-2} cutoffs for lower order clusters in order to scan the cutoff for order {order}")

    dataset, cutoffs, a0 = expansion.dataset, expansion.cutoffs, expansion.a0

    scan_data = {'cutoff': [], 'rmse_train': [], 'rmse_val': [], 'bic': [], 'num_params': [], 'num_nonzero_params': []}

    for co in cutoff_values:
        current_expansion = Expansion(dataset, copy(cutoffs), a0)
        current_expansion.cutoffs.append(co)

        current_expansion.init_cluster_space()
        current_expansion.init_structure_container()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            current_expansion.fit()

            for warning in caught:
                if issubclass(warning.category, UserWarning):
                    if 'Warning: Condition number is large' in str(warning.message):
                        logger.debug(f"Linear problem is ill-conditioned for order {order} and Cutoff value {co:5.3f}")
                        break

        scan_data['cutoff'].append(co)
        scan_data['rmse_train'].append(1000*current_expansion.rmse_train)
        scan_data['rmse_val'].append(1000*current_expansion.rmse_val)
        scan_data['bic'].append(0.001*current_expansion.bic)
        scan_data['num_params'].append(current_expansion.num_params)
        scan_data['num_nonzero_params'].append(current_expansion.num_nonzero_params)

    with open(write_dir / f'scan_{order}.out', 'w') as f:
        f.write(f"Cutoffs = {expansion.cutoffs}\n")
        f.write(f"Order = {order}\n\n")

        f.write(f"{'Cutoff':<25} {'RMSE Train [meV/atom]':<25} {'RMSE Val [meV/atom]':<25} {'BIC (x1e+3)':<25} {'# Params':<25} {'# Nonzero Params':<25}\n\n")
        for i, co in enumerate(cutoff_values):
            f.write(f"{co:<25.3f} {scan_data['rmse_train'][i]:<25.5f} {scan_data['rmse_val'][i]:<25.5f} {scan_data['bic'][i]:<25.5f} {scan_data['num_params'][i]:<25} {scan_data['num_nonzero_params'][i]:<25}\n")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes: list[Axes] = axes

    ax = axes[0]
    ax.plot(cutoff_values, scan_data['rmse_train'], '-o', label='Train')
    ax.plot(cutoff_values, scan_data['rmse_val'], '-s', label='Validation')
    ax.set_ylabel(r'RMSE [meV/atom]')
    ax.legend()

    ax = axes[1]
    ax.plot(cutoff_values, scan_data['bic'], '-o')
    ax.set_ylabel(r'BIC ($\times 10^3)$')

    ax = axes[2]
    ax.plot(cutoff_values, scan_data['num_params'], '-o', label='Total')
    ax.plot(cutoff_values, scan_data['num_nonzero_params'], '-s', label='Nonzero')
    ax.set_xlabel(f'Order {order} cutoff [Å]')
    ax.set_ylabel('Number of parameters')
    ax.legend()
    
    fig.align_ylabels(axes)

    fig.savefig(write_dir / f'scan_{order}.png', bbox_inches='tight')