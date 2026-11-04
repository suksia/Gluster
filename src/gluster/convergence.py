import argparse, logging, sys, yaml
from pathlib import Path
from copy import copy
import matplotlib.pyplot as plt

from calculation import Calculation, run_vasp_in_parallel

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger('Gluster')


######################## input ########################

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Path to input file')
args = parser.parse_args()

input_fp = Path(args.input).resolve()

if not input_fp.exists():
    raise FileNotFoundError(f"[{input_fp}] File could not be found")

with open(input_fp, 'r') as f:
    input_yml: dict = yaml.safe_load(f)

with open(input_fp, 'r') as f:
    input_yml_lines = f.readlines()

logger.debug(f'Loaded input file {input_fp}')


################## project directory ##################

TOP_DIR: Path = Path(input_yml['dir']) / input_yml['name']

if TOP_DIR.parent.exists() is False:
    raise ValueError(f'Directory {TOP_DIR.parent} does not exist')

TOP_DIR.mkdir(exist_ok=True)

with open(TOP_DIR / 'last_input.yaml', 'w') as f:
    f.writelines(input_yml_lines)

logger.debug(f'Created benchmark directory at {TOP_DIR.as_posix()}')


####################### run vasp ######################

vasp_file_lines = {
    'incar': input_yml['incar'],
    'kpoints': input_yml['kpoints'],
    'poscar': input_yml['poscar'],
    'potcar': input_yml['potcar'],
}

if 'convergence' not in input_yml.keys():
    raise KeyError('Missing convergence section in input file')

converge: dict = input_yml['convergence']
studies: dict[str, dict] = {k: None for k in ['kpoints', 'encut', 'ediff', 'ediffg']}
cid = 0

for setting in studies.keys():
    if setting in converge.keys():
        setting_dict: dict = converge[setting]
        calcs: list[Calculation] = []
        values = []

        for vi, value in enumerate(setting_dict[setting]):
            if setting == 'kpoints':
                setting_str = f"{value[0]}x{value[1]}x{value[1]}"
            elif setting == 'ediffg':
                setting_str = 'neg' + str(value)[1:]
            else:
                setting_str = str(value)

            cdir = TOP_DIR / setting / setting_str 
            if 'ntasks' in setting_dict.keys():
                if type(setting_dict['ntasks']) == list:
                    ntasks = setting_dict['ntasks'][vi]
                elif type(setting_dict['ntasks']) == int:
                    ntasks = setting_dict['ntasks']
                else:
                    ntasks = 1

                if type(setting_dict['kpar']) == list:
                    kpar = setting_dict['kpar'][vi]
                elif type(setting_dict['kpar']) == int:
                    kpar = setting_dict['kpar']
                else:
                    kpar = 1

                if type(setting_dict['ncore']) == list:
                    ncore = setting_dict['ncore'][vi]
                elif type(setting_dict['ncore']) == int:
                    ncore = setting_dict['ncore']
                else:
                    ncore = 1

            calc = Calculation(cid, cdir, vasp_file_lines, ntasks, kpar, ncore)

            if setting == 'kpoints':
                calc.kpoints.update_mesh(value)
            else:
                calc.incar.update_setting(setting, value)

            calcs.append(calc)
            values.append(value)

            cid += 1

        studies[setting] = {'calculations': calcs, 'values': values}


for setting, setting_dict in studies.items():
    if setting_dict is None:
        continue

    logger.debug(f"Running convergence calculations for {setting.upper()}")

    queue = {calc.cid: calc for calc in setting_dict['calculations']}
    run_vasp_in_parallel(queue)


######################## output #######################

convergence_path = TOP_DIR / 'convergence.out'

if not convergence_path.exists():
    convergence_file = open(convergence_path, 'w')

    convergence_file.write(input_yml['name'] + '\n\n')
    convergence_file.write(f"{'ID':<10} {'KPOINTS':<10} {'ENCUT':<10} {'EDIFF':<10} {'EDIFFG':<10} {'energy [eV]':<10} {'time [sec]':<10}\n")
else:
    convergence_file = open(convergence_path, 'a')
    convergence_file.write('\n')

for setting, setting_dict in studies.items():
    values, energies, times = [], [], []

    for calc, val in zip(setting_dict['calculations'], setting_dict['values']):
        calc: Calculation = calc

        if setting == 'kpoints':
            val = f"{val[0]}x{val[1]}x{val[1]}"

        values.append(val)
        energies.append(calc.energy)
        times.append(calc.elapsed_time)

        if setting == 'kpoints':
            f.write(f"{calc.cid:<10} {val:<10} {' ':10} {' ':<10} {' ':<10} {calc.energy:<10.2f} {calc.elapsed_time:<10.2f}\n")
        elif setting == 'encut':
            f.write(f"{calc.cid:<10} {' ':<10} {val:<10} {' ':<10} {' ':<10} {calc.energy:<10.2f} {calc.elapsed_time:<10.2f}\n")
        elif setting == 'ediff':
            f.write(f"{calc.cid:<10} {' ':<10} {' ':10} {val:<10.2f} {' ':<10} {calc.energy:<10.2f} {calc.elapsed_time:<10.2f}\n")
        elif setting == 'ediffg':
            f.write(f"{calc.cid:<10} {' ':<10} {' ':10} {' ':<10} {val:<10.2f} {calc.energy:<10.2f} {calc.elapsed_time:<10.2f}\n")

    fig = plt.figure()

    plt.plot(values, energies)
    ax2 = plt.twinx()
    ax2.plot(values, times)

    plt.xlabel(setting.upper())
    plt.ylabel('Energy [eV]')
    ax2.set_ylabel('Elapsed time [sec]')

    plt.title(f"{setting.upper()} Convergence Study")

    fig.savefig(TOP_DIR / f"{setting}.png", bbox_inches='tight')