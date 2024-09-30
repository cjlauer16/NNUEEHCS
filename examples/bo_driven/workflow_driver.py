import click
import os
import parsl
import yaml
from itertools import product
from parsl.app.app import python_app
from parsl.app.app import join_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher
from parsl.data_provider.files import File as ParslFile
from parsl import set_stream_logger
from parsl.launchers import SingleNodeLauncher
from parsl.executors import HighThroughputExecutor

def get_config(config_filename):
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    return config


@bash_app(cache=True)
def run_bo(benchmark, uq_method, dataset, output,
           stdout=parsl.AUTO_LOGNAME,
           stderr=parsl.AUTO_LOGNAME):
    import sh
    import os
    try:
        os.unsetenv('SLURM_CPU_BIND')
        os.unsetenv('SLURM_CPU_BIND_LIST')
        os.unsetenv('SLURM_CPUS_ON_NODE')
        os.unsetenv('SLURM_CPUS_PER_TASK')
        os.unsetenv('SLURM_CPU_BIND_TYPE')
        os.unsetenv('SLURM_JOB_NAME')
    except KeyError:
        pass
    python = sh.Command('python3')
    command = python.bake('bo.py', '--benchmark', benchmark, 
                          '--uq_method', uq_method, 
                          '--dataset', dataset, 
                          '--output', output,
                          '--restart'
                          )
    print(str(command))
    return str(command)

@click.command()
@click.option('--config', default='./config.yaml', help='Path to the config file', required=False)
@click.option('--output', default='workflow_output', help='Path to the output directory.', required=False)
@click.option('--parsl_rundir', default='./rundir', help='Path to the parsl run directory', required=False)
def main(config, output, parsl_rundir):

    config_filename = config
    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )

    slurm_provider = SlurmProvider(
        partition="gpuA40x4",
        account="mzu-delta-gpu",
        scheduler_options="#SBATCH --gpus-per-task=1 --cpus-per-gpu=16 --nodes=1 --ntasks-per-node=1 --nodes=1",
        worker_init='source ~/activate.sh',
        nodes_per_block=1,
        max_blocks=3,
        init_blocks=1,
        parallelism=1,
        exclusive=False,
        mem_per_node=64,
        walltime="8:55:00",
        cmd_timeout=500,
        launcher=SingleNodeLauncher()
    )

    parsl_config = Config(
        retries=5,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=16,
                available_accelerators=1,
                cpu_affinity='block',
                mem_per_worker=64,
                worker_debug=False,
                label="BO_Search_Exec",
                provider=slurm_provider
            )
        ]
    )
    parsl.load(parsl_config)

    config = get_config(config_filename)
    benches = config['benchmarks'].keys()
    uq_methods = config['uq_methods'].keys()
    dsets = ['tails', 'gaps']

    total = list(product(benches, uq_methods, dsets))

    # This causes issues when submitting one job from another, see:
    # https://bugs.schedmd.com/show_bug.cgi?id=14298
    try:
        os.unsetenv('SLURM_CPU_BIND')
        os.unsetenv('SLURM_CPU_BIND_LIST')
        os.unsetenv('SLURM_CPUS_ON_NODE')
        os.unsetenv('SLURM_CPUS_PER_TASK')
        os.unsetenv('SLURM_CPU_BIND_TYPE')
        os.unsetenv('SLURM_JOB_NAME')
    except KeyError:
        pass

    results = list()
    for bench, uq_method, dset in total:
        print(f'Running {bench} with {uq_method} on {dset}')
        res = run_bo(bench, uq_method, dset, output)
        results.append(res)

    for res in results:
        print(res.result())


if __name__ == '__main__':
    main()
