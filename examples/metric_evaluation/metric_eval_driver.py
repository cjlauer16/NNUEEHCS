import click
import os
import parsl
import yaml
from itertools import product
from parsl.app.app import python_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SingleNodeLauncher
from parsl.data_provider.files import File as ParslFile
from parsl.executors import HighThroughputExecutor

def get_config(config_filename):
    with open(config_filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

@bash_app(cache=True)
def run_evaluate_metrics(config, benchmark, uq_method, dataset, results_dir, output_file,
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
    command = python.bake('evaluate_metrics.py', 
                          '--config_file', config,
                          '--benchmark', benchmark, 
                          '--dataset', dataset,
                          '--method', uq_method,
                          '--results_dir', results_dir,
                          '--output', output_file)
    print(str(command))
    return str(command)

@python_app
def combine_results(result_files, output_file):
    import pandas as pd
    import os
    
    combined_df = pd.DataFrame()
    for file in result_files:
        try:
            if os.path.exists(file):
                df = pd.read_csv(file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                # Create a record for failed job
                benchmark, dataset, method = os.path.basename(file).replace('.csv', '').split('_')
                failed_row = pd.DataFrame({
                    'benchmark': [benchmark],
                    'dataset': [dataset],
                    'method': [method],
                    'trial': ['N/A'],
                    'metric': ['FAILED'],
                    'objective': ['FAILED'],
                    'value': [float('nan')]
                })
                combined_df = pd.concat([combined_df, failed_row], ignore_index=True)
        except Exception as e:
            # Handle any other errors during processing
            benchmark, dataset, method = os.path.basename(file).replace('.csv', '').split('_')
            failed_row = pd.DataFrame({
                'benchmark': [benchmark],
                'dataset': [dataset],
                'method': [method],
                'trial': ['N/A'],
                'metric': ['ERROR'],
                'objective': [str(e)[:100]],  # Truncate error message if too long
                'value': [float('nan')]
            })
            combined_df = pd.concat([combined_df, failed_row], ignore_index=True)
    
    combined_df.to_csv(output_file, index=False)
    return output_file

@click.command()
@click.option('--config', default='./config.yaml', help='Path to the config file', required=False)
@click.option('--output', default='workflow_output', help='Path to the output directory.', required=False)
@click.option('--parsl_rundir', default='./rundir', help='Path to the parsl run directory', required=False)
@click.option('--results_dir', default=None, help='Path to the results directory for evaluation', required=False)
@click.option('--max_tasks', default=None, type=int, help='Maximum number of tasks to run (for testing)', required=False)
@click.option('--local', is_flag=True, help='Run tasks locally instead of on the cluster', required=False)
@click.option('--skip-completed', is_flag=True, help='Skip tasks if their output file already exists.', required=False)
def main(config, output, parsl_rundir, results_dir, max_tasks, local, skip_completed):

    config_filename = config
    config_data = get_config(config_filename)
    slurm_settings = config_data.get('metric_eval_slurm_config', {})

    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )

    # Construct scheduler options string dynamically and add to settings
    gpus_per_task = slurm_settings.pop('gpus_per_task', 1) # Remove key after getting value
    cpus_per_gpu = slurm_settings.pop('cpus_per_gpu', 16)
    nodes = slurm_settings.pop('nodes', 1)
    ntasks_per_node = slurm_settings.pop('ntasks_per_node', 1)
    scheduler_opts = (
        f"--gpus-per-task={gpus_per_task} "
        f"--cpus-per-gpu={cpus_per_gpu} "
        f"--nodes={nodes} "
        f"--ntasks-per-node={ntasks_per_node}"
    )
    slurm_settings['scheduler_options'] = f"#SBATCH {scheduler_opts}"

    slurm_provider = SlurmProvider(
        **slurm_settings,
        launcher=SingleNodeLauncher()
    )

    provider = local_provider if local else slurm_provider
    
    parsl_config = Config(
        retries=20,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=16,
                available_accelerators=1,
                cpu_affinity='block',
                mem_per_worker=64,
                worker_debug=False,
                label="Metric_Eval_Exec",
                provider=provider
            )
        ]
    )
    parsl.load(parsl_config)

    benches = config_data['benchmarks'].keys()
    uq_methods = config_data['uq_methods'].keys()
    dsets = ['tails', 'gaps']

    total = list(product(benches, uq_methods, dsets))
    print(f"Total number of tasks: {len(total)}")
    print(f"Tasks are: {total}")
    
    # Limit the number of tasks if max_tasks is specified
    if max_tasks is not None:
        print(f"Limiting to {max_tasks} tasks for testing")
        total = total[:max_tasks]

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

    # Run metric evaluation tasks
    if results_dir is None:
        results_dir = output
    
    os.makedirs(f"{output}/metric_eval", exist_ok=True)
    eval_results = list()
    result_files = list()
    
    for bench, uq_method, dset in total:
        output_file = f"{output}/metric_eval/{bench}_{dset}_{uq_method}.csv"
        result_files.append(output_file)

        if skip_completed and os.path.exists(output_file):
            print(f"Skipping completed task: {bench} with {uq_method} on {dset} (output file exists: {output_file})")
            continue 

        print(f'Running metric evaluation for {bench} with {uq_method} on {dset}')
        res = run_evaluate_metrics(config_filename, bench, uq_method, dset, 
                                   results_dir, output_file)
        eval_results.append(res)
    
    # Wait for all submitted evaluation tasks to complete
    print(f"Waiting for {len(eval_results)} submitted tasks to complete...")
    for res in eval_results:
        try:
            print(res.result())
        except Exception as e:
            print(f"Task failed with error: {e}")
    
    # Combine all results into a single CSV file
    combined_output = f"{output}/all_metrics_evaluated.csv"
    combine_task = combine_results(result_files, combined_output)
    print(f"Combined results saved to: {combine_task.result()}")


if __name__ == '__main__':
    main()
