#!/bin/bash
#SBATCH --account=mzu-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g

python3 workflow_driver.py --output 2025-03-28_output_percentile95_uethpt --parsl_rundir 2025-03-28_rundir_kde --config config_kde.yaml
#python3 workflow_driver.py --output 2025-03-03_output_max_auroc --parsl_rundir 2025-03-03_rundir_kde --config config_kde.yaml
