#!/usr/bin/bash
#SBATCH -J scip
#SBATCH -N1 --ntasks-per-node=1 -c1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:L40S:2          
#SBATCH --time=8:00:00                 
#SBATCH -o scoring-%j.out
 
cd $SLURM_SUBMIT_DIR

module load anaconda3/2022.05

conda activate model-unlearning

echo "Python path"

which python

echo "Running python file"

python evals/scoring.py

echo "Done running python file"

conda deactivate



