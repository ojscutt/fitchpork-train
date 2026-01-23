#! /usr/bin/env bash
#SBATCH --nodes 1
#SBATCH --ntasks 18
#SBATCH --qos=bbgpu
#SBATCH --account=daviesgr-pcann
#SBATCH --time 20:00:00
#SBATCH --gres=gpu:a100_80:1
#SBATCH --output=/rds/projects/d/daviesgr-pcann/repos_data/ojscutt/fitchpork/slurm/slurm-%A.out
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

set -e

module purge
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

export VENV_BASE="py311tf215"

export VENV_DIR="/rds/projects/d/daviesgr-pcann/venvs/${VENV_BASE}"
export VENV_PATH="${VENV_DIR}/${VENV_BASE}-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    python3 -m venv --system-site-packages ${VENV_PATH}
fi

# Activate the virtual environment
source ${VENV_PATH}/bin/activate

# Store pip cache in /scratch directory, instead of the default home directory location
PIP_CACHE_DIR="/scratch/${USER}/pip"

# Perform any required pip installations
pip install "tensorflow[and-cuda]>=2.15.0,<2.16"
pip install pandas
pip install tables

# Run scripts
python -u fitchpork-pca.py