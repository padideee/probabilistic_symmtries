#!/bin/bash
ROOT=${1:-$SLURM_TMPDIR}

if [ -d "$ROOT/venv" ]; then
    echo "There is already a virtual environment existing at $ROOT/venv. Exiting."
    exit 1
fi

# Load modules
module load python/3.10
module load libffi OpenSSL

# Create the virtual environment
python -m venv $ROOT/venv

# Export Mujoco GL environment variable when activating the environment
echo 'export MUJOCO_GL="egl"' >> $ROOT/venv/bin/activate

# Activate the virtual environment
source $ROOT/venv/bin/activate

# Install the requirements
pip install --upgrade pip
pip install -r requirements.txt

# Install rotate_env
rm -rf $SLURM_TMPDIR/symmetry_RL  # Remove folder if it already exists
git clone https://github.com/sahandrez/symmetry_RL.git $SLURM_TMPDIR/symmetry_RL
# Remove the dependencies in setup.py to avoid installation errors
sed -i "/install_requires=\[/,/]/c\install_requires=[]," $SLURM_TMPDIR/symmetry_RL/setup.py
pip install $SLURM_TMPDIR/symmetry_RL/
