#!/usr/bin/env bash
module load atlas/3.11.17
module load anaconda/3-2.5.0/python-3.5
source activate bsf-env

python run_csp_modeling.py $1
