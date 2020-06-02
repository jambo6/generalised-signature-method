"""
This file contains basic variables and definitions that we wish to make easily accessible for any script that requires
it.

from definitions import *
"""
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[0])
DATA_DIR = ROOT_DIR + '/data'
MODELS_DIR = ROOT_DIR + '/models'
RESULTS_DIR = ROOT_DIR + '/experiments/results'

# Havok data location
if ROOT_DIR == '/home/morrill/Documents/signature-best-practices':
    DATA_DIR = '/scratch/morrill/signature-best-practices/data'
    MODELS_DIR = ROOT_DIR + '/models/havok_models'

# Save aws models in a different location so we can pull easily
if ROOT_DIR == '/home/ubuntu/signature-best-practices':
    DATA_DIR = '/run/user/1000/data_store'
    MODELS_DIR = ROOT_DIR + '/models/aws_models'

# Packages/functions used everywhere
from src.omni.decorators import *
from src.omni.functions import *
from src.omni.base import *


