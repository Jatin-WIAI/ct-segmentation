"""Constants related to the repository
"""
from os.path import dirname, join, realpath

# Path to the ultrasound repository. Go up 3 directories: ROOT_DIR/src/utils/constants.py
REPO_PATH = dirname(dirname(dirname(realpath(__file__))))

CONFIG_DIR = join(REPO_PATH, "config")

DATA_ROOT_DIR = "/scratche/users/sansiddh/abdomenCT-1k/"