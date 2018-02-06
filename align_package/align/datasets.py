import os, sys

# get the name of the package directory
module_path = os.path.dirname(os.path.abspath(__file__))

# then construct the path to the text file
CHILDES_directory = os.path.join(module_path, 'data/CHILDES')

# Run as: `from align.datasets import CHILDES_directory`
