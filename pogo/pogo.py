import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import utils
import yaml


def main(seq_path):
    # Read the pogo/config.yaml file
    config = yaml.safe_load(open('pogo/config.yaml', 'r'))

    if seq_path.endswith('/'):
        seq_path = seq_path[:-1]  # Remove trailing slash if present

    config['seq_id'] = seq_path.split('/')[-1]  # Extract the sequence ID from the path

    # Write the config to a file
    with open('pogo/config_temp.yaml', 'w') as f:
        yaml.dump(config, f)
    print("Wrote the config to pogo/config_temp.yaml")

    # Run the PoGO executable
    os.system('./pogo/build/pogo')





if __name__ == "__main__":
    if utils.isMultiSequence():
        seq_list = utils.getOutputDataDirs()
    else:
        seq_list = [utils.getOutputDataDir()]
    for output_path in seq_list:
        main(output_path)