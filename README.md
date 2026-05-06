**OFFLINE VERSION OF DR-POGO USED FOR THE PAPER, NOT MAINTAINED OR TESTED SINCE**


# DRPoGO: Direct Radar Map Alighnment for Loop Closure in Pose Graph Optimization
This repository provides the codebase for radar-based coarse registration and evaluation against ground truth. The pipeline includes local map generation using DRO, loop closure proposal using RaPlace, and pose graph optimization.

## Dependencies

All dependencies should be in `requirements.txt`. Please install them in you virtual environment with
```
pip install -r requirements.txt
```

For __pogo__ you will need Eigen3 and Ceres Solver:
```bash
sudo apt-get install python3-tk
sudo apt-get install python3-dev
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libeigen3-dev
sudo apt-get install libceres-dev
sudo apt-get install libyaml-cpp-dev
```

## Compile (pogo part)

To compile pogo:
```bash
cd pogo
mkdir -p build
cd build
cmake ..
make -j8
cd ../..
```

## Run the full pipeline

You need to run the following steps in order:

1. Run odometry (DRO)
2. Run the loop closure detection (RaPlace)
3. Run the coarse registration (Coarse Registration)
4. Run the pose graph optimization (PoGO)

### Run DRO
First, download data from the Boreas dataset [here](https://www.boreas.utias.utoronto.ca/#/download). DRO generates local maps that accounts for motion distortion of the radar scans.

Then copy the example config file `DRO/config_example.yaml` to `DRO/config.yaml` and modify the parameters as needed, especially the `data_path` as follows.
```yaml
  data:
    data_path: /absolute/path/to/Boreas/<sequence>
```

In the root of the repository, run the following command to generate local maps:
```bash
python dro/radar_gp_state_estimation.py
```

It will output the local maps in the `output/<SEQ-NAME>/local_maps` folder, each names with the radar scan's first timestamp. 

### Run RaPlace
Simply run RaPlace as follows in the root of the repository (all the paths should be autonomatically using what was specified in the DRO config file):
```bash
python raplace/raplace.py
```

It will generate a CSV of proposed scan pairs in the `output/<SEQ-NAME>/raplace_loops.csv` folder.
Each row contains the following columns:
- `time_i`: Timestamp of the first scan in the pair \[s\].
- `time_j`: Timestamp of the second scan in the pair \[s\].
- `scan_i_name`: Name of the first scan in the pair.
- `scan_j_name`: Name of the second scan in the pair.
- `score`: The score of the proposed loop closure as defined in RaPlace (not used)
- `min_dist`: The minimum dist between the scores as defined in RaPlace (not used)

### Run the feature-based coarse registration

In the root of the repository, run the following command:
```bash
python coarse_registration/coarse_registration.py
```

It will generate a CSV of coarse registration results in the `output/<SEQ-NAME>/coarse_registration.csv` folder.
Each row contains the following columns:
- `scan_i_name`: Name of the first scan in the pair.
- `scan_j_name`: Name of the second scan in the pair.
- `x`, `y`, `theta`: The estimated transformation from scan i to scan j.

### Run PoGO

You can modify the configuration file `pogo/config.yaml` to adjust the parameters for the pose graph optimization.
To run, in the root of the repository, run the following command:
```bash
pogo/build/pogo
```

### For paper and evaluation

To plot the trajectory estimates and get metrics:
```bash
python script_for_paper/eval.py
```
This will diplay the ATE errors and write to file different trajectory estimates (aligned with the 1st pose in `<SEQ-NAME>_trajectories.pdf` or fully aligned in `<SEQ-NAME>_XXXX_aligned.pdf`)

## Release TODOs


- [ ] Create a script to get the pogo parameters automatically from a calib sequence
- [ ] Check the "save local maps" option in DRO (seems to not be as wanted)
