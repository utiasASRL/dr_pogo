# Dr-PoGO: Direct Radar Pose-Graph Optimization

Dr-PoGO is a radar-based SLAM framework.
It combines **Direct Radar Odometry (DRO)** with **RaPlace** loop-closure detection and a **pose-graph optimizer** to produce globally consistent trajectories from FMCW radar data.

This branch contains a ROS 2 implementation of the full Dr-PoGO pipeline for online operation and real-time visualization in RViz2.
The branch `offline` contains the standalone version of the pipeline for offline processing as used in the original paper.

If you find this code useful, please consider citing our paper:

**Dr-PoGO**
```
@inproceedings{legentil2026drpogo,
  title={Dr-PoGO: Direct Radar Pose-Graph Optimization},
    author={{Le Gentil}, Cedric and Weican, Li and Brizi, Leonardo and Barfoot, Timothy D.},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

**DRO**
```
@inproceedings{legentil2025dro,
  title={Dro: Doppler-aware direct radar odometry},
  author={{Le Gentil}, Cedric and Brizi, Leonardo and Lisus, Daniil and Qiao, Xinyuan and Grisetti, Giorgio and Barfoot, Timothy D.},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2025}
}
```

The loop-closure detection module is directly adapted from RaPlace's original code [here](https://github.com/hyesu-jang/RaPlace), so please also consider citing their work.



## Architecture overview


| Node | Language | Role |
|------|----------|------|
| `boreas_player` | Python | Replays a Boreas sequence (radar + IMU) as ROS 2 topics |
| `dro_node` | Python | Doppler-aware direct radar odometry |
| `raplace_node` | Python | Loop-closure detection using RaPlace |
| `registration_node` | Python | Feature-based registration and direct refinement of loop-closure transformations |
| `pogo_node` | C++ | Pose-graph optimizer (Ceres) |

## Dependencies

### ROS 2 (provided by your ROS 2 installation)
- `rclpy`, `rclcpp`
- `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `std_msgs`
- `message_filters`
- `cv_bridge`
- `yaml-cpp`

### C++ libraries (system)
- [Ceres Solver](http://ceres-solver.org/)
- [Eigen3](https://eigen.tuxfamily.org/)

These can typically be installed via your package manager (e.g., `sudo apt install libceres-dev libeigen3-dev` on Ubuntu).

### Python (pip)
Install all Python dependencies with:
```bash
pip install -r requirements.txt
```

The `requirements.txt` covers: `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `opencv-python`, `matplotlib`, `PyYAML`, `pyboreas`, `torch`, `torchvision`.

## Build

First clone this repository into your ROS 2 workspace's `src/` directory, then build with:
```bash
cd <your_ros2_workspace>/src
git clone git@github.com:utiasASRL/dr_pogo.git
```

Then build the workspace with:
```bash
cd <your_ros2_workspace>
colcon build --packages-select dr_pogo --symlink-install
source install/setup.bash
```

## Usage

### 1. Prepare the config files

Copy the config file `config/config_dro_boreas_rt.yaml` to `config/config_dro.yaml` and edit the parameters as needed.
You can also customize the RaPlace, registration, and pose-graph config files if desired (all parameters should have reasonable defaults though).

### 2. Launch the full pipeline

```bash
ros2 launch dr_pogo dr_pogo.launch.py
```

This starts all four estimation nodes plus an RViz2 visualizer with the bundled `config/rviz.rviz` preset.


### 3. Play a Boreas sequence

```bash
ros2 run dr_pogo boreas_player -p <path_to_sequence> -r <playback_rate>
# Example:
ros2 run dr_pogo boreas_player -p /data/boreas/boreas-2024-12-03-12-54 -r 1.0
```

You can also make it play as fast as DRO allows by setting `-r 0` (preventing to wait between messages if your hardware is fast enough to process the data faster than real-time, and allows for slower hardware to keep up by slowing down the playback rate as needed).

## Configuration

All YAML config files live under `config/`.

| File | Node | Key parameters |
|------|------|----------------|
| `config_dro.yaml` | `dro_node` | Sensor extrinsics (`T_axle_radar`), range limits, GP lengthscales |
| `config_raplace.yaml` | `raplace_node` | `min_time_diff`, `max_odom_drift`, `max_img_size` |
| `config_registration.yaml` | `registration_node` | `lowe_ratio`, `ransac_thr`, `max_img_size` |
| `config_pogo.yaml` | `pogo_node` | Odometry/loop noise std-devs, loss scales, `estimate_bias` |


## Custom messages

| Message | Fields |
|---------|--------|
| `RadarInfo` | Radar scan metadata (timestamps, frequency, etc.) |
| `LocalMapInfo` | Accumulated local map header, resolution, 2-D pose (`x`, `y`, `theta`) |
| `LoopCandidate` | Query/candidate timestamps, match score, image paths, local map resolution |

## Output

Atop ROS2 topics shown in RViz, Dr-PoGO outputs the odometry and pose-graph optimized trajectories in files in an output directory specified in the launch file (default is in the install space under `<ros2_ws>/install/dr_pogo/share/dr_pogo/<sequence_id>/`):
- `odometry_result/<sequence_id>.txt`: DRO odometry trajectory using the Boreas format.
- `pose_graph_traj.txt`: Pose-graph optimized trajectory in with `timestamp(us) x y theta` format.


### TODOs

- [ ] Improve documentation
- [ ] Make DRO faster with compilation (e.g., using `torch.compile` but need some tricks)
- [ ] Add the 3D odometry output as for 3DRO