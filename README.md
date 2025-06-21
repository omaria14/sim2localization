# SIM2LOCALIZATION

Localization methodology that handles map changes and poor initial guesses, based on segmentation of both source and target (simulated) scans, matching segments, and ICP refinement.

[![Watch the demo video](https://img.youtube.com/vi/gEH-HKmtNh8/0.jpg)](https://youtu.be/gEH-HKmtNh8)


### Methodology Main Steps:

1. **Generate simulated scans**  
   Create simulated scans as target scans to represent the map.
   ➜ Using scan-based representations of the map makes it easier to segment it and match its segments with those of the source scan.
   ➜ To increase the chance of segment matches, generate multiple simulated scans around the initial pose guess with random variations.

2. **Segment both scans**  
   Segment the source scan and each simulated target scan.

3. **Create match pairs**  
   Consider all plausible combinations of source and target segments that could yield a valid global transformation:
   - Single-segment matches (`1 source → 1 target`)
   - Multi-segment matches (`2 source → 2 target`)
   
   Compute a global transformation for each proposed set of matches.
    ➜ Select the transformation with the highest total score, calculated by summing the similarity scores of the matched segments that support that transformation.

4. **Refine with ICP**  
   Run ICP using only the source segments that have a high probability of not being new or dynamic.  
   ➜ Apply the transformation with the highest score for the final alignment.

---

## 🛠️ Build Instructions

### Main Requirements:
- **ROS Noetic**
- **Open3D**

### Suggested Steps:

#### 1️⃣ Set up the Velodyne simulator (underlay workspace):
```bash
# Create and navigate to the Velodyne simulator workspace
mkdir -p ~/velodyne_sim_ws/src
cd ~/velodyne_sim_ws/src

# Clone the Velodyne simulator repository
git clone https://github.com/lmark1/velodyne_simulator

# Build the simulator
cd ~/velodyne_sim_ws
catkin_make
```

#### 2️⃣ Set up the SIM2LOCALIZATION workspace:
```bash
# Create and navigate to the sim2localization workspace
mkdir -p ~/sim2localization_ws/src
cd ~/sim2localization_ws/src

# Clone this SIM2LOCALIZATION repository
git clone https://github.com/your_user/sim2localization
```

#### 3️⃣ Source and build:
```bash
# Source Velodyne simulator workspace
source ~/velodyne_sim_ws/devel/setup.bash

# Build SIM2LOCALIZATION
cd ~/sim2localization_ws
catkin_make
```

---
# Usage
```bash
roslaunch sim2localization sim.launch

rosrun sim2localization simulated_scans_handler.py 

rosrun sim2localization main.py
```