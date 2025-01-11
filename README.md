# live-pose

Intel RealSense Depth Camera compatible Python package for live 6 DOF pose estimation.
<br>
For Jetson device use [Jetpose](https://github.com/Kaivalya192/Jetpose).

## Table of Contents

- [Installation](#installation)
- [Preparation](#preparation)
- [Usage](#usage)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Kaivalya192/live-pose.git
    cd live-pose
    ```

## Preparation

### Docker Build

1. **Build the Docker container**:
    ```sh
    cd docker
    docker build --network host -t foundationpose .
    ```

2. **Install Weights**:
   - Download the weights from [this link](https://drive.google.com/drive/folders/1wJayPZzZLZb6sxm6EeOQCJvzOAibJ693?usp=sharing) and place them under `live-pose/FoundationPose/weights`.

## Usage

### Running the Container

1. **Run the container**:
    ```sh
    bash docker/run_container.sh
    ```
    note: To run on windows install `Cygwin` and execute `./docker/run_container_win.sh`
### Building packages

1. **Build**:
    ```sh
    CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build.bash
    ```
### Running the Model

1. **Run the live pose estimation**:
    ```sh
    bash run_live.sh
    ```
    note: To run on windows install `Cygwin` and execute `./run_live.sh`
   
3. **Locate the .obj file**:
    <br> Note: For novel object you can use [Object Recustruction Framework](https://github.com/Kaivalya192/Object_Reconstruction) </br>
    
4. **Masking**:
    <br> here select the Boundry points of object in first frame </br>
