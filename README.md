# live-pose

Intel RealSense Depth Camera compatible Python package for live 6 DOF pose estimation.

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
    note: To run on windows install `Cygwin` and execute `./docker/run_container.sh`

### Running the Model

1. **Run the live pose estimation**:
    ```sh
    bash run_live.sh
    ```
    note: To run on windows install `Cygwin` and execute `./run_live.sh`

