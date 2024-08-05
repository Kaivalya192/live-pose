DIR=$(pwd)

cd $DIR/FoundationPose/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11

cd ${DIR}