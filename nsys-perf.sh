PROFILE_DIR=profile

if [ ! -d "$PROFILE_DIR" ]; then
    mkdir -p "$PROFILE_DIR"
    echo "created $PROFILE_DIR"
fi

PROFILE_FILE=./${PROFILE_DIR}/profile_`date +%Y%m%d-%H%M`

echo ${PROFILE_FILE}

nsys profile --cudabacktrace=all -o ${PROFILE_FILE} conda run -n gsplatlab python ./pygsplat/gsplat_rasterize_test.py

nsys-ui ${PROFILE_FILE}.nsys-rep