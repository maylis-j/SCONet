if [ "$2" == "" ]; then
    CFG_FILE="configs/pointcloud/AbdomenCT-1K.yaml"
else
    CFG_FILE=$2
fi

if  [ "$1" == "" ]; then
    echo "Error : must provide path to surf3d as first command line argument"
    exit
fi

python save_infos_dataset.py --config $CFG_FILE
python contour_detection.py --config $CFG_FILE
python pointcloud_generation.py --config $CFG_FILE --exe_path $1 -rle -s
