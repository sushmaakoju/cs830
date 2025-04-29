#!/bin/bash
#test
path="/home/csg/sa1725/a02/worlds/test1.vw"
script_path="/home/csg/sa1725/sakoju/a02/code.py"

python3 code.py "$1" "$2" "$3" "$4"

#./vw-validator -novis -time 500 -- ./run.sh uniform-cost h0 ./worlds/test1.vw
#./make-vw 5 5 0.15 10 -seed 56
#./make-vw 5 5 0.3 4 -seed 56