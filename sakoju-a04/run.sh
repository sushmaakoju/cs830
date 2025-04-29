#!/bin/bash
#test
# path="/home/csg/sa1725/sakoju_a01/test1.vw"
# script_path="/home/csg/sa1725/sakoju/a02/code.py"

python3 code.py "$@"

#python2 ./color-validator ./color-reference fc 5 < examples/test1.col
#python2 ./color-validator ./run.sh dfs 5 < examples/test1.col
#python2 ./color-validator ./run.sh mcv 10 -restart < examples/queen5_5.col
#./run.sh dfs 3 -restart < ./examples/test0.col
#./color-reference fc 3 < examples/test0.col
#python2 ./color-validator ./run.sh dfs 5 < examples/queen5_5.col 
#./make-vw 5 5 0.15 10 -seed 56
#./make-vw 5 5 0.3 4 -seed 56