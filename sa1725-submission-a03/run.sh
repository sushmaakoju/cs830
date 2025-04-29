#!/bin/bash
#test
path="/home/csg/sa1725/sakoju_a01/test1.vw"
script_path="/home/csg/sa1725/sakoju/a02/code.py"

python3 code.py "$@"

#./rrt-validator -grad -o grad-1-space-2.pdf -- ./rrt-reference -grad -seed 1 < ./examples/simple-0.sw
#./rrt-reference -grad -seed 1 < ./examples/space-1.sw > space-1.txt
#./rrt-reference -grad -seed 1 < ./examples/test1.sw > test1.txt
#./make-vw 5 5 0.15 10 -seed 56
#./make-vw 5 5 0.3 4 -seed 56