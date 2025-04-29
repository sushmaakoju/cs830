#!/bin/bash
#test
# path="/home/csg/sa1725/sakoju_a01/test1.vw"
# script_path="/home/csg/sa1725/sakoju/a02/code.py"
export PYTHONDONTWRITEBYTECODE=1
python3 -B code.py "$@"
# python3 -B test.py "$@"


# test all inputs
# dir="./problems/cnf/complex"
# dir="./problems/cnf/simple"
# for file in "$dir"/*; do
#     ./run_test.sh < "$file" 2>&1
# done

#~cs730/submit a5-grad 
