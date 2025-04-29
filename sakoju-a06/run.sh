#!/bin/bash
#test
# path="/home/csg/sa1725/sakoju_a01/test1.vw"
# script_path="/home/csg/sa1725/sakoju/a02/code.py"
$ pip install antlr4-tools
pip install antlr4-python3-runtime
antlr4 -Dlanguage=Python3 cnf.g4 
python3 code.py "$@"

#~cs730/submit a5-grad 
# python3 make3cnf.py 100 400 > ./sample-cnf/100_400.cnf

# ./sat-validator ./run.sh < ./sample-cnf/10_30.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/10_40.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/25_75.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/25_100.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/50_150.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/50_200.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/75_225.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/75_300.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/100_300.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/100_400.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/150_300.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/150_600.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/200_600.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./run.sh < ./sample-cnf/200_800.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/10_30.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/10_40.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/25_75.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/25_100.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/50_150.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/50_200.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/75_225.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/75_300.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/100_300.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/100_400.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/150_600.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/150_450.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/200_800.cnf 2>&1 | tee a "$output_file"
# ./sat-validator ./sat-reference < ./sample-cnf/200_600.cnf 2>&1 | tee a "$output_file"