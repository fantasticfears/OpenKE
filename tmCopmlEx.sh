#!/bin/bash
set -ex

# array=( 0 1 2 3 4 5 6 7 8 9)
array=( 0 1 2 )
for i in "${array[@]}"
do
	name=ComplEx-FB15K237-neg_20-dim_100-epoch_1000-alpha_${i//./}
	start=$SECONDS
	python3 train_complex.py > logs/running.$name
	duration=$(( SECONDS - start ))
	echo $duration >> logs/duration.$name
done

