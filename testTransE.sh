#!/bin/zsh
# set -ex

TIMEFMT='%J   %U  user %S system %P cpu %*E total'$'\n'\
'avg shared (code):         %X KB'$'\n'\
'avg unshared (data/stack): %D KB'$'\n'\
'total (sum):               %K KB'$'\n'\
'max memory:                %M MB'$'\n'\
'page faults from disk:     %F'$'\n'\
'other page faults:         %R'

run=( 0 1 2 3 4 5 6 7 8 9 )
array=( 500 )
for j in "${run[@]}"
do
    for i in "${array[@]}"
    do
        { time ( python3 test_transe.py dataset$i ) } > logs/valid.duration.transe.dataset$i.$j 2>&1
    done
done
