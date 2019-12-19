#!/bin/sh
for run in {1..100}
    do
        time python3 ./go_play.py -n 5 -p1 my -p2 random -t 100;
        echo $run;
    done
