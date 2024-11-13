#!/bin/bash

make clean >> /dev/null
make quicksort >> /dev/null

sudo perf record -g ./quicksort
perf report >> report.txt