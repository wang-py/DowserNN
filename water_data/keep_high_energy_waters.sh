#!/bin/bash

INPUT_PDB=$1

OUTPUT_PDB=$2

ENERGY_CUTOFF=-4.0

awk -v cutoff="$ENERGY_CUTOFF" 'BEGIN {print cutoff}'

grep 'OW' $INPUT_PDB | awk '(substr($0, 61, 5) + 0) > -4.0' > $OUTPUT_PDB
