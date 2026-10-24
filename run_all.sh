#!/bin/bash
cd /home/claude/work
python3 stage_main.py > results/main.log 2>&1
python3 stage_variants.py > results/variants.log 2>&1
python3 stage_misc.py > results/misc.log 2>&1
python3 stage_nested.py > results/nested.log 2>&1
echo ALLDONE > results/all.done
