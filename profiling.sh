#!/usr/bin/env bash
python -m cProfile -s "cumtime" main_profiling.py > profiling_report.txt