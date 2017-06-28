#!/usr/bin/env bash
kernprof -l main_profiling.py
python -m line_profiler main_profiling.py.lprof