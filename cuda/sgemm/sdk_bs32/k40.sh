#!/bin/bash

nvprof --print-gpu-summary --csv ./matrixMul -device=0 -wA=320 -hA=320 -wB=320 -hB=320 2> k40_matrixMul_bs32_summary.csv
nvprof --print-gpu-trace   --csv ./matrixMul -device=0 -wA=320 -hA=320 -wB=320 -hB=320 2> k40_matrixMul_bs32_trace.csv
nvprof --metrics all       --csv ./matrixMul -device=0 -wA=320 -hA=320 -wB=320 -hB=320 2> k40_matrixMul_bs32_metrics.csv
