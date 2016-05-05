#!/bin/bash
nvcc -gencode arch=compute_30,code=sm_30 -cubin test.cu
