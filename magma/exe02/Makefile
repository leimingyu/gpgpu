INC:=
INC+=-I/opt/magma-2.0.0-beta3/include
INC+=-I/opt/magma-2.0.0-beta3/control
INC+=-I/usr/local/cuda-7.0/include

LIB:=
LIB+=-L/opt/magma-2.0.0-beta3/lib
LIB+=-L/opt/magma-2.0.0-beta3/testing
LIB+=-L/opt/magma-2.0.0-beta3/testing/lin
LIB+=-L/usr/local/cuda-7.0/lib64
LIB+=-L/opt/openblas/lib

OPT:=
OPT+= -O2 -fPIC -fopenmp
OPT+= -DADD_ -Wall
OPT+= -DHAVE_CUBLAS -DMIN_CUDA_ARCH=200

LNK:=
LNK+= -lmagma #-lmagma_sparse
LNK+= -ltest 
LNK+= -llapacktest
LNK+= -lcudart -lcublas -lcusparse
LNK+= -lopenblas

all: lu 

lu: lu.cpp
	g++ $(OPT) $(LIB) $(INC) $+ -o $@ $(LNK)

.PHONY: clean
clean:
	rm -rf lu 


