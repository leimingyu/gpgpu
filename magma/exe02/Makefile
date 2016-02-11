MGMDIR=/opt/magma-2.0.0

INC:=
INC+=-I$(MGMDIR)/include
INC+=-I$(MGMDIR)/control
INC+=-I/usr/local/cuda/include

LIB:=
LIB+=-L$(MGMDIR)/lib
LIB+=-L$(MGMDIR)/testing
LIB+=-L$(MGMDIR)/testing/lin
LIB+=-L/usr/local/cuda/lib64
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


