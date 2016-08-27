INC:=
#INC+=-I/opt/magma-2.0.0-beta3/include
INC+=-I/opt/magma-2.0.0/include
INC+=-I/usr/local/cuda-7.0/include

LIB:=
#LIB+=-L/opt/magma-2.0.0-beta3/lib
LIB+=-L/opt/magma-2.0.0/lib
LIB+=-L/usr/local/cuda-7.0/lib64

OPT:=
OPT+= -O2 

LNK:=
LNK+= -lcudart -lcublas
LNK+= -lmagma -lmagma_sparse

all: isamax

isamax: isamax.cpp
	g++ $(INC) $(LIB) $(OPT) $+ -o $@ $(LNK)

.PHONY: clean
clean:
	rm -rf isamax
