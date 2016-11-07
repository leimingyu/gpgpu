OPENBLAS_INC=/opt/openblas/include


ATLAS_DIR=/home/ubuntu/Downloads/atlas/ATLAS-3.8.4-arm/build_tx1
ATLAS_INC=$(ATLAS_DIR)/include
ATLAS_LIB=$(ATLAS_DIR)/lib

#LIBS ?= -llapack -lf77blas -lcblas -latlas 
#LIBS ?= -lptlapack -lptf77blas -lptcblas -latlas -lpthread
LIBS ?= -llapack -lptf77blas -lptcblas -latlas -lpthread

all: sgemm sgemv

sgemm: sgemm.c
	g++ -I$(ATLAS_INC) -I$(OPENBLAS_INC) -L$(ATLAS_LIB) $+ -o $@  $(LIBS)

sgemv: sgemv.c
	g++ -I$(ATLAS_INC) -I$(OPENBLAS_INC) -L$(ATLAS_LIB) $+ -o $@  $(LIBS)

clean:
	rm sgemm sgemv
