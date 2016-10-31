ATLAS_INC=/home/leiming/Downloads/atlas/ATLAS_3.10.3/ATLAS_intel4790k/include
ATLAS_LIB=/home/leiming/Downloads/atlas/ATLAS_3.10.3/ATLAS_intel4790k/lib

#LIBS ?= -llapack -lf77blas -lcblas -latlas 
LIBS ?= -lptlapack -lptf77blas -lptcblas -latlas -lpthread

all: sgemm sgemv

sgemm: sgemm.c
	g++ -I$(ATLAS_INC) -L$(ATLAS_LIB) $+ -o $@  $(LIBS)

sgemv: sgemv.c
	g++ -I$(ATLAS_INC) -L$(ATLAS_LIB) $+ -o $@  $(LIBS)

clean:
	rm sgemm sgemv
