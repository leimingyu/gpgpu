MKLROOT=/opt/intel/mkl
MKL_INC=$(MKLROOT)/include/
MKL_LIB=$(MKLROOT)/lib/intel64

LIBS ?= -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl 

all: sgemm

sgemm: sgemm.c
	g++ -DMKL_ILP64 -m64 -I$(MKL_INC) -L$(MKL_LIB) -Wl,--no-as-needed $+ -o $@  $(LIBS)

clean:
	rm sgemm
