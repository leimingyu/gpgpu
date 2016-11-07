OPENBLAS_INC=/opt/openblas/include/
OPENBLAS_LIB=/opt/openblas/lib/

LIBS ?= -lopenblas -lpthread -lgfortran

all: sgemm sgemv sgemv1

sgemm: sgemm.c
	g++ -I$(OPENBLAS_INC) -L$(OPENBLAS_LIB) $+ -o $@  $(LIBS)

sgemv: sgemv.c
	g++ -I$(OPENBLAS_INC) -L$(OPENBLAS_LIB) $+ -o $@  $(LIBS)

sgemv1: sgemv1.c
	g++ -I$(OPENBLAS_INC) -L$(OPENBLAS_LIB) $+ -o $@  $(LIBS)

clean:
	rm sgemm sgemv sgemv1 sgemm
