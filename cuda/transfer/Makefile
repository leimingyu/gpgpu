CUDA_DIR = /usr/local/cuda-7.5

GCC ?= g++
NVC = $(CUDA_DIR)/bin/nvcc  -ccbin $(GCC)

OS_TYPE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

NVCC_FLAGS   :=	
NVCC_FLAGS   += -O2 
NVCC_FLAGS   += -m${OS_TYPE}

NVINC = -I$(CUDA_DIR)/include/ -I$(CUDA_DIR)/samples/common/inc
LDFLAG := 
ifeq ($(OS_TYPE), 64)
  LDFLAG += -L$(CUDA_DIR)/lib64
else
  LDFLAG += -L$(CUDA_DIR)/lib
endif

LIBS :=
LIBS += -lcudart

SMS ?= 30 35 50 52
ifeq ($(GENCODE_FLAGS),)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

all: main 

main.o: main.cu
	$(NVC) $(NVINC) $(GENCODE_FLAGS) -o $@ -c $<

main: main.o
	$(NVC) $(NVINC) $(LDFLAG) $(GENCODE_FLAGS) -o $@ $+ $(LIBS) 

.PHONY: clean
clean:
	rm -rf main main.o

clobber: clean
