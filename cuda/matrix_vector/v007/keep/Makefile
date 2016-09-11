.KEEP_STATE:

all: cuda

DFLAGS      =

OPTFLAGS    := -O3
OPTFLAGS    += #-g -pg

INCFLAGS    := -I.

#CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS) -DBLOCK_SHARED_MEM_OPTIMIZATION=1
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)

LDFLAGS     = $(OPTFLAGS)

#-------- cuda -----------------------------------------#
CUDA_DIR = /usr/local/cuda

OS_TYPE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

CUFLAGS   := -O3 --ptxas-options=-v -m${OS_TYPE} --keep

CU_INC := -I$(CUDA_DIR)/include
CU_INC += -I$(CUDA_DIR)/samples/common/inc

CU_LD :=
ifeq ($(OS_TYPE), 64)
	CU_LD += -L$(CUDA_DIR)/lib64
else
	CU_LD += -L$(CUDA_DIR)/lib
endif

CULIBS      :=
CULIBS      += -lcudart 

#SMS ?= 30 35 50 52
SMS ?= 52
ifeq ($(GENCODE_FLAGS),)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

CC          = g++
NVCC        = nvcc 

#-------- magma -----------------------------------------#
MAGMA_DIR = /opt/magma-2.0.2
#MAGMA_DIR = /opt/magma-2.0.0

MAGMA_INC :=
MAGMA_INC += $(CU_INC)
MAGMA_INC += -I$(MAGMA_DIR)/include 

MAGMA_LD :=
MAGMA_LD += $(CU_LD)
MAGMA_LD += -L$(MAGMA_DIR)/lib

MAGMA_LIBS := 
#MAGMA_LIBS += $(CULIBS) -lmagma -lmagma_sparse 
MAGMA_LIBS += $(CULIBS) -lmagma -llapack# -lopenblas

MAGMA_FLAGS = $(CUFLAGS) -DADD_


# --------  CUDA Version -------------------------------------
CUDA_SRC = cuda_main.cu 
cuda: cuda_main
cuda_main: $(CUDA_SRC) 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $(CUDA_SRC) $(CULIBS)

#---------------------------------------------------------------------
clean:
	rm -rf *.o gmon.out \
		cuda_main
