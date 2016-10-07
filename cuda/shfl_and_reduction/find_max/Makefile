.KEEP_STATE:

all: v001 v001_a

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

CUFLAGS   := -O3 --ptxas-options=-v -m${OS_TYPE}

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

# --------  CUDA Version -------------------------------------
v001: v001.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

v001_a: v001_a.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

#---------------------------------------------------------------------
clean:
	rm -rf *.o v001 v001_a 
