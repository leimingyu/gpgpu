.KEEP_STATE:

all: cublas_main cuda_v001 cuda_v002 cuda_v003 cuda_v004 cuda_v005 \
	cuda_v006 cuda_v007 cuda_v008 cuda_v010 cuda_v011

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

# --------  CUBLAS Version -------------------------------------
cublas_main: cublas_main.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS) -lcublas


# --------  CUDA Version -------------------------------------
cuda_v001: cuda_v001.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v002: cuda_v002.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v003: cuda_v003.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v004: cuda_v004.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v005: cuda_v005.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v006: cuda_v006.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v007: cuda_v007.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v008: cuda_v008.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v010: cuda_v010.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

cuda_v011: cuda_v011.cu 
	$(NVCC) $(CUFLAGS) $(CU_INC) $(CU_LD) $(GENCODE_FLAGS) -o $@ $+ $(CULIBS)

# -------- MAGMA  Version -------------------------------------
#MAGMA_SRC = magma_main.cu magma_wtime.cu
#magma: magma_main
#magma_main: $(MAGMA_SRC) 
#	$(NVCC) $(MAGMA_FLAGS) $(MAGMA_INC) $(MAGMA_LD)  $(GENCODE_FLAGS) -o $@ $(MAGMA_SRC) $(MAGMA_LIBS)

#---------------------------------------------------------------------
clean:
	rm -rf *.o gmon.out \
		cublas_main \
		cuda_v001 \
		cuda_v002 \
		cuda_v003 \
		cuda_v004 \
		cuda_v005 \
		cuda_v006 \
		cuda_v007 \
		cuda_v008 \
		cuda_v010 \
		cuda_v011
