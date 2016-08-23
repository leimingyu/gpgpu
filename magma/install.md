Download software from http://icl.cs.utk.edu/magma/software/.

```{engine='bash'}
sudo apt-get install gfortran 

git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/opt/openblas install

sudo vim /etc/ld.so.conf.d/openblas.conf
add /opt/openblas/lib
sudo ldconfig
```
In the bashrc file, I add the openblas source dir there, including the cuda dir.

```{engine='bash'}
export CUDADIR=/usr/local/cuda                                                                     
# openblas                                                                      
export OPENBLASDIR=/opt/openblas 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/openblas/lib

# Then exit ~/.bashrc and source it
source ~/.bashrc
```

Next, I go back to magma-2.0.0-beta3 folder, copy the make.inc.openblas to make.inc file, and run make.
```
tar -xvzf magma-2.0.2.tar.g
cd magma-2.0.2/
cp make.inc.openblas make.inc
make
```
Noted that, you may need to add new architecture support (for maxwell) in the Makefile. Then run make.
```
#GPU_TARGET ?= Fermi Kepler Maxwell                                             
GPU_TARGET ?= Kepler Maxwell

ifneq ($(findstring Maxwell, $(GPU_TARGET)),)                                   
    GPU_TARGET += sm50 sm52                                                     
endif  

ifneq ($(findstring sm50, $(GPU_TARGET)),)                                      
    MIN_ARCH ?= 500                                                             
    NV_SM    += -gencode arch=compute_50,code=sm_50                             
    NV_COMP  := -gencode arch=compute_50,code=compute_50                        
endif                                                                           
ifneq ($(findstring sm52, $(GPU_TARGET)),)                                      
    MIN_ARCH ?= 520                                                             
    NV_SM    += -gencode arch=compute_52,code=sm_52                             
    NV_COMP  := -gencode arch=compute_52,code=compute_52                        
endif 
```

copy the files to /opt dir and set up the path in ~/.bashrc
```

export PATH=$PATH:/opt/magma-2.0.0/include                                      
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/magma-2.0.0/lib
```
