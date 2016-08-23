Download software from http://icl.cs.utk.edu/magma/software/.

```{engine='bash'}
sudo apt-get install gfortran 

git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/opt/openblas install
```
