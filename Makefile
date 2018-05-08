# Makefile for compilation of gcc and cuda version of pmuca ising2D

CPU_FLAGS=-pedantic -Wall -Wextra -O3 -std=c++0x -I./Random123/include/

GPU_ARCHS=-arch=sm_35 -rdc=true -I./Random123/include/ -lineinfo

# opencl path
export CPLUS_INCLUDE_PATH=/net/nfs/opt/cuda-7.5/include
export LIBRARY_PATH=/net/nfs/opt/cuda-7.5/lib64/:$LIBRARY_PATH
export LD_RUN_PATH=/net/nfs/opt/cuda-7.5/lib64/:$LD_RUN_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH+$LD_LIBRARY_PATH:}/net/nfs/opt/cuda-7.5/lib64/
ifeq ($(CONFIG),debug)
	OPT =-O0 -g
else
	OPT =
endif


all: cl

cl: ising2D_cl

ising2D_cl: ising2D_cl.cpp
	g++ ising2D_cl.cpp -lOpenCL $(CPU_FLAGS) $(OPT) -o $@

clean:
	rm -f ising2D_cl
