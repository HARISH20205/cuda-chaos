.PHONY: 01 01_obj 01_obj_exe_run 02 03 clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89

01: 
	@$(GCC) -o 01 01.c
	@./01

02: 
	@$(GCC) -o 02 02.c
	@./02


03_cu:
	@$(NVCC) $(CUDA_FLAGS) 03.cu -o 03_cu
	@./03_cu


clean:
	rm -f 01 02 03_cu *.o