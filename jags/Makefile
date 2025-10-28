.PHONY: 01 clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89


mainc:
	@$(NVCC) $(CUDA_FLAGS) mainc.cu -o mainc.out
	@./mainc.out


clean:
	rm -f mainc.out *.o