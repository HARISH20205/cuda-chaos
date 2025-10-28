.PHONY: add myadd clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89




add:
	@$(NVCC) $(CUDA_FLAGS) add.cu -o add.out
	@./add.out

myadd:
	@$(NVCC) $(CUDA_FLAGS) myadd.cu -o myadd.out
	@./myadd.out

clean:
	rm -f add.out myadd.out *.out