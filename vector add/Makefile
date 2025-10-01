.PHONY: add myadd clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89




add:
	@$(NVCC) $(CUDA_FLAGS) add.cu -o add
	@./add

myadd:
	@$(NVCC) $(CUDA_FLAGS) myadd.cu -o myadd
	@./myadd

clean:
	rm -f add myadd *.o