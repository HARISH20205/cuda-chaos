.PHONY: idxx simple clean

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89




idxx:
	@$(NVCC) $(CUDA_FLAGS) idxx.cu -o idxx.out
	@./idxx.out

simple:
	@$(NVCC) $(CUDA_FLAGS) simple.cu -o simple.out
	@./simple.out

clean:
	rm -f idxx.out simple.out *.out