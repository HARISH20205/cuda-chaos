.PHONY: test 

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89


ARGS ?= 10000 10000


test: 
	@$(GCC) -o test test.c
	@./test

testc:
	@$(NVCC) $(CUDA_FLAGS) test_cuda.cu -o test_cuda
	@./test_cuda

main:
	@$(GCC) -o main main.c
	@./main 


mainc:
	@$(NVCC) $(CUDA_FLAGS) main.cu -o main
	@./main $(ARGS)


clean:
	rm -f test main *.o


