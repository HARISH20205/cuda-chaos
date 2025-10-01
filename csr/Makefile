.PHONY: test 

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89



test: 
	@$(GCC) -o test test.c
	@./test

main:
	@$(GCC) -o main main.c
	@./main



mainc:
	@$(NVCC) $(CUDA_FLAGS) main.cu -o main
	@./main


clean:
	rm -f test main *.o


