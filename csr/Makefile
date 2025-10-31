.PHONY: mainc 

GCC = gcc
NVCC = nvcc
CUDA_FLAGS = -arch=sm_89 -gencode arch=compute_89,code=sm_89

# here -arch=sm_89 means Ada Lovelace architecture as mine is 4060
# -gencode arch=compute_89,code=sm_89 means to generate code for the architecture 89
# compute_89 means the virtual architecture and sm_89 means the real architecture

# make mainc ARGS="10000 10000"
mainc:
	@$(NVCC) $(CUDA_FLAGS) main.cu -o mainc.out
	@./mainc.out $(ARGS)

ARGS ?= 10000 10000

clean:
	rm -f *.out


test: 
	@$(GCC) -o test.out test.c
	@./test.out

testc:
	@$(NVCC) $(CUDA_FLAGS) test_cuda.cu -o test_cuda.out
	@./test_cuda.out

main:
	@$(GCC) -o main.out main.c
	@./main.out


