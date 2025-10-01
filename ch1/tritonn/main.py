import torch
import triton
import triton.language as tl


# 1. Define the Triton GPU kernel using @triton.jit
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input vector
    y_ptr,  # Pointer to the second input vector
    output_ptr,  # Pointer to the output vector
    n_elements,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program (thread block) processes
):
    """
    This is a Triton kernel for adding two vectors element-wise.
    Each program (thread block) processes BLOCK_SIZE elements.
    """
    # Get the ID of the current program (thread block) along the x-axis.
    # We're launching a 1D grid of blocks, so axis=0 is sufficient.
    pid = tl.program_id(axis=0)

    # Calculate the starting offset for this program's block of elements.
    block_start = pid * BLOCK_SIZE

    # Generate a range of offsets (indices) within this program's block.
    # E.g., if BLOCK_SIZE is 128, offsets will be [0, 1, ..., 127]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle elements at the end of the array that might not
    # fill a full BLOCK_SIZE. This prevents out-of-bounds memory access.
    mask = offsets < n_elements

    # Load data from memory. Only load elements where the mask is True.
    # For masked-out elements, fill with 0.0 (or any default).
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Perform the element-wise addition.
    output = x + y

    # Store the result back to memory, again using the mask.
    tl.store(output_ptr + offsets, output, mask=mask)


# 2. Wrapper function to easily call the Triton kernel
def triton_add(x: torch.Tensor, y: torch.Tensor):
    # Ensure inputs are on CUDA and have the same type
    assert x.is_cuda and y.is_cuda and x.dtype == y.dtype
    assert x.shape == y.shape
    n_elements = x.numel()

    # Create the output tensor
    output = torch.empty_like(x)

    # Calculate the number of programs (thread blocks) needed.
    # Each program handles BLOCK_SIZE elements.
    # We'll use a heuristic BLOCK_SIZE for this simple example.
    # For real applications, autotuning is recommended.
    BLOCK_SIZE = 1024  # A common block size for many GPUs
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)

    # Launch the kernel!
    # The `[(num_programs,)]` syntax tells Triton the grid dimensions.
    # (Here, a 1D grid with 'num_programs' blocks).
    add_kernel[(num_programs,)](
        x.data_ptr(),
        y.data_ptr(),
        output.data_ptr(),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,  # Pass BLOCK_SIZE as a compile-time constant
    )

    return output


# 3. Test the Triton kernel
if __name__ == "__main__":
    # Define input size
    size = 2**20  # 1 million elements

    # Create random tensors on the GPU
    torch.manual_seed(0)
    x_gpu = torch.randn(size, device="cuda", dtype=torch.float32)
    y_gpu = torch.randn(size, device="cuda", dtype=torch.float32)

    # Run with Triton
    output_triton = triton_add(x_gpu, y_gpu)

    # Run with PyTorch (for comparison)
    output_torch = x_gpu + y_gpu

    # Verify correctness
    # Check if the results are close (due to potential floating point differences)
    assert torch.allclose(output_triton, output_torch, atol=1e-5, rtol=1e-5)
    print("Triton and PyTorch results match!")

    # 4. Benchmarking (optional, but good practice)
    # This shows the performance benefit
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["size"],  # Argument names to use for the x-axis of the plot
            x_vals=[2**i for i in range(12, 28)],  # Values for the x-axis (input sizes)
            line_arg="provider",  # Group lines by this argument
            line_vals=["triton", "torch"],  # Values for the 'provider' argument
            line_names=["Triton", "Torch"],  # Names to display for each line
            styles=[("blue", "-"), ("green", "-")],  # Line styles
            ylabel="GB/s",  # Y-axis label
            plot_name="Vector Addition",  # Title of the plot
            args={"BLOCK_SIZE": 1024},  # Fixed arguments for the benchmark
        )
    )
    def benchmark_add(size, provider, BLOCK_SIZE):
        x = torch.randn(size, device="cuda", dtype=torch.float32)
        y = torch.randn(size, device="cuda", dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]  # Percentiles for measuring performance

        if provider == "torch":
            ms, gbps = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        if provider == "triton":
            ms, gbps = triton.testing.do_bench(
                lambda: add_kernel[(triton.cdiv(size, BLOCK_SIZE),)](
                    x.data_ptr(),
                    y.data_ptr(),
                    torch.empty_like(x).data_ptr(),
                    size,
                    BLOCK_SIZE=BLOCK_SIZE,
                ),
                quantiles=quantiles,
            )
        return ms, gbps

    # Uncomment to run the benchmark and generate a plot
    # benchmark_add.run(print_data=True, show_threads=True)
