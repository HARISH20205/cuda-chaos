import torch
import triton
import triton.language as tl

# Triton kernel
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

# Python function to launch kernel
def add_vectors(x, y):
    n_elements = x.numel()
    output = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    end_event.record()

    # Wait for GPU to finish
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    
    return output, elapsed_ms

# Example usage
x = torch.arange(10_000, device='cuda', dtype=torch.float32)
y = torch.arange(10_000, device='cuda', dtype=torch.float32)
z, time_ms = add_vectors(x, y)

print("Result (first 10 elements):", z[:10])
print(f"Kernel execution time: {time_ms:.3f} ms")
