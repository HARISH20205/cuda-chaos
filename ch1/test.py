import torch


def time_elapsed_pytorch(func, inp):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(5):
        func(inp)
    start.record()
    func(inp)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


b = torch.randn(10000, 10000).cuda()


def square_2(n):
    return n * n


def power_2(n):
    return n**2


time_elapsed_pytorch(torch.square, b)
time_elapsed_pytorch(square_2, b)
time_elapsed_pytorch(power_2, b)

print("=============")
print("Profiling torch.square")
print("=============")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    torch.square(b)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling square_2(n*n)")
print("=============")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    square_2(b)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling power_2(n**2)")
print("=============")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    power_2(b)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
