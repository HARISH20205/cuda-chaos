import torch

from torch.utils.cpp_extension import load_inline


cpp_source = """
std::string hello_world(){
    return "Hello World";
}
int add(int a,int b){
    return a+b;
}
"""

my_module = load_inline(
    name="my_module",
    cpp_sources=[cpp_source],
    functions=["hello_world", "add"],
    verbose=True,
    build_directory="/home/harish/code/CUDA/ch1/build",
    with_cuda=True,
)

print(my_module.hello_world())
print(my_module.add(1, 2))
