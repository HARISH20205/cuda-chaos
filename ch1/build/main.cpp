#include <torch/extension.h>

std::string hello_world(){
    return "Hello World";
}
int add(int a,int b){
    return a+b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("hello_world", torch::wrap_pybind_function(hello_world), "hello_world");
m.def("add", torch::wrap_pybind_function(add), "add");
}