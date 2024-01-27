fn generate_cuda() {

    bindgen::Builder::default()
        .header("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include/cuda_runtime_api.h")
        .generate_comments(false)
        .rustified_enum(".+")
        .generate()
        .unwrap()
        .write_to_file("./rustlang/CUDA.rs")
        .unwrap();

    csbindgen::Builder::default()
        .input_bindgen_file("./rustlang/CUDA.rs")
        .csharp_dll_name("cudart64_12.dll")
        .csharp_namespace("BitTensor.Native")
        .csharp_class_name("CUDA")
        .csharp_class_accessibility("public")
        .generate_csharp_file("./dotnet/CUDA.g.cs")
        .unwrap();
}

fn generate_cublas() {

    bindgen::Builder::default()
        .header("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include/cublas_v2.h")
        .clang_args(&[
            "-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
        ])
        .generate_comments(false)
        .allowlist_function("^cublas\\w+")
        .rustified_enum(".+")
        .generate()
        .unwrap()
        .write_to_file("./rustlang/cuBLAS.rs")
        .unwrap();

    csbindgen::Builder::default()
        .input_bindgen_file("./rustlang/cuBLAS.rs")
        .csharp_dll_name("cublas64_12.dll")
        .csharp_namespace("BitTensor.Native")
        .csharp_class_name("cuBLAS")
        .csharp_class_accessibility("public")
        .generate_csharp_file("./dotnet/cuBLAS.g.cs")
        .unwrap();
}

fn generate_curand() {

    bindgen::Builder::default()
        .header("/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include/curand.h")
        .clang_args(&[
            "-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
        ])
        .generate_comments(false)
        .allowlist_function("^curand\\w+")
        .rustified_enum(".+")
        .generate()
        .unwrap()
        .write_to_file("./rustlang/cuRAND.rs")
        .unwrap();

    csbindgen::Builder::default()
        .input_bindgen_file("./rustlang/cuRAND.rs")
        .csharp_dll_name("curand64_10.dll")
        .csharp_namespace("BitTensor.Native")
        .csharp_class_name("cuRAND")
        .csharp_class_accessibility("public")
        .generate_csharp_file("./dotnet/cuRAND.g.cs")
        .unwrap();
}

fn generate_cudnn() {

    bindgen::Builder::default()
        .header("/Program Files/NVIDIA GPU Computing Toolkit/cuDNN/include/cudnn.h")
        .clang_args(&[
            "-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
        ])
        .generate_comments(false)
        .allowlist_function("^cudnn\\w+")
        .blocklist_type("CUstream_st")
        .rustified_enum(".+")
        .generate()
        .unwrap()
        .write_to_file("./rustlang/cuDNN.rs")
        .unwrap();

    csbindgen::Builder::default()
        .input_bindgen_file("./rustlang/cuDNN.rs")
        .csharp_dll_name("cudnn64_8.dll")
        .csharp_namespace("BitTensor.Native")
        .csharp_class_name("cuDNN")
        .csharp_class_accessibility("public")
        .generate_csharp_file("./dotnet/cuDNN.g.cs")
        .unwrap();
}

fn main() {
    generate_cuda();
    generate_curand();
    generate_cublas();
    generate_cudnn();
}
