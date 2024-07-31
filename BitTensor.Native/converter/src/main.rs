fn generate_binding(name: &str, source_header: &str, clang_args: &[&str], allowlist: &str, dll_name: &str) {

    const CS_NAMESPACE: &str = "BitTensor.CUDA.Interop";

    let rs_file = format!("./rustlang/{}.rs", name);
    let cs_file = format!("./dotnet/{}.g.cs", name);

    bindgen::Builder::default()
        .header(source_header)
        .clang_args(clang_args)
        .generate_comments(false)
        .allowlist_function(allowlist)
        .allowlist_type(allowlist)
        .rustified_enum(".+")
        .generate()
        .unwrap()
        .write_to_file(rs_file.as_str())
        .unwrap();

    csbindgen::Builder::default()
        .input_bindgen_file(rs_file.as_str())
        .csharp_dll_name(dll_name)
        .csharp_namespace(CS_NAMESPACE)
        .csharp_class_name(name)
        .csharp_class_accessibility("public")
        .generate_csharp_file(cs_file)
        .unwrap();
}

fn main() {
    generate_binding(
        "cudaRT",
        "/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include/cuda_runtime_api.h",
        &[],
        "^\\w+",
        "cudart64_12.dll",
    );

    generate_binding(
        "cuRAND",
        "/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include/curand.h",
        &["-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"],
        "^curand\\w+",
        "curand64_10.dll",
    );

    generate_binding(
        "cuBLAS",
        "/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include/cublas_v2.h",
        &["-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"],
        "^cublas\\w+",
        "cublas64_12.dll",
    );

    generate_binding(
        "cuTENSOR",
        "/Program Files/NVIDIA cuTENSOR/v2.0/include/cutensor.h",
        &[
            "-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include",
            "-I/Program Files/NVIDIA cuTENSOR/v2.0/include/"],
        "^cutensor\\w+",
        "cutensor.dll",
    );

    generate_binding(
        "cuDNN",
        "/Program Files/NVIDIA/CUDNN/v9.2/include/12.5/cudnn.h",
        &["-I/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5/include"],
        "^cudnn\\w+",
        "cudnn64_9.dll",
    );
}
