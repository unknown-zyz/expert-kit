fn main() -> Result<(), Box<dyn std::error::Error>> {
    // tonic_build::("../ek-proto/ek")?;
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }
    tonic_build::configure().build_server(true).compile_protos(
        &[
            "../ek-proto/ek/control/v1/control.proto",
            "../ek-proto/ek/worker/v1/expert.proto",
            "../ek-proto/ek/object/v1/object.proto",
            "../ek-proto/onnx/onnx.proto",
        ],
        &["../ek-proto"],
    )?;

    let cuda_root = std::env::var_os("CUDA_HOME")
        .or_else(|| std::env::var_os("CUDA_PATH"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/opt/cuda"));
    let nvtx_include = std::env::var_os("EK_NVTX_INCLUDE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| cuda_root.join("include"));
    let nvtx_header = nvtx_include.join("nvtx3/nvToolsExt.h");
    let mut nvtx = cc::Build::new();
    nvtx.file("src/worker/nvtx_shim.c");
    if nvtx_header.is_file() {
        nvtx.include(&nvtx_include).define("EK_HAVE_NVTX", None);
        println!("cargo:rerun-if-changed={}", nvtx_header.display());
    } else {
        println!(
            "cargo:warning=NVTX header not found at {}; profiling ranges will be disabled",
            nvtx_header.display()
        );
    }
    nvtx.compile("ek_nvtx_shim");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=EK_NVTX_INCLUDE");
    println!("cargo:rerun-if-changed=src/worker/nvtx_shim.c");
    eprintln!("protobuf built");
    Ok(())
}
