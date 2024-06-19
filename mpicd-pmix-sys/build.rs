use std::path::PathBuf;

fn main() {
    // Find libpmix.
    let pmix = pkg_config::Config::new()
        .atleast_version("4.0.0")
        .probe("pmix")
        .expect("failed to find pmix library");

    // Add proper link paths (including rpath) and names.
    for link_path in &pmix.link_paths {
        println!("cargo::rustc-link-search=native={}", link_path.display());
    }
    for link_file in &pmix.link_files {
        println!("cargo::rustc-link-lib={}", link_file.display());
    }

    // Generate the bindings.
    let builder = bindgen::builder();
    let mut clang_args = vec![];
    for include_path in &pmix.include_paths {
        clang_args.push(format!("-I{}", include_path.display()));
    }
    let bindings = builder
        .clang_args(clang_args.iter())
        .header("src/pmix.h")
        .allowlist_item("[Pp][Mm][Ii][Xx]_.+")
        .derive_default(true)
        .generate()
        .expect("failed to generate bindings");
    let out_dir = std::env::var("OUT_DIR").expect("missing $OUT_DIR env variable");
    let mut path = PathBuf::from(out_dir);
    path.push("bindings.rs");
    bindings
        .write_to_file(path)
        .expect("failed to write bindings");
}
