use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize)]
struct Extension {
    name: &'static str,
    extension_version: &'static str,
}

#[derive(Serialize)]
struct ApiLayer {
    name: &'static str,
    library_path: PathBuf,
    api_version: &'static str,
    implementation_version: &'static str,
    description: &'static str,
    instance_extensions: &'static [Extension],
}

#[derive(Serialize)]
struct Manifest {
    file_format_version: &'static str,
    api_layer: ApiLayer,
}

fn main() {
    let cmd = std::env::args().nth(1).expect("Expect one verb: `install`");
    if cmd == "install" {
        std::env::set_current_dir(
            std::path::Path::new(&std::env::var_os("CARGO_MANIFEST_DIR").unwrap()).join(".."),
        )
        .unwrap();
        let mut cmd = std::process::Command::new("cargo");
        cmd.args(["build", "--release"]);
        let status = cmd.status().expect("Failed to run cargo build");
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }

        let home = std::env::var_os("HOME").unwrap();
        let home = std::path::Path::new(&home);
        let local_share = home.join(".local").join("share");
        let api_layer_dir = local_share
            .join("openxr")
            .join("1")
            .join("api_layers")
            .join("explicit.d");

        let libdir = home.join(".local").join("lib").join("xr_passthrough_layer");
        match std::fs::create_dir(&libdir) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => panic!("Failed to create libdir: {}", e),
        }
        match std::fs::create_dir_all(&api_layer_dir) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => panic!("Failed to create api_layer_dir: {}", e),
        }
        std::fs::copy(
            std::path::Path::new("target")
                .join("release")
                .join("libxr_passthrough_layer.so"),
            libdir.join("liblayer.so"),
        )
        .unwrap();
        let manifest = Manifest {
            file_format_version: "1.0.0",
            api_layer: ApiLayer {
                name: "XR_APILAYER_YX_passthrough",
                library_path: libdir.join("liblayer.so"),
                api_version: "1.0",
                implementation_version: "1",
                description: "Passthrough",
                instance_extensions: &[Extension {
                    name: "XR_HTC_passthrough",
                    extension_version: "1",
                }],
            },
        };

        let json_file = std::fs::OpenOptions::new()
            .truncate(true)
            .write(true)
            .create(true)
            .open(api_layer_dir.join("XR_APILAYER_YX_passthrough.json"))
            .unwrap();
        serde_json::to_writer_pretty(json_file, &manifest).unwrap();
    } else {
        panic!("Unknown command: {}", cmd);
    }
}
