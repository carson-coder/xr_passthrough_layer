{
  inputs.fenix = {
    inputs.nixpkgs.follows = "nixpkgs";
    url = "github:nix-community/fenix";
  };
  inputs.rust-manifest = {
    flake = false;
    url =
      "https://static.rust-lang.org/dist/2025-02-24/channel-rust-nightly.toml";
  };
  description = "xr_passthrough_layer";

  outputs = { self, nixpkgs, fenix, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        overlays = [ fenix.overlays.default ];
      };
      rust-toolchain =
        (pkgs.fenix.fromManifestFile inputs.rust-manifest).withComponents [
          "rustfmt"
          "rust-src"
          "clippy"
          "rustc"
          "cargo"
        ];
    in with pkgs; {
      devShells.${system}.default = mkShell {
        nativeBuildInputs =
          [ pkg-config cmake rust-toolchain rust-analyzer-nightly cargo-bloat ];
        buildInputs = [ systemdLibs linuxHeaders openvr xorg.libxcb ];
        shellHook = ''
          export LD_LIBRARY_PATH="${
            lib.makeLibraryPath [
              systemdLibs
              libglvnd
              vulkan-loader
              util-linux
              shaderc
              xorg.libX11
              xorg.libXcursor
              xorg.libXi
              libxkbcommon
              openxr-loader
              gcc.cc.lib
            ]
          }:$LD_LIBRARY_PATH"
        '';
        LIBCLANG_PATH = lib.makeLibraryPath [ llvmPackages_17.libclang.lib ];
        SHADERC_LIB_DIR = "${lib.getLib shaderc}/lib";

        BINDGEN_EXTRA_CLANG_ARGS =
          # Includes with normal include path
          (builtins.map (a: ''-I"${a}/include"'') [
            # add dev libraries here (e.g. pkgs.libvmi.dev)
            linuxHeaders
          ]) ++ [
            ''
              -isystem "${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${
                lib.versions.major pkgs.llvmPackages_latest.libclang.version
              }/include"''
            ''
              -isystem "${stdenv.cc.cc}/include/c++/${
                lib.getVersion stdenv.cc.cc
              }"''
            ''
              -isystem "${stdenv.cc.cc}/include/c++/${
                lib.getVersion stdenv.cc.cc
              }/${stdenv.hostPlatform.config}"''
            ''-isystem "${glibc.dev}/include"''
          ];
      };
    };
}
