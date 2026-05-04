with import <nixpkgs> {};

stdenv.mkDerivation {
  name = "xr_passthrough_layer";
  src = ./.;
  nativeBuildInputs = [
    pkg-config
    cmake
    rust-toolchain
    rust-analyzer
    cargo-bloat
    shaderc
  ];
  buildInputs = [
    systemdLibs
    linuxHeaders
    openvr
    xorg.libxcb
  ];
  shellHook = ''
    export LD_LIBRARY_PATH="${
      lib.makeLibraryPath [
        systemdLibs
        libglvnd
        vulkan-loader
        util-linux
        shaderc
        libuvc
        SDL2
        opencv
        openvr
        xorg.libX11
        xorg.libXcursor
        xorg.libXi
        libxkbcommon
        openxr-loader
        gcc.cc.lib
      ]
    }:$LD_LIBRARY_PATH"
  '';
  LIBCLANG_PATH = lib.makeLibraryPath [ llvmPackages_21.libclang.lib ];
  SHADERC_LIB_DIR = "${lib.getLib shaderc}/lib";
  BINDGEN_EXTRA_CLANG_ARGS =
    # Includes with normal include path
    (builtins.map (a: ''-I"${a}/include"'') [
      # add dev libraries here (e.g. pkgs.libvmi.dev)
      linuxHeaders
    ])
    ++ [
      ''-isystem "${pkgs.llvmPackages_latest.libclang.lib}/lib/clang/${lib.versions.major pkgs.llvmPackages_latest.libclang.version}/include"''
      ''-isystem "${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}"''
      ''-isystem "${stdenv.cc.cc}/include/c++/${lib.getVersion stdenv.cc.cc}/${stdenv.hostPlatform.config}"''
      ''-isystem "${glibc.dev}/include"''
    ];
}
