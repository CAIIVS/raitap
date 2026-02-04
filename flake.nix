{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pkgsCuda = import nixpkgs {
          inherit system;
          config.cudaSupport = true;
          config.allowUnfree = true;
        };

        # Create a merged ROCm directory with all necessary includes
        rocmMerged = with pkgs; symlinkJoin {
          name = "rocm-merged";
          paths = with rocmPackages; [
            clr
            rocthrust
            rocprim
            hipsparse
            hipblas
            hipblaslt
            hipblas-common
            hipsolver
          ];
        };

        # Create a merged CUDA directory
        cudaMerged = with pkgs; symlinkJoin {
          name = "cuda-merged";
          paths = [
            cudaPackages.cuda_cudart
            cudaPackages.cuda_nvcc
            cudaPackages.libcublas
            cudaPackages.libcusparse
            cudaPackages.libcusolver
          ];
        };

        # Common build inputs for all shells
        commonBuildInputs = with pkgs; [
          stdenv.cc.cc.lib
          glib
          zlib
          libGL
          fontconfig
          xorg.libX11
          libxkbcommon
          freetype
          dbus
          libsForQt5.wrapQtAppsHook
        ];

        # Python environments for CPU-only (fast, uses binary cache)
        makeCpuEnvsForPython = dist:
          dist.withPackages (p:
            let
              mpl = (p.matplotlib.override { enableQt = true; });
            in 
            with p; [
              numpy
              safetensors
              mpl
              torch
              torchvision
              scikit-learn
              hydra-core
              mlflow.override { matplotlib = mlp; }
              shap
            ]);

        # Python environments for ROCm
        makeRocmEnvsForPython = dist:
          dist.withPackages (p:
            let
              mpl = (p.matplotlib.override { enableQt = true; });
            in 
            with p; [
              numpy
              safetensors
              mpl
              torchWithRocm
              (torchvision.override { torch = torchWithRocm; })
              scikit-learn
              hydra-core
              mlflow.override { matplotlib = mlp; }
              shap
            ]);

        # Python environments for CUDA
        makeCudaEnvsForPython = dist:
          dist.withPackages (p:
            let
              mpl = (p.matplotlib.override { enableQt = true; });
            in 
            with p; [
              numpy
              safetensors
              mpl
              torchWithCuda
              (torchvision.override { torch = torchWithCuda; })
              scikit-learn
              hydra-core
              mlflow.override { matplotlib = mlp; }
              shap
            ]);

        pyDists = with pkgs; [
          python313
        ];
        cpuPyEnvs = map makeCpuEnvsForPython pyDists;
        rocmPyEnvs = map makeRocmEnvsForPython pyDists;
        cudaPyEnvs = map makeCudaEnvsForPython pyDists;
      in {
        devShells = {
          # Default to CUDA (more common)
          default = pkgs.mkShell {
            packages =
              [
                pkgs.uv
              ]
              ++ pyDists ++ cpuPyEnvs;

            buildInputs = commonBuildInputs;

            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system";
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
                stdenv.cc.cc.lib
                zlib
                libGL
              ])}:$LD_LIBRARY_PATH";
              echo "🖥️  CPU-only environment loaded (fast, no GPU support)"
              echo "   For GPU support use: nix develop .#cuda or .#rocm"
            '';
          };

          # CPU-only shell (explicit)
          cpu = pkgs.mkShell {
            packages =
              [
                pkgs.uv
              ]
              ++ pyDists ++ cpuPyEnvs;

            buildInputs = commonBuildInputs;

            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system";
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (with pkgs; [
                stdenv.cc.cc.lib
                zlib
                libGL
              ])}:$LD_LIBRARY_PATH";
              echo "🖥️  CPU-only environment loaded"
            '';
          };

          # CUDA shell
          cuda = pkgs.mkShell {
            packages =
              [
                pkgsCuda.uv
                cudaMerged
              ]
              ++ (with pkgsCuda; [python313]) ++ cudaPyEnvs;

            buildInputs = commonBuildInputs;

            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system";
              export CUDA_PATH="${cudaMerged}";
              export CUDA_HOME="${cudaMerged}";
              export CPLUS_INCLUDE_PATH="${cudaMerged}/include:''${CPLUS_INCLUDE_PATH:-}";
              export C_INCLUDE_PATH="${cudaMerged}/include:''${C_INCLUDE_PATH:-}";
              export LD_LIBRARY_PATH="${cudaMerged}/lib:${pkgsCuda.lib.makeLibraryPath (with pkgsCuda; [
                stdenv.cc.cc.lib
                zlib
                libGL
              ])}:$LD_LIBRARY_PATH";
            '';
          };

          # ROCm shell (for AMD GPUs)
          rocm = pkgs.mkShell {
            packages =
              [
                pkgs.uv
                rocmMerged
              ]
              ++ pyDists ++ rocmPyEnvs;

            buildInputs = with pkgs; [
              stdenv.cc.cc.lib
              glib
              zlib
              libGL
              fontconfig
              xorg.libX11
              libxkbcommon
              freetype
              dbus
              libsForQt5.wrapQtAppsHook
            ];

            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system";
              export ROCM_PATH="${rocmMerged}";
              export HIP_PATH="${rocmMerged}";
              export CPLUS_INCLUDE_PATH="${rocmMerged}/include:$CPLUS_INCLUDE_PATH";
              export C_INCLUDE_PATH="${rocmMerged}/include:$C_INCLUDE_PATH";
              export LD_LIBRARY_PATH="${rocmMerged}/lib:${pkgs.lib.makeLibraryPath (with pkgs; [
                stdenv.cc.cc.lib
                zlib
                libGL
              ])}:$LD_LIBRARY_PATH";
            '';
          };
        };
      }
    );
}
