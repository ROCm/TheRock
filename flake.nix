{
  description = "TheRock (The HIP Environment and ROCm Kit)";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.xz";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flake-root.url = "github:srid/flake-root";
  };

  outputs =
    inputs@{ flake-parts, flake-root, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } rec {
      imports = [
        flake-root.flakeModule
      ];
      systems = [
        "x86_64-linux"
        "x86_64-darwin"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      flake = rec {
        # TODO: currently we assume "nightly" releases for venv rocm packages,
        # but we could expose every "channel"
        families = [
          "gfx101X-dgpu"
          "gfx103X-all"
          "gfx103X-dgpu"
          "gfx110X-all"
          "gfx110X-dgpu"
          "gfx1150"
          "gfx1151"
          "gfx1152"
          "gfx1153"
          "gfx900"
          "gfx906"
          "gfx908"
          "gfx90a"
          "gfx120X-all"
          "gfx90X-dcgpu"
          "gfx94X-dcgpu"
          "gfx950-dcgpu"
        ];
        forAllFamilies = f: inputs.nixpkgs.lib.genAttrs families (family: f family);
      };
      perSystem =
        { config, pkgs, ... }:
        let
          mkFamilyShell =
            family:
            let
              venvPath = if family == "" then ".venv" else ".venv.${family}";
              ifFamily = s: if family == "" then "" else s;
            in
            pkgs.mkShellNoCC {
              inputsFrom = [ config.flake-root.devShell ];
              packages = with pkgs; [
                autoconf
                automake
                bison
                ccache
                cmake
                dvc
                flex
                gfortran # also includes g++, etc.
                git
                libtool
                ncurses # just for libtinfo, since therock builds its own ncurses
                ninja
                pkg-config
                python3
                texinfo
                (stdenv.mkDerivation rec {
                  pname = "patchelf-rocm";
                  version = "d0f70eea5397606c486857e0a105e53ec123904a";

                  src = fetchGit {
                    url = "https://github.com/NixOS/${pname}";
                    rev = "${version}";
                  };

                  patchPhase = ''
                    PATCHELF_GIT_REF="${version}"
                    SHORT_GIT_REF="''${PATCHELF_GIT_REF:0:12}"
                    BASE_VERSION="$(cat version)"
                    LOCAL_VERSION="''${BASE_VERSION}+therock.''${SHORT_GIT_REF}"
                    printf "%s\n" "''${LOCAL_VERSION}" > version
                  '';

                  nativeBuildInputs = [ autoreconfHook ];
                })
                (pkgs.writeShellApplication {
                  name = "therock-update-python";
                  text = ''
                    set -x
                    cd "$FLAKE_ROOT"
                    pip install --upgrade pip
                    pip install --upgrade -r requirements.txt
                    ${ifFamily "pip install --upgrade 'rocm[libraries,devel]' --index-url=https://rocm.nightlies.amd.com/v2/${family}"}
                  '';
                })
                (pkgs.writeShellApplication {
                  name = "therock-fetch-sources";
                  text = ''
                    set -x
                    cd "$FLAKE_ROOT"
                    python3 ./build_tools/fetch_sources.py
                  '';
                })
              ];
              shellHook = ''
                cd "$FLAKE_ROOT"
                if [ ! -d ${venvPath} ]; then
                  printf "[shell_hook] Creating ${venvPath}\n"
                  python3 ./build_tools/setup_venv.py ${venvPath} \
                  ${ifFamily "--packages 'rocm[libraries,devel]' --index-name nightly --index-subdir ${family}"}
                fi
                printf "[shell_hook] Activating ${venvPath}\n"
                source ${venvPath}/bin/activate
                if [ ! -d .ccache ]; then
                  printf "[shell_hook] Creating .ccache\n"
                  eval "$(python3 ./build_tools/setup_ccache.py)"
                else
                  printf "[shell_hook] Activating .ccache\n"
                  export CCACHE_CONFIGPATH="$PWD"/.ccache/ccache.conf
                fi
                printf "[shell_hook] Helper commands available:\n"
                compgen -c therock- | sed 's/^/\t/'
              '';
            };
        in
        {
          # This flake just defines devShells, so that `nix develop .` and `nix
          # develop .#<family>` work, but we could also expose package(s).
          # Upstream nixpkgs already maintains rocm-modules, so we would be
          # duplicating much of that.
          devShells = flake.forAllFamilies (family: mkFamilyShell family) // {
            default = mkFamilyShell "";
          };
          formatter = pkgs.nixfmt-tree;
        };
    };
}
