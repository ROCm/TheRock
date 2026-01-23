## Create a list of packages from package.json
## Exclude disabled packages
## Compare packages.txt with list of packages from package.json

import json
import re
import os
import subprocess
import glob
from datetime import datetime


CURPATH = os.path.dirname(os.path.abspath(__file__))
print(CURPATH)
JSON_FILE = os.path.join(CURPATH, 'package.json')
BUILD_OUTPUT = os.path.join(CURPATH, 'built_packages.txt')
SKIPPED_OUTPUT = os.path.join(CURPATH, 'skipped_packages.txt')
INSTALL_DIR = os.path.join(CURPATH, '../../../install_test')

## todo, input for folder where to install packages
## by default use package.json can pass another json in cmd line

class Parser:
    def __init__(self, json_path):
        self.json_path = json_path
        self.version_path = None
        self.pkg_type = None
        self.gfx = []

        self.enabled_pkg_json_set = set()
        self.build_set = set()
        self.skipped_set = set()
        self.pkgs = []
        
        # store content of json 
        self.json_content = self.load_json_file(self.json_path)


    def load_json_file(self, filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                return data
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None


    def parse_skipped_packages(self, skipped_output_path):
        # Open and read the file
        with open(skipped_output_path, 'r') as file:
            lines = file.readlines()

        # Pattern to match all packages that start with amd 
        regex_pkg = r'(^amd\S+)'

        # ROCm Version: 7.11.0~20251224
        regex_rocm_ver = r'# ROCm Version:\s*(\d+\.\d+)\.\d+~\d+'


        # Process each line
        for line in lines:
            line = line.strip()  # Remove whitespace/newlines
            if line:  # Skip empty lines
                amd_pkg_name_match = re.match(regex_pkg, line)

                if amd_pkg_name_match:
                    package_name = amd_pkg_name_match.group(1)
                    self.skipped_set.add(package_name)
                

    def parse_build_packages(self, build_ouput_path):
        # Open and read the file
        with open(build_ouput_path, 'r') as file:
            lines = file.readlines()

        # Pattern to match all packages that start with amd 
        regex_pkg = r'(^amd\S+)'

        # find pkg type 
        regex_pkg_type = r'# Package Type:\s*(\w+)'

        # ROCm Version: 7.11.0~20251224
        regex_rocm_ver = r'# ROCm Version:\s*(\d+\.\d+)\.\d+~\d+'

        # Graphics Architecture
        regex_gfx = r'^# Graphics Architecture:\s*(.+)$'

        # Process each line
        for line in lines:
            line = line.strip()  # Remove whitespace/newlines
            if line:  # Skip empty lines
                amd_pkg_name_match = re.match(regex_pkg, line)
                pkg_type_match = re.match(regex_pkg_type, line)
                rocm_ver_match = re.match(regex_rocm_ver, line)
                gfx_match = re.match(regex_gfx, line)

                if amd_pkg_name_match:
                    package_name = amd_pkg_name_match.group(1)
                    self.build_set.add(package_name)
                
                if pkg_type_match:
                    self.pkg_type = pkg_type_match.group(1)
                
                if rocm_ver_match:
                    self.rocm_version = rocm_ver_match.group(1)
                
                if gfx_match:
                    arch_line = gfx_match.group(1)
                    # Extract gfx values
                    gfx_pattern = r'gfx\d+x?'
                    self.gfx = re.findall(gfx_pattern, arch_line)


    def get_enabled_packages(self):
        # need to auto generate expected gfx version numbers 
        # after building this info will be provided 
        
        self.enabled_packages_dict = {}
        
        for pkg in self.json_content:
            if pkg.get('DisablePackaging') != 'True':
                name = pkg.get('Package')

                if name in self.skipped_set:
                    continue

                # if package is deb rename devel suffix to dev by slicing last 2 chars out 
                if self.pkg_type == 'DEB' and name[-5:] == 'devel':
                    name = name[:-2]

                if pkg.get('Gfxarch') == 'True':
                    for gfx_suffix in self.gfx:

                        name_gfx = ''.join([name, '-', gfx_suffix])
                        self.enabled_pkg_json_set.add(name_gfx)

                    for gfx_suffix in self.gfx:
                        name_gfx = ''.join([name, self.rocm_version, '-', gfx_suffix])
                        self.enabled_pkg_json_set.add(name_gfx)

                else:
                    self.enabled_pkg_json_set.add(''.join([name, self.rocm_version]))

    
    def missing_pkg(self):
        miss_set = self.enabled_pkg_json_set.difference(self.build_set)

        print(f"\n{'='*60}")
        if len(miss_set) == 0:
            print("PASS")
            print("No missing packages")
        else:
            print("FAIL\n")
            print("Missing in build_set:")
            for p in miss_set:
                print(p)

        print(f"\n{'='*60}")
            


    def ensure_directory_exists(self, path):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory exists: {path}")
        return path


    def get_package_info(self, dir):
        """Detect package type and commands"""
        deb_path = os.path.join(dir, '*.deb')
        rpm_path = os.path.join(dir, '*.rpm')

        deb_files = glob.glob(deb_path)
        rpm_files = glob.glob(rpm_path)

        print(deb_path)
        
        packages = []
        
        if deb_files:
            packages.append({
                'type': 'DEB',
                'files': deb_files,
                'install_cmd': ['sudo', 'dpkg', '-i'],
                'fix_deps_cmd': ['sudo', 'apt-get', 'install', '-f', '-y']
            })
        
        if rpm_files:
            packages.append({
                'type': 'RPM',
                'files': rpm_files,
                'install_cmd': ['sudo', 'rpm', '-ivh'],
                'fix_deps_cmd': ['sudo', 'yum', 'install', '-y']
            })
        
        return packages

    def install_deb_files_with_logging(self, directory='.', log_file='install.log'):
        """Install DEB and RPM packages with detailed logging"""
        self.ensure_directory_exists(directory)

        os.chdir(directory)
        package_groups = self.get_package_info(directory)
        
        if not package_groups:
            print("No package files found")
            return
         
        total_successful = 0
        total_failed = 0
        
        with open(log_file, 'w') as log:
            log.write(f"Installation started: {datetime.now()}\n")
            log.write(f"Directory: {os.path.abspath(directory)}\n\n")
            
            for pkg_group in package_groups:
                pkg_type = pkg_group['type']
                pkg_files = pkg_group['files']
                install_cmd = pkg_group['install_cmd']
                fix_deps_cmd = pkg_group['fix_deps_cmd']
                
                log.write(f"\n{'='*60}\n")
                log.write(f"Installing {pkg_type} Packages\n")
                log.write(f"{'='*60}\n")
                log.write(f"Total {pkg_type} files: {len(pkg_files)}\n\n")
                
                print(f"\n{'='*60}")
                print(f"Installing {pkg_type} Packages")
                print(f"{'='*60}")
                print(f"Found {len(pkg_files)} {pkg_type} files\n")
                
                successful = []
                failed = []
                
                # Install each package
                for pkg_file in pkg_files:
                    filename = os.path.basename(pkg_file)
                    print(f"Installing: {filename}")
                    log.write(f"\n{'-'*60}\n")
                    log.write(f"Installing: {filename}\n")
                    log.write(f"{'-'*60}\n")
                    
                    try:
                        result = subprocess.run(
                            install_cmd + [pkg_file],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        print(f"  ✓ Success")
                        log.write(f"Status: SUCCESS\n")
                        log.write(f"Output:\n{result.stdout}\n")
                        successful.append(filename)
                        
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Failed")
                        log.write(f"Status: FAILED\n")
                        log.write(f"Error:\n{e.stderr}\n")
                        failed.append(filename)
                
                # Fix dependencies
                print(f"\nFixing {pkg_type} dependencies...")
                log.write(f"\n{'-'*60}\n")
                log.write(f"Fixing {pkg_type} dependencies\n")
                log.write(f"{'-'*60}\n")
                
                try:
                    result = subprocess.run(
                        fix_deps_cmd,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    log.write(f"Dependency fix output:\n{result.stdout}\n")
                    print(f"  ✓ Dependencies fixed")
                except subprocess.CalledProcessError as e:
                    log.write(f"Dependency fix error:\n{e.stderr}\n")
                    print(f"  ✗ Dependency fix failed")
                
                # Summary for this package type
                log.write(f"\n{pkg_type} Summary:\n")
                log.write(f"  Successful: {len(successful)}\n")
                log.write(f"  Failed: {len(failed)}\n")
                
                if failed:
                    log.write(f"\nFailed {pkg_type} packages:\n")
                    for pkg in failed:
                        log.write(f"  - {pkg}\n")
                
                total_successful += len(successful)
                total_failed += len(failed)
                
                # Print summary for this type
                print(f"\n{pkg_type} Summary:")
                print(f"  Successful: {len(successful)}")
                print(f"  Failed: {len(failed)}")
                
                if failed:
                    print(f"\nFailed {pkg_type} packages:")
                    for pkg in failed:
                        print(f"  - {pkg}")
            
            # Overall summary
            log.write(f"\n{'='*60}\n")
            log.write(f"OVERALL INSTALLATION SUMMARY\n")
            log.write(f"{'='*60}\n")
            log.write(f"Total successful: {total_successful}\n")
            log.write(f"Total failed: {total_failed}\n")
            log.write(f"\nCompleted: {datetime.now()}\n")
        
        # Print overall summary
        print(f"\n{'='*60}")


# if everything built can be installed pass 
# if not being built then give warning         
def main():
    json_parser = Parser(JSON_FILE)
    json_parser.parse_skipped_packages(SKIPPED_OUTPUT)
    json_parser.parse_build_packages(BUILD_OUTPUT)
    json_parser.get_enabled_packages()
    json_parser.missing_pkg()


    json_parser.install_deb_files_with_logging(INSTALL_DIR)



main()

