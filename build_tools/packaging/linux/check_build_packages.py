## Create a list of packages from package.json
## Exclude disabled packages
## Compare packages.txt with list of packages from package.json

import json
import re

JSON_FILE = "package.json"
BUILD_OUTPUT = "built_packages.txt"
SKIPPED_OUTPUT = "skipped_packages.txt"

class Parser:
    def __init__(self, json_path):
        self.json_path = json_path
        self.version_path = None
        self.pkg_type = None
        self.gfx = []

        self.enabled_pkg_json_set = set()
        self.build_set = set()
        self.skipped_set = set()
        
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

    
    def missing_packages(self):
        print("\nMissing in build_set")
        return self.enabled_pkg_json_set.difference(self.build_set)

# if everything built can be installed pass 
# if not being built then give warning         
def main():
    json_parser = Parser(JSON_FILE)
    json_parser.parse_skipped_packages(SKIPPED_OUTPUT)
    json_parser.parse_build_packages(BUILD_OUTPUT)
    json_parser.get_enabled_packages()

    print(json_parser.missing_packages())
    print("\nEnabled set:")
    print(json_parser.enabled_pkg_json_set)

    print("\nSkipped set:")
    print(json_parser.skipped_set)
    
    print("\nbuild set:")
    print(json_parser.build_set)

    print()


main()

