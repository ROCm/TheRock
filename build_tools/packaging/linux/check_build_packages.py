## Create a list of packages from package.json
## Exclude disabled packages
## Compare packages.txt with list of packages from package.json

import json
import re

JSON_FILE = "package.json"
VERSION_FILE = "../../../version.json"
GRAPHICS_SUFFIX = "-gfx"
BUILD_OUTPUT = "Packages.txt"

class Parser:
    def __init__(self, json_path, version_path):
        self.json_path = json_path
        self.version_path = version_path
        self.enabled_packages_dict = {}
        self.build_dict = {}

        self.enabled_pkg_json_set = set()
        self.build_set = set()
        
        # store content of json 
        self.json_content = self.load_json_file(self.json_path)

        # get rocm-version 
        self.rocm_version = self.find_rocm_version(self.version_path)


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


    def find_rocm_version(self, filepath):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
                ver = data.get('rocm-version')
                # remove last 2 characters i.e 7.11.0 -> 7.11
                return ver[:-2]
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None


    # TODO add logic:
    # rpm is devel 
    # deb is dev 
    def get_enabled_packages(self):
        # need to auto generate expected gfx version numbers 
        # after building this info will be provided 
        gfx_num_arr = ['1150', '1151', '120x', '94x', '950'] 
        
        self.enabled_packages_dict = {}
        
        for pkg in self.json_content:
            if pkg.get('DisablePackaging') != 'True':
                name = pkg.get('Package')

                if pkg.get('Gfxarch') == 'True':
                    for num in gfx_num_arr:
                        name_gfx = ''.join([name, GRAPHICS_SUFFIX, num])
                        #self.enabled_packages_dict[name_gfx] = name_gfx
                        self.enabled_pkg_json_set.add(name_gfx)

                    for num in gfx_num_arr:
                        name_gfx = ''.join([name, self.rocm_version, GRAPHICS_SUFFIX, num])
                        #self.enabled_packages_dict[name_gfx] = name_gfx
                        self.enabled_pkg_json_set.add(name_gfx)

                else:
                    self.enabled_packages_dict[name] = name


    def find_missing_packages(self, build_ouput_path):
        # Open and read the file
        with open(build_ouput_path, 'r') as file:
            lines = file.readlines()

        # Pattern to match everything before first _ or ~ 
        pattern = r'^([^_~]+)'

        # Process each line
        for line in lines:
            line = line.strip()  # Remove whitespace/newlines
            if line:  # Skip empty lines
                match = re.match(pattern, line)
                if match:
                    package_name = match.group(1)
                    #self.build_dict[package_name] = package_name
                    self.build_set.add(package_name)
    
    def missing_packages(self):
        return self.enabled_pkg_json_set.difference(self.build_set)

# if everything built can be installed pass 
# if not being built then give warning         
def main():
    json_parser = Parser(JSON_FILE, VERSION_FILE)
    json_parser.get_enabled_packages()
    json_parser.find_missing_packages(BUILD_OUTPUT)


main()

