#!/usr/bin/env python3

import json

def read_package_json_file():
    '''Reads a JSON file and returns the parsed data.'''
    with open("package.json", 'r') as file:
        data = json.load(file)
    return data


def is_packaging_disabled(package):
    '''
    Checks if packaging is disabled for a given package.

    Parameters:
    package (dict): A dictionary containing package details.

    Returns:
    bool: True if 'DisablePackaging' key exists, False otherwise.
    '''
    return "DisablePackaging" in package


def get_package_info(pkgname):
    '''Function to retrieve package details stored in a JSON file for the provided package name'''
    # Load JSON data from a file
    data = read_package_json_file()

    for package in data:
        if package.get("Package") == pkgname:
            return package
            break

    return None

def check_for_gfxarch(pkgname):
    '''The function will determine whether the gfxarch should be appended to the package name
       gfxarch is not required for Devel package''' 
    if pkgname.endswith("-devel"):
        return False

    pkg_info = get_package_info(pkgname)
    if str(pkg_info.get("Gfxarch", "false")).strip().lower() == "true":
        return True
    return False

def get_package_list():
    '''Read package.json and get the list of package names
       Exclude the package marked as Disablepackaging'''
    data = read_package_json_file()

    pkg_list = [pkg["Package"] for pkg in data if not is_packaging_disabled(pkg)]
    return pkg_list

def version_to_str(version_str):
    ''' Function will change rocm version to string
         Ex : 7.1.0 -> 70100'''

    major, minor, patch = version_str.split(".")
    return f"{int(major):01d}{int(minor):02d}{int(patch):02d}"


