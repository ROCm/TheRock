#! /usr/bin/env python

import os
import subprocess

def get_output_for_cmd(cmd_str):
    return subprocess.run(cmd_str,shell=True,text=True,stdout=subprocess.PIPE).stdout

def recurse_git_diff(diffcmd):
    lines = list(map(str.rstrip,get_output_for_cmd(diffcmd).split('\n')))
    i=0
    while i<len(lines):
        diff_line = lines[i]
        if diff_line=='':
            break
        i+=1
        index_line = lines[i]
        if index_line.split(' ')[-1]=="160000":
            i+=5
            submodule_line = lines[i]
            submodule = diff_line.split(' ')[-1][2:]

            sha_dirty = submodule_line.split(' ')[-1]
            sha_dirty_parts = sha_dirty.split('-')
            print("pushd "+submodule)
            print("git reset --hard "+sha_dirty_parts[0])
            if len(sha_dirty_parts) > 1 and sha_dirty_parts[1]=="dirty":
                recurse_git_diff(submodule)
            print("popd")
            i+=1
        else:
            i+=1
            print("patch -p1 <<DIRTY_EOF")
            while i < len(lines) and lines[i].split(' ')[0]!="diff":
                print(lines[i])
                i+=1
            print("DIRTY_EOF")

print("git reset --hard "+get_output_for_cmd("git rev-parse HEAD~1").rstrip())
recurse_git_diff("git diff")
recurse_git_diff("git diff HEAD~1")
