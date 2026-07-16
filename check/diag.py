#!/usr/bin/env python3

import sys

sys.dont_write_bytecode = True

from utils.argparse_handler import args_update
from utils import *
from task import TaskManager


@tic_toc()
def main():

    args_update()
    clear()
    logo()

    taskmgr = TaskManager()
    taskmgr.run("system")
    taskmgr.run("post")


if __name__ == "__main__":
    main()
