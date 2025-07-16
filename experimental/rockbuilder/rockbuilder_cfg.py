#!/usr/bin/env python

# curses works both on linux and windows
#   linux: pip install curses
#   windows: pip install windows-curses
import configparser
import curses
import os
import sys
from pathlib import Path, PurePosixPath

const__str_rocm_sdk_whl_server_url = "rocm_sdk_whl_server_url"


def get_rocm_builder_root_dir():
    current_file_path = os.path.abspath(__file__)
    ret = Path(os.path.dirname(current_file_path)).resolve()
    return ret


# Basic heuristic vefication to check whether rocm_sdk directory looks valid
#
# return True if path is valid
def is_valid_rocm_home_path(rocm_home_path):
    rocm_home_bin = rocm_home_path / "bin"
    rocm_home_lib = rocm_home_path / "lib"
    ret = rocm_home_bin.is_dir() and rocm_home_lib.is_dir()
    return ret


# Check whether the ROCM_HOME env variable is defined
# and whether it points to existing SDK installation.
#
# return either Path to ROCM_SDK or None
def get_rocm_home_path_if_available():
    ret = None
    if "ROCM_HOME" in os.environ:
        rocm_home = Path(os.environ["ROCM_HOME"]).resolve()
        if is_valid_rocm_home_path(rocm_home):
            ret = rocm_home
    return ret


# get rocm_sdk directory that would be procuded by therock build
def get_therock_rocm_sdk_build_dir():
    current_file_path = os.path.abspath(__file__)
    ret = os.path.dirname(current_file_path)
    ret = ret + "../../../build/dist/rocm"
    ret = Path(ret).resolve()
    return ret


# Check whether the ROCM_SDK is already build and available in default path
#
# return either Path to ROCM_SDK or None
def get_local_rocm_sdk_path_if_available():
    ret = None
    rocm_home = get_therock_rocm_sdk_build_dir()
    if is_valid_rocm_home_path(rocm_home):
        ret = rocm_home
    return ret


class SelectionItem:
    def __init__(self, name, key, value, selected):
        self.name = name
        self.key = key
        self.value = value
        self.selected = selected

    def is_selected(self):
        return self.selected

    def set_selected(self, selection):
        self.selected = selection

    def toggle_selected(self):
        self.selected = not self.selected

    def get_name(self):
        return self.name

    def get_key(self):
        return self.key

    def get_value(self):
        return self.value


class ConfigSelection:
    def __init__(self, header, selection_dict):
        self.header = header
        self.selection_dict = selection_dict


class BaseSelectionList:
    def __init__(self, stdscr, config_header, title, multi_selection):
        # todo, find out how to generate the list of supported GPUs
        self.title = title
        self.config_header = config_header
        self.stdscr = stdscr
        # whether to allow selecting multiple items
        self.multi_selection = multi_selection
        self.item_list = []
        self.item_selection_listeners = []

    def add_item_selection_listener(self, new_listener):
        self.item_selection_listeners.append(new_listener)

    def set_item_list(self, new_item_list):
        self.item_list.clear()
        self.item_list.extend(new_item_list)

    def get_item_cnt(self):
        return len(self.item_list)

    def get_config_header(self):
        return self.config_header

    def get_item(self, indx):
        return self.item_list[indx]

    def fire_item_selection_event(self, item, selected):
        for listener in self.item_selection_listeners:
            listener.handle_item_selected(self, item, selected)

    def set_multi_selection(self, enable):
        self.multi_selection = enable
        for ii, item in enumerate(self.item_list):
            if item.is_selected():
                self.toggle_item_selection(ii)
                break

    # handle the item selection logic
    def toggle_item_selection(self, indx):
        if self.multi_selection:
            # allow selecting 0-n items simultaneously
            item = self.item_list[indx]
            item.toggle_selected()
            new_state = item.is_selected()
            self.fire_item_selection_event(item, new_state)
        else:
            # allow selecting only one item at a time
            for ii, item in enumerate(self.item_list):
                if ii == indx:
                    item.set_selected(True)
                    # notify in this case only from the item selected
                    self.fire_item_selection_event(item, True)
                else:
                    item.set_selected(False)

    def show(self, indx_cursor, indx_first_item, indx_first_row):
        index_base = indx_first_row
        self.stdscr.addstr(index_base, 0, self.title)
        # display the list of item_list for the user to choose from
        index_base = index_base + 2
        for ii, gpu_item in enumerate(self.item_list):
            if indx_first_item + ii == indx_cursor:
                if gpu_item.is_selected():
                    self.stdscr.addstr(
                        index_base + ii, 0, f"> [X] {gpu_item.name}"
                    )  # cursor + selected
                else:
                    self.stdscr.addstr(
                        index_base + ii, 0, f"> [ ] {gpu_item.name}"
                    )  # cursor + not selected
            else:
                if gpu_item.is_selected():
                    self.stdscr.addstr(
                        index_base + ii, 0, f"  [X] {gpu_item.name}"
                    )  # selected
                else:
                    self.stdscr.addstr(
                        index_base + ii, 0, f"  [ ] {gpu_item.name}"
                    )  # not selected

    # get config selections
    #
    # - hrd contains the title for the selections. (section name in ini-file)
    # - selection_dict contains key-value pairs for the user selections
    # - each value stored to dictionary is itself actually an array of values
    #
    def get_config_selections(self):
        selection_dict = {}
        for ii, item in enumerate(self.item_list):
            if item.is_selected():
                val_arr = selection_dict.get(item.get_key())
                if not val_arr:
                    val_arr = []
                val_arr.append(item.get_value())
                selection_dict[item.get_key()] = val_arr
        section = self.get_config_header()
        return ConfigSelection(section, selection_dict)


class GpuSelectionList(BaseSelectionList):
    def __init__(self, stdscr):
        super().__init__(
            stdscr, "build_targets", f"Select target GPUs for the build", True
        )

    # Override the default selection logic because selection logic depends from the SDK selected
    #
    # - If we have selected PIP wheel install, we can not at the moment support multiple GPU-families at a same time
    # - If we select the local SDK, we can have multiple GPU's selected.
    # def toggle_item_selection(self, indx):
    #    for ii, item in enumerate(self.item_list):
    #        if ii == indx:
    #            item.set_selected(True)
    #        else:
    #            item.set_selected(False)


# Show list of possible ROCM_SDK's from which user can use for the build
#
# Items that does not exist will not be shown
# 1) Local ROCM_SDK specified by the ROCM_HOME environment variable if it exist
# 2) Local ROCM SDK build to directory build/dist/rocm if it exist
# 3) ROCM_SDK
class SDKSelectionList(BaseSelectionList):
    def __init__(self, stdscr):
        super().__init__(
            stdscr, "rocm_sdk", f"Select ROCM SDK used for the build", False
        )

        def_sel = True
        whl_server_base_url = "https://d2awnip2yjpvqn.cloudfront.net/v2/"
        rocm_home = get_rocm_home_path_if_available()
        if rocm_home:
            # add rocm home to list of SDK's to select
            self.item_list.append(
                SelectionItem(
                    "Use local ROCm SDK specified by the ROCM_HOME: "
                    + rocm_home.as_posix(),
                    "rocm_sdk_dir",
                    rocm_home.as_posix(),
                    def_sel,
                )
            )
            def_sel = False
        rocm_home = get_local_rocm_sdk_path_if_available()
        if rocm_home:
            # add an option/selection to use the rocm sdk that has been build locally
            self.item_list.append(
                SelectionItem(
                    "Use local ROCm SDK: " + rocm_home.as_posix(),
                    "rocm_sdk_dir",
                    rocm_home.as_posix(),
                    def_sel,
                )
            )
            def_sel = False
        else:
            # add an option/selection to build the rocm sdk locally
            rocm_home = get_therock_rocm_sdk_build_dir()
            self.item_list.append(
                SelectionItem(
                    "Build TheRock ROCm SDK: " + rocm_home.as_posix(),
                    "rocm_sdk_build",
                    rocm_home.as_posix(),
                    def_sel,
                )
            )
            def_sel = False
        # add an option/selection to use the rocm sdk that will be installed from the python wheel
        self.item_list.append(
            SelectionItem(
                "Use ROCm SDK PIP install from Server: " + whl_server_base_url,
                const__str_rocm_sdk_whl_server_url,
                whl_server_base_url,
                def_sel,
            )
        )

    # Override the default selection logic because we should only allow
    # one SDK to be selected at a time.
    # When one SDK location is selected, previous selections are disabled.
    # def toggle_item_selection(self, indx):
    #    for ii, item in enumerate(self.item_list):
    #        if ii == indx:
    #            item.set_selected(True)
    #        else:
    #            item.set_selected(False)


class SelectionListManager:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.selection_list_arr = []

    def add_selection_list(self, new_list):
        self.selection_list_arr.append(new_list)

    def get_total_selection_list_item_cnt(self):
        ret = 0
        for ii, selection_list in enumerate(self.selection_list_arr):
            ret = ret + selection_list.get_item_cnt()
        return ret

    def get_last_row_indx(self):
        ret = 0
        for ii, selection_list in enumerate(self.selection_list_arr):
            ret = ret + selection_list.get_item_cnt() + 3
        return ret

    def show(self, indx_cursor):
        cnt_items_viewed = 0
        indx_first_row = 0
        for ii, selection_list in enumerate(self.selection_list_arr):
            selection_list.show(indx_cursor, cnt_items_viewed, indx_first_row)
            cnt_items_viewed = cnt_items_viewed + selection_list.get_item_cnt()
            indx_first_row = indx_first_row + selection_list.get_item_cnt() + 3

    def on_selection_key_pressed(self, indx_cursor):
        indx_first_item = 0
        indx_last_item = 0
        for ii, selection_list in enumerate(self.selection_list_arr):
            indx_last_item = indx_last_item + selection_list.get_item_cnt()
            if (indx_cursor >= indx_first_item) and (indx_cursor < indx_last_item):
                selection_list.toggle_item_selection(indx_cursor - indx_first_item)
            indx_first_item = indx_last_item

    def save_selection(self):
        config = configparser.ConfigParser()
        # add sections and options
        for ii, selection_list in enumerate(self.selection_list_arr):
            config_value = selection_list.get_config_selections()
            section = config_value.header
            config.add_section(section)
            # get dictionary storing key/value pairs saved under section
            cfg_dict = config_value.selection_dict
            for ii, new_key in enumerate(cfg_dict.keys()):
                new_val = cfg_dict[new_key]
                config[section][new_key] = str(new_val)
        # save the configuration to a file
        fname = get_rocm_builder_root_dir()
        fname = fname / "rockbuilder.ini"
        with open(fname.as_posix(), "w") as configfile:
            config.write(configfile)


class UiManager:
    def __init__(self, stdscr):
        key_name_gpus = "gpus"
        self.stdscr = stdscr
        # init curses based display to show text based ui
        self.stdscr.clear()

        self.gpu_pip_wheel_list = []
        self.gpu_pip_wheel_list.append(
            SelectionItem("gfx110X-dgpu", key_name_gpus, "gfx110X-dgpu", False)
        )
        self.gpu_pip_wheel_list.append(SelectionItem("gfx1151", key_name_gpus, "gfx1151", False))
        self.gpu_pip_wheel_list.append(
            SelectionItem("gfx120X-all", key_name_gpus, "gfx120X-all", True)
        )
        self.gpu_pip_wheel_list.append(
            SelectionItem("gfx94X-dcgpu", key_name_gpus, "gfx94X-dcgpu", False)
        )
        self.gpu_pip_wheel_list.append(
            SelectionItem("gfx950-dcgpu", key_name_gpus, "gfx950-dcgpu", False)
        )

        self.gpu_build_target_list = []
        self.gpu_build_target_list.append(SelectionItem("gfx90a", key_name_gpus, "gfx90a", False))
        self.gpu_build_target_list.append(SelectionItem("gfx942", key_name_gpus, "gfx942", False))
        self.gpu_build_target_list.append(SelectionItem("gfx950", key_name_gpus, "gfx950", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1100", key_name_gpus, "gfx1100", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1101", key_name_gpus, "gfx1101", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1102", key_name_gpus, "gfx1102", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1103", key_name_gpus, "gfx1103", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1150", key_name_gpus, "gfx1150", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1151", key_name_gpus, "gfx1151", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1200", key_name_gpus, "gfx1200", False))
        self.gpu_build_target_list.append(SelectionItem("gfx1201", key_name_gpus, "gfx1201", False))

        self.sdk_list = SDKSelectionList(stdscr)
        self.gpu_list = GpuSelectionList(stdscr)
        self.gpu_list.set_item_list(self.gpu_build_target_list)

    def show(self):
        self.sdk_list.add_item_selection_listener(self)

        list_mngr = SelectionListManager(self.stdscr)
        list_mngr.add_selection_list(self.sdk_list)
        list_mngr.add_selection_list(self.gpu_list)

        indx_cursor = 0
        while True:
            list_mngr.show(indx_cursor)
            total_item_cnt = list_mngr.get_total_selection_list_item_cnt()
            last_row_indx = list_mngr.get_last_row_indx()
            self.stdscr.addstr(
                last_row_indx,
                0,
                f"Keys: Up, Down, Space, Enter and Esc",
            )
            self.stdscr.refresh()
            # Get user input for navigation or selection
            key = self.stdscr.getch()
            if key == curses.KEY_UP:
                indx_cursor = (indx_cursor - 1) % total_item_cnt
            elif key == curses.KEY_DOWN:
                indx_cursor = (indx_cursor + 1) % total_item_cnt
            elif key == ord(" "):
                list_mngr.on_selection_key_pressed(indx_cursor)
            elif key == curses.KEY_ENTER or key in [10, 13]:  # Handle Enter key
                list_mngr.save_selection()
                break  # Exit the selection process
            elif key == 27:  # ESC-Key
                break

    def handle_item_selected(self, sender, item, selected):
        key = item.get_key()
        if key == const__str_rocm_sdk_whl_server_url:
            self.stdscr.clear()
            self.gpu_list.set_item_list(self.gpu_pip_wheel_list)
            self.gpu_list.set_multi_selection(False)
        else:
            self.stdscr.clear()
            self.gpu_list.set_item_list(self.gpu_build_target_list)
            self.gpu_list.set_multi_selection(True)


def main(stdscr):
    ui_manager = UiManager(stdscr)
    ui_manager.show()


# execute the curses application
curses.wrapper(main)
