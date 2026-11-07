from importlib import import_module
from typing import Literal, Any, overload, Union
from textwrap import dedent
from abc import ABC, abstractmethod

from devices import WindowsNT, ManyLinux
from utils import *
from utils.status import _SuccessType, _HintType, _WarningType, _FatalType


ALL_TASKS: list[str] = ["TheRock", "PyTorch", "JAX", "TensorFlow"]
ALL_TASKS_EXAMINE_NAME = [tsk.upper() for tsk in ALL_TASKS]


class TaskManager:

    os = os_type()

    def __init__(self):

        from . import TheRock, PyTorch

        task = config.DEFAULT_TASK

        if task not in ALL_TASKS:
            message("FATAL_ERROR", f"Recieved Unknown build tasks {task}")

        task_module_pyfile = "task." + task.lower()

        try:
            module = import_module(task_module_pyfile)
            cls = getattr(module, task)
            self.task: Union[TheRock, PyTorch] = cls()
        except AttributeError as e:
            if task not in ALL_TASKS:
                message(
                    "ERROR",
                    dedent(
                        f"""\
                                        Specify build task is case sensitive.
                                            Avail cases: TheRock, PyTorch, JAX, TensorFlow, XGBoost, MXNet
                                            Your case: {task}"""
                    ),
                )
                message(
                    "FATAL_ERROR", f"    traceback: Found Invalid build case {task}"
                )
            elif task.upper() not in ALL_TASKS_EXAMINE_NAME:
                message(
                    "ERROR",
                    dedent(
                        f"""\
                                        Found not supported build task {task}.
                                        If building {task} should be supported, please contact to TheRock dev team for support."""
                    ),
                )
                message("FATAL_ERROR", f"Found not supported build task {task}.")
            else:
                message(
                    "FATAL_ERROR",
                    dedent(
                        f"""\
                                              Unknown Import Error. Please contact TheRock dev team to fix.
                                                  traceback: {e.with_traceback()}"""
                    ),
                )
        finally:
            pass

    def run(self, task: Literal["system", "post"], /):
        self.task.run(task)


class BuildTasks(ABC):

    os: Literal["Windows", "Linux", "BSD", "macOS"] = os_type()

    def __init__(self):

        self.device: Union[WindowsNT, ManyLinux]
        self.post_list: list[Any]

        if self.os == "Windows":
            self.device = WindowsNT()
        elif self.os == "Linux":
            self.device = ManyLinux()
        elif self.os == "BSD":
            ...
        elif self.os == "macOS":
            ...
        else:
            message("FATAL_ERROR", f"Unknown OS Type {self.os}")

        message("REPRINT", "")

    @abstractmethod
    def __WINDOWS__(
        self,
    ) -> dict[str, Union[_SuccessType, _HintType, _WarningType, _FatalType]]:
        device: WindowsNT
        ...

    @abstractmethod
    def __LINUX__(
        self,
    ) -> dict[str, Union[_SuccessType, _HintType, _WarningType, _FatalType]]:
        device: ManyLinux
        ...

    def result_summary_display(
        self,
        result: dict[str, Union[_SuccessType, _HintType, _WarningType, _FatalType]],
    ) -> None:

        results_list = [
            subitem
            for item in result.values()
            for subitem in (item.values() if isinstance(item, dict) else [item])
        ]

        total_success = results_list.count(SUCCESS)
        total_hint = results_list.count(HINT)
        total_warning = results_list.count(WARNING)
        total_error = results_list.count(FATAL)

        _1 = cstring(total_success, "SUCCESS")
        _2 = cstring(total_hint, "HINT")
        _3 = cstring(total_warning, "WARNING")
        _4 = cstring(total_error, "ERROR")

        message(
            f"\n\t===========       Compoments check: {_1} Success, {_2} Hint(s), {_3} Warning(s),  {_4} Fatal Error(s)      ==========="
        )

        if config.JSON_OUTPUT:
            self.result_summary_dumper(
                result=result,
                result_count=[total_success, total_hint, total_warning, total_error],
            )

    def result_summary_dumper(
        self,
        result: dict[str, Union[_SuccessType, _HintType, _WarningType, _FatalType]],
        result_count: list[int],
    ) -> None:

        import json, time

        localtime = time.localtime(time.time())
        fmttime = time.strftime("%Y-%m-%d_%H%M%S", localtime)

        filename = f"TheRock_EnvDiag_{self.os}_{config.DEFAULT_TASK}_{fmttime}.json"

        final_output = convert_paths_to_posix(
            {
                "Stats": {
                    "command": COMMAND_LINE,
                    "task": config.DEFAULT_TASK,
                    "Success": result_count[0],
                    "Hint": result_count[1],
                    "Warning": result_count[2],
                    "Error": result_count[3],
                    "timestamp": fmttime,
                    "gitstamp": githead,
                },
                "Device": self.device.__info__(),
                "Diagnose": result,
            }
        )

        with open(filename, "w", encoding="utf-8") as j:
            json.dump(final_output, j, indent=4, default=self.status_serializer)

        message(f"\n\t===========       generating: {filename}\t\t===========")

    @overload
    def run(self, mode: Literal["system"], /): ...
    @overload
    def run(self, mode: Literal["post"], /): ...

    def run(self, mode: Literal["system", "post"]):

        if mode == "system":
            if config.DISPLAY_SYSTEM_INFO:
                message(self.device)
        else:
            message("STATUS", f"The build task identification is")
            message(config.DEFAULT_TASK)
            message("")
            if not config.DISABLE_TEST:
                if self.os == "Windows":
                    result = self.__WINDOWS__()
                elif self.os == "Linux":
                    result = self.__LINUX__()

                if result is None:
                    message("FATAL_ERROR", f"Test dict returns a None value.")

                if None in result or Ellipsis in result:

                    message("FATAL_ERROR", "Found invalid None value in result list.")

                self.result_summary_display(result)

        return None

    def status_serializer(self, obj):
        if hasattr(obj, "__TYPE__"):
            return obj.__TYPE__
        else:
            return repr(obj)


from pathlib import Path


def convert_paths_to_posix(data):

    if isinstance(data, dict):
        return {k: convert_paths_to_posix(v) for k, v in data.items()}

    elif isinstance(data, list):
        return [convert_paths_to_posix(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_paths_to_posix(item) for item in data)

    elif isinstance(data, Path):
        return data.as_posix()

    else:
        return data
