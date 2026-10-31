import re
import os
from pathlib import Path
from typing import Iterable, Union, Dict


def cmake_variable_finder(
    file: Union[Path, str], hint: Iterable[str], output: Union[str, Iterable[str]]
) -> Dict[str, str]:
    file_path = Path(file).resolve()
    hint_set = set(hint)

    if output == "all":
        target_vars = hint_set
    else:
        output_set = {output} if isinstance(output, str) else set(output)
        invalid_vars = output_set - hint_set
        if invalid_vars:
            raise ValueError(
                f"Error: variables in output is not allowed in hint range -> {invalid_vars}"
            )
        target_vars = output_set

    state = {
        "CMAKE_CURRENT_LIST_FILE": file_path.as_posix(),
        "CMAKE_CURRENT_LIST_DIR": file_path.parent.as_posix(),
    }

    def resolve_value(val: str) -> str:
        for _ in range(15):
            if "${" not in val and "$ENV{" not in val:
                break
            val = re.sub(
                r"\$ENV\{([A-Za-z0-9_]+)\}",
                lambda m: os.environ.get(m.group(1), ""),
                val,
            )
            val = re.sub(
                r"\$\{([A-Za-z0-9_]+)\}", lambda m: state.get(m.group(1), ""), val
            )
        return val

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Cannot find file: {file_path}")
        return {}

    content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)

    command_pattern = re.compile(r"([A-Za-z0-9_]+)\s*\((.*?)\)", re.DOTALL)

    for match in command_pattern.finditer(content):
        cmd = match.group(1).lower()
        args_raw = match.group(2)

        args = []
        for m in re.finditer(r'"(?:\\.|[^"\\])*"|\S+', args_raw):
            val = m.group(0)
            val = val.strip("\"'")
            args.append(val)

        if not args:
            continue

        if cmd == "set":
            var_name = args[0]
            var_value = resolve_value(args[1]) if len(args) > 1 else ""
            state[var_name] = var_value

        elif cmd == "get_filename_component":
            if len(args) >= 3:
                var_name = args[0]
                file_name = resolve_value(args[1])
                mode = args[2].upper()

                p = Path(file_name)

                if mode in ("PATH", "DIRECTORY"):
                    state[var_name] = p.parent.as_posix()
                elif mode == "REALPATH":
                    state[var_name] = p.resolve().as_posix()
                elif mode == "ABSOLUTE":
                    state[var_name] = p.absolute().as_posix()
                elif mode == "NAME":
                    state[var_name] = p.name
                elif mode == "EXT":
                    state[var_name] = p.suffix
                elif mode == "NAME_WE":
                    state[var_name] = p.stem

    # 6. 整理最終要求的輸出結果
    results = {}
    for var in target_vars:
        if var.startswith("ENV{") and var.endswith("}"):
            env_key = var[4:-1]
            results[var] = os.environ.get(env_key, "")
        else:
            results[var] = state.get(var, None)

    return results
