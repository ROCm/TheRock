# Bash style guide

## Core principles

> [!WARNING]
> Bash is **strongly discouraged** for nontrivial usage in .yml GitHub Actions
> workflow files and script files.
>
> **Use Python scripts in most cases instead**.

Writing and maintaining safe and portable scripts in Bash is significantly
harder than it is in Python. When appropriate, we write Bash scripts following
some of the guidelines at https://google.github.io/styleguide/shellguide.html.

Those sections are particularly noteworthy:

- https://google.github.io/styleguide/shellguide.html#variable-expansion
- https://google.github.io/styleguide/shellguide.html#quoting

### When to use Bash

Use Bash for:

- Simple automation scripts with few conditionals
- Wrapping existing command-line tools
- Environment setup and configuration
- Quick one-off tasks

Avoid Bash for:

- Complex logic with loops and conditionals
- Data processing and transformation
- Anything that needs to be tested thoroughly
- Cross-platform scripts (prefer Python)

## Style guidelines

### Setting bash modes

Scripts should generally set modes like

```bash
set -euo pipefail
```

Benefits:

- **Fail-fast:** Exits immediately on errors
- **Safe variables:** Treats undefined variables as errors
- **Pipeline safety:** Fails if any command in a pipeline fails
- **Debuggability:** With `-x`, shows exactly what commands are running

Explanation of each mode:

- `set -e` Exits if any command has a non-zero exit status
- `set -u` Treats undefined variables as errors
- `set -o pipefail` Uses the return code of a failing command as the pipeline
  return code
- `set -x` Prints commands to the terminal (useful for debugging and CI logging)

See https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425 for
an explanation of what each of these options does.
