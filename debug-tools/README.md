# Debug Tools

This is the home of the TheRock's build config for ROCm debug tools, which
include ROCgdb, ROCdbgapi, and the ROCr debug agent.

The source code for ROCdbgapi and the ROCr debug agent may eventually be
migrated to the rocm-systems super-repo.

## Structure

The debug tools are organized as follows:

```
amd-dbgapi: The ROCdbgapi source code.
rocgdb/source: The ROCgdb source code.
rocr-debug-agent: The ROCr debug agent source code.
```

## Additional information

### ROCgdb dependency on terminfo for TUI mode

ROCgdb’s TUI (Text User Interface) mode uses ncurses. This library relies on
finding a valid terminfo database to function properly.

TheRock builds its own ncurses library, which includes the terminfo
database. However, because ncurses does not provide a way to specify a
relative path to the database at configure/build time, it is important to
understand how the terminfo lookup works to ensure TUI mode remains functional.

If ROCgdb is launched via its launcher shell script (bin/rocgdb), the script
automatically points to the database using the TERMINFO environment variable.

If the ROCgdb binary is invoked directly and TERMINFO is not set, the terminfo
lookup logic checks the following paths in order:

- `$HOME/.terminfo`
- `/usr/share/terminfo`
- `/usr/lib/terminfo`
- `/etc/terminfo`
- `/lib/terminfo`
- The build-time prefix path (which may no longer be accessible)

If the lookup fails to return any valid entries, ncurses provides the following
terminal types as compiled-in fallbacks:

- xterm / xterm-256color
- vt100
- linux
- screen / screen-256color
- tmux / tmux-256color
- ansi

If the user's system terminal type does not match any of these fallbacks, the
user must install a terminfo dependency in one of the lookup paths listed
above. Otherwise, the TUI mode in ROCgdb will remain unavailable.
