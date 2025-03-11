# Git Maintenance Chores

This is a running log of various chores that may need to be carried out to
manage the project sources and upstream connections.

## Rebase Sub-Projects (happy path)

If there are no problematic local patches that conflict with upstream projects,
then this command is sufficient to fast-forward all submodules to upstream
heads:

```
./build_tools/fetch_sources.py --remote
```

If that fails, it will most likely be when applying patches and you will get
a message indicating that the `git am` command failed and left some state
in a submodule. When this happens, you should clean up by going into the
submodule, looking around and aborting the command. Then proceed to the
next section for rebasing with conflicts:

```
git am --abort
```

You can return the source tree to a consistent state by running (without 
arguments):

```
./build_tools/fetch_sources.py
```

# Rebase Sub-Projects (conflicts)

If you have or suspect patch conflicts when doing a normal `--remote` update
of submodules, there are multiple procedures that can be effective. This
section provides a starting point. Actually deciding what to do with conflicts
is a case by case activity.

First, update to a pristine state with all submodule pointers pointing to their
remote heads (without applying local patches):

```
./build_tools/fetch_sources.py --remote --no-apply-patches
# Capture new submodule heads, inspect and ensure things look sound.
git add -A
git commit -m "Rebase submodules (for conflict prep)"
```

In the above, it is important to capture the submodule heads in a pristine
state, prior to any patches being applied. We'll squash all of the commits to
land.

