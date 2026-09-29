# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# therock_autotools.cmake
# Running the tools of a GNU autotools build from a TheRock sub-project.
#
# Several sub-projects build this way: the bundled autotools sysdeps, and
# rocgdb.  They do not agree on the steps.  A sysdep runs configure, make, make
# install, while rocgdb adds installs for its docs and for binutils, and
# interleaves plain cmake -E steps that must not go through any shell at all.
# What they share is the mechanics of running one command: which shell it needs,
# how configure has to be named, which compiler and host triple to name, and
# what environment to carry.
#
# Windows is where those mechanics diverge.  It offers no POSIX environment for
# a configure script, so commands go through an MSYS2 UCRT64 login shell, and
# configure has to be named by a relative path.  therock_autotools_shell()
# resolves all of that once and hands back command prefixes, leaving each recipe
# to spell out its own steps.
#
# The super-project includes this file as well, for
# therock_autotools_subproject_args(), which supplies the declare arguments
# such a sub-project needs.  Nothing else here runs at include time.

include_guard(GLOBAL)

if(WIN32)
  include("${THEROCK_SOURCE_DIR}/cmake/therock_msys2.cmake")
endif()

# therock_autotools_shell(
#   BINARY_DIR <dir>              Build directory.  Commands run here.
#   [SOURCE_DIR <dir>]            Directory holding configure, required with
#                                 CONFIGURE_VAR.
#   [PKG_CONFIG_PATH <dirs>]      A CMake ';'-list of pkg-config directories.
#                                 Joined and delivered in whatever way survives
#                                 on this host.
#   [CONFIGURE_ENV <VAR=VALUE>...]  Extra environment, configure only.  A CC
#                                 or CXX here replaces the compiler this
#                                 function would name for that language.
#   [RUN_VAR <var>]             Out: prefix running an argv.
#   [RUN_CONFIGURE_VAR <var>]     Out: the same, plus the compiler and
#                                 CONFIGURE_ENV.
#   [CONFIGURE_VAR <var>]         Out: how to name configure.  A list on
#                                 Windows, so expand it unquoted.
#   [CONFIGURE_ARGS_VAR <var>]    Out: configure arguments describing the host.
#                                 Pass them ahead of the package's own, which is
#                                 also what lets a package override one by
#                                 repeating it.
# )
#
# Each *_VAR names a variable set in the caller's scope.  Use them as the
# leading words of a COMMAND:
#
#   COMMAND ${RUN_CONFIGURE} ${CONFIGURE} --prefix=... --enable-foo
#   COMMAND ${RUN} make V=1
#
# No CONFIGURE_ENV value may contain a ';'.  These are expanded into the command
# lists above, and a ';' inside an element cannot be told apart from the
# separator between elements, so the value would arrive split into one argument
# per fragment.  Escaping does not save it: the escape is spent by the first
# unquoted expansion, so it breaks again as soon as a caller builds one list out
# of another.  PKG_CONFIG_PATH is the one value that needs ';'
# (pkgconf splits on it for Windows, a ':' being ambiguous with the drive
# letter), which is why it is a parameter of its own instead of something to
# pass in CONFIGURE_ENV.
function(therock_autotools_shell)
  cmake_parse_arguments(PARSE_ARGV 0 arg
    ""
    "BINARY_DIR;SOURCE_DIR;PKG_CONFIG_PATH;RUN_VAR;RUN_CONFIGURE_VAR;CONFIGURE_VAR;CONFIGURE_ARGS_VAR"
    "CONFIGURE_ENV")

  if(NOT arg_BINARY_DIR)
    message(FATAL_ERROR "therock_autotools_shell: BINARY_DIR is required")
  endif()
  if(arg_CONFIGURE_VAR AND NOT arg_SOURCE_DIR)
    message(FATAL_ERROR
      "therock_autotools_shell: CONFIGURE_VAR needs SOURCE_DIR")
  endif()

  # Environment carried by every command, and the environment's delivery
  # mechanism.  _shell_prefix is what turns a plain argv into one that runs in
  # the shell the host needs, and is empty where none is needed.
  set(_shell_env "")
  set(_shell_prefix "")
  set(_host_args "")

  if(WIN32)
    # Pin --build and --host to the mingw triplet.  They are equal, so configure
    # treats this as a native build instead of a cross one.  Pinning them also
    # keeps configure from deriving the triplet itself: config.guess reads
    # MSYS2's uname and guesses x86_64-pc-mingw64, which config.sub accepts as
    # its own triplet instead of normalizing to mingw32.  Configure logic
    # across the autotools projects keys on *-*-mingw32* in places, so a mingw64
    # triplet would be quietly misidentified, which could result in configure
    # failures, or worse, misconfiguration: bfd/acinclude.m4 would leave
    # USE_BINARY_FOPEN undefined and open binary files in text mode, for
    # example.
    set(_host_args --build=x86_64-w64-mingw32 --host=x86_64-w64-mingw32)

    # Commands cross into the shell as an argv instead of as a string to be
    # re-parsed: the script hands its own positional arguments straight to exec,
    # so nothing needs quoting and one argument list serves every host.
    #
    # The script is a file instead of a 'bash -c' argument.  Its main job is
    # the PKG_CONFIG_PATH export below, which has to run inside the shell, and
    # carrying the exec passthrough alongside it costs nothing and puts the
    # quotes in 'exec "$@"' where no command-line rewriting can reach them.
    # Keep it that way: "cmake -E chdir", for one, drops a double quote
    # embedded in an argument on Windows, and a mangled 'exec "$@"' fails
    # *silently* -- the positional arguments shift by one, exec runs with none,
    # which succeeds, and the shell exits 0 having run nothing.
    #
    # PKG_CONFIG_PATH is exported here instead of being inherited, because the
    # UCRT64 profile a login shell runs overwrites it with the ucrt64 system
    # pkgconfig directories, which would aim configure at msys2's copies of our
    # dependencies.  Setting it past that point also keeps its ';' separators
    # off the command lists.  Build the body as one string, with every append
    # quoted.  A ';' in a value (and PKG_CONFIG_PATH is full of them here) would
    # otherwise turn the append into several list elements, and file(WRITE)
    # joins those with nothing at all, quietly running the directories together
    # into one unusable path.
    set(_script "${arg_BINARY_DIR}/msys2-run.sh")
    set(_script_body
      "# Run a command in an MSYS2 UCRT64 login shell.  Generated by\n")
    string(APPEND _script_body
      "# TheRock's therock_autotools_shell(); see the comment there.\n")
    if(arg_PKG_CONFIG_PATH)
      string(APPEND _script_body
        "export PKG_CONFIG_PATH=\"${arg_PKG_CONFIG_PATH}\"\n")
    endif()
    string(APPEND _script_body "exec \"$@\"\n")
    file(WRITE "${_script}" "${_script_body}")

    # MSYSTEM selects the ucrt64 subsystem, so that gcc resolves to the native
    # toolchain instead of the msys2-runtime one, which would build against
    # the Cygwin-like MSYS2 runtime.  make stays MSYS2's own, which only runs
    # the commands and so does not matter.  CHERE_INVOKING stops /etc/profile
    # changing directory to $HOME, so the build directory stays the cwd.
    #
    # HOME names an empty directory of the build's own.  The login shell is
    # what MSYS2 documents for running its tools, but after /etc/profile it
    # also runs the user's ~/.bash_profile or ~/.profile, and MSYS2 takes HOME
    # from the environment first, else from the Windows profile directory.
    # Whatever those files export (PATH, CFLAGS, ...) would otherwise reach
    # the build, making a local build differ from a CI one.  The directory
    # must exist: /etc/profile fills a missing home from /etc/skel.
    set(_home "${arg_BINARY_DIR}/msys2-home")
    file(MAKE_DIRECTORY "${_home}")
    set(_shell_env MSYSTEM=UCRT64 CHERE_INVOKING=1 "HOME=${_home}")
    set(_shell_prefix "${THEROCK_MSYS2_BASH}" -l "${_script}")

    if(arg_CONFIGURE_VAR)
      # configure is named by a relative, colon-free path, and handed to an
      # explicit "sh".  The two guard against opposite failures.
      #
      # Relative, because a "C:/..." srcdir trips an autoconf >= 2.70 regression
      # that splits the auxiliary-file search list on ':' and tears the drive
      # letter off (see therock_msys2.cmake).
      #
      # Through "sh", because autoconf takes $srcdir from $0, and MSYS2 rewrites
      # the argv[0] of an exec'd *script* into POSIX form.  Exec configure
      # directly and $srcdir comes out as "/c/build/.../../source", which a
      # native compiler cannot read.  A package that feeds $srcdir to the
      # compiler then fails: gmp's sizeof(mp_limb_t) probe #includes
      # "$srcdir/gmp-h.in", reports the size as 0, and dies with the misleading
      # "Oops, mp_limb_t doesn't seem to work".  Naming sh makes *sh* the
      # program being exec'd, so the script path stays an ordinary argument and
      # arrives unrewritten.
      therock_msys2_configure_srcdir(
        "${arg_BINARY_DIR}" "${arg_SOURCE_DIR}" _rel_configure)
      set(_configure sh "${_rel_configure}")
    endif()
  else()
    if(arg_PKG_CONFIG_PATH)
      string(REPLACE ";" ":" _pkg_config_path "${arg_PKG_CONFIG_PATH}")
      set(_shell_env "PKG_CONFIG_PATH=${_pkg_config_path}")
    endif()
    if(arg_CONFIGURE_VAR)
      set(_configure "${arg_SOURCE_DIR}/configure")
    endif()
  endif()

  # These prefixes do NOT change directory: the caller's target sets
  # WORKING_DIRECTORY to BINARY_DIR instead.  That is simply less machinery, and
  # it keeps "cmake -E chdir" out of the picture, which on Windows rewrites the
  # child command line and mangles quotes.
  #
  # BINARY_DIR is created here so that WORKING_DIRECTORY is valid even for a
  # build directory the recipe has not made yet.
  file(MAKE_DIRECTORY "${arg_BINARY_DIR}")

  # Name the compiler instead of leaving configure to find one on PATH.  On
  # Windows that is the difference between building with the declared ucrt64 gcc
  # and building with whatever the MSYS2 login shell happens to offer, and it is
  # what carries the compiler launcher through.
  #
  # The launcher (ccache and friends) is a CMake list, a program and any
  # arguments, and goes in front of the compiler joined with spaces, as
  # therock_meson_env.cmake does for meson.
  #
  # On Windows the program is conventionally a plain name, which the MSYS2
  # shell cannot resolve: it has a PATH of its own and no knowledge of the one
  # CMake was run with.  So there it is resolved to an absolute path, on every
  # configure (NO_CACHE), so that a changed launcher takes effect.  If it
  # cannot be found, configure fails, naming it.  A missing launcher breaks
  # the CMake-built sub-projects too, and passed through unresolved it would
  # only surface inside the shell as configure's "C compiler cannot create
  # executables".
  set(_compiler_env)
  foreach(_lang C CXX)
    if(_lang STREQUAL "C")
      set(_env_var CC)
    else()
      set(_env_var CXX)
    endif()
    # A caller's own CC or CXX, such as a wrapper adding flags, goes in
    # CONFIGURE_ENV.  Naming the compiler here as well would leave two values
    # of the variable on one "cmake -E env" line, the later one winning.
    set(_caller_value ${arg_CONFIGURE_ENV})
    list(FILTER _caller_value INCLUDE REGEX "^${_env_var}=")
    if(_caller_value)
      continue()
    endif()
    set(_cc "${CMAKE_${_lang}_COMPILER}")
    set(_launcher ${CMAKE_${_lang}_COMPILER_LAUNCHER})
    if(_launcher AND WIN32)
      list(POP_FRONT _launcher _launcher_program)
      # find_program skips the search if the variable already has a value,
      # as it would from the previous language.
      unset(_launcher_path)
      find_program(_launcher_path NAMES "${_launcher_program}" NO_CACHE)
      if(NOT _launcher_path)
        message(FATAL_ERROR
          "Compiler launcher '${_launcher_program}' "
          "(CMAKE_${_lang}_COMPILER_LAUNCHER) not found")
      endif()
      list(PREPEND _launcher "${_launcher_path}")
    endif()
    # configure splits CC and CXX into words at spaces, so a path containing
    # one (e.g. a launcher under "C:/Program Files") would be cut in two, and
    # fail as "C compiler cannot create executables".  Say so here instead.
    foreach(_word IN LISTS _launcher ITEMS "${_cc}")
      if(_word MATCHES " ")
        message(FATAL_ERROR
          "'${_word}' (from CMAKE_${_lang}_COMPILER or "
          "CMAKE_${_lang}_COMPILER_LAUNCHER) contains a space, which an "
          "autotools configure cannot handle in CC or CXX.  Use a copy at a "
          "path without spaces.")
      endif()
    endforeach()
    if(_launcher)
      string(REPLACE ";" " " _launcher "${_launcher}")
      set(_cc "${_launcher} ${_cc}")
    endif()
    list(APPEND _compiler_env "${_env_var}=${_cc}")
  endforeach()

  # The two prefixes differ only in the variables handed to cmake -E env, which
  # have to precede its "--", so the common part is spelled once.
  set(_head "${CMAKE_COMMAND}" -E env ${_shell_env})

  if(arg_RUN_VAR)
    set("${arg_RUN_VAR}" ${_head} -- ${_shell_prefix} PARENT_SCOPE)
  endif()
  if(arg_RUN_CONFIGURE_VAR)
    set("${arg_RUN_CONFIGURE_VAR}"
      ${_head} ${_compiler_env} ${arg_CONFIGURE_ENV} -- ${_shell_prefix}
      PARENT_SCOPE)
  endif()
  if(arg_CONFIGURE_VAR)
    set("${arg_CONFIGURE_VAR}" "${_configure}" PARENT_SCOPE)
  endif()
  if(arg_CONFIGURE_ARGS_VAR)
    set("${arg_CONFIGURE_ARGS_VAR}" ${_host_args} PARENT_SCOPE)
  endif()
endfunction()

# therock_autotools_subproject_args(<target_name> <out_var>)
#
# Set <out_var> to the therock_cmake_subproject_declare() arguments that every
# sub-project built through therock_autotools_shell() needs, for splicing into
# its declare call alongside its own arguments, which add to these.  That is
# this file as an FPRINT_FILE_GLOBS and an EXTRA_DEPENDS entry on every host,
# plus, on Windows, what therock_msys2_host_gnu_toolchain_args() returns for
# <target_name>, which fails configure if MSYS2 is missing.
#
# The caller still names COMPILER_TOOLCHAIN itself (host-gnu on Windows),
# literally in the declare call: build_tools' static consumer-graph analyzer
# groups a declare call's arguments by the keywords it can see before
# expanding any variable, so one hidden in <out_var> would be invisible to it.
function(therock_autotools_subproject_args target_name out_var)
  # This file decides how the sub-project's configure and make are run, so it
  # belongs in the sub-project's fingerprint.  The fingerprint is a content
  # hash computed when the super-project configures, which decides the
  # artifact's identity but does not re-run a build, so an edit to this file
  # re-runs the sub-project's through EXTRA_DEPENDS.
  set(_args
    FPRINT_FILE_GLOBS "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
    EXTRA_DEPENDS "${CMAKE_CURRENT_FUNCTION_LIST_FILE}")
  if(WIN32)
    therock_msys2_host_gnu_toolchain_args("${target_name}" _msys2_args)
    list(APPEND _args ${_msys2_args})
  endif()
  set("${out_var}" ${_args} PARENT_SCOPE)
endfunction()
