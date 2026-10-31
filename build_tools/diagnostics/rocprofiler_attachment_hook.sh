# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Private BASH_ENV for one diagnostic child process tree. Other grep calls pass
# through. Installed tests, their timeout counter and their exit policy are intact.
grep() {
  local __rap_arg __rap_pid="" __rap_candidate="" __rap_tid="" __rap_valid=true
  if [[ -z "${ATTACHMENT_DIAG_ROOT:-}" || $# -lt 3 || "$1" != "-ql" || "$2" != "rocp-bg-attach" ]]; then
    command grep "$@"
    return $?
  fi
  for __rap_arg in "${@:3}"; do
    __rap_candidate="${__rap_arg#/proc/}"
    __rap_candidate="${__rap_candidate%%/*}"
    __rap_tid="${__rap_arg#"/proc/${__rap_candidate}/task/"}"
    __rap_tid="${__rap_tid%/comm}"
    if [[ -n "${__rap_candidate}" && "${__rap_candidate}" != *[!0-9]* &&
          ( "${__rap_tid}" == '*' || ( -n "${__rap_tid}" && "${__rap_tid}" != *[!0-9]* ) ) &&
          "${__rap_arg}" == "/proc/${__rap_candidate}/task/${__rap_tid}/comm" ]]; then
      if [[ -z "${__rap_pid}" ]]; then
        __rap_pid="${__rap_candidate}"
      elif [[ "${__rap_pid}" != "${__rap_candidate}" ]]; then
        __rap_valid=false
      fi
    else
      __rap_valid=false
    fi
  done
  if [[ "${__rap_valid}" != true || -z "${__rap_pid}" ]]; then
    command grep "$@"
    return $?
  fi
  local __rap_sequence=$(( ${__RAP_POLL_SEQUENCE:-0} + 1 ))
  __RAP_POLL_SEQUENCE=${__rap_sequence}
  local __rap_error="${ATTACHMENT_DIAG_ROOT}/poll-errors/${BASHPID}-${__rap_pid}-${__rap_sequence}.txt"
  local __rap_fd __rap_result __rap_begin=${EPOCHREALTIME:-unknown}
  # If capture cannot open its file, execute the original command anyway. The
  # separate collector-completeness gate must not replace the product result.
  if ! exec {__rap_fd}>"${__rap_error}"; then
    command grep "$@"
    return $?
  fi
  if command grep "$@" 2>&"${__rap_fd}"; then
    __rap_result=0
  else
    __rap_result=$?
  fi
  exec {__rap_fd}>&- || :
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${__rap_begin}" "${EPOCHREALTIME:-unknown}" "${BASHPID}" \
    "${__rap_pid}" "${__rap_result}" "${__rap_error}" \
    >>"${ATTACHMENT_DIAG_ROOT}/poll-events.tsv" || :
  return "${__rap_result}"
}
