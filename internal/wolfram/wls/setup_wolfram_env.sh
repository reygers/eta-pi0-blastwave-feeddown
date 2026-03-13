#!/usr/bin/env bash
# Source this file to configure Wolfram kernel variables for wolframscript.
# Usage:
#   source wls/setup_wolfram_env.sh

_WOLFRAM_KERNEL_DEFAULT="/Applications/Wolfram.app/Contents/MacOS/WolframKernel"

if [ -n "${WOLFRAMSCRIPT_KERNELPATH:-}" ] && [ -x "${WOLFRAMSCRIPT_KERNELPATH}" ]; then
  _WOLFRAM_KERNEL="${WOLFRAMSCRIPT_KERNELPATH}"
elif [ -x "${_WOLFRAM_KERNEL_DEFAULT}" ]; then
  _WOLFRAM_KERNEL="${_WOLFRAM_KERNEL_DEFAULT}"
else
  echo "ERROR: Could not find WolframKernel." 1>&2
  echo "Set WOLFRAMSCRIPT_KERNELPATH manually, e.g.:" 1>&2
  echo "  export WOLFRAMSCRIPT_KERNELPATH=/path/to/WolframKernel" 1>&2
  return 1 2>/dev/null || exit 1
fi

export WOLFRAMSCRIPT_KERNELPATH="${_WOLFRAM_KERNEL}"
export WolframKernel="${_WOLFRAM_KERNEL}"

echo "Configured Wolfram kernel:"
echo "  WOLFRAMSCRIPT_KERNELPATH=${WOLFRAMSCRIPT_KERNELPATH}"
echo "  WolframKernel=${WolframKernel}"

unset _WOLFRAM_KERNEL_DEFAULT
unset _WOLFRAM_KERNEL
