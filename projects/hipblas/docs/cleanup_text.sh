#!/bin/bash

# Script to clean up RST and other text files
# By Lee Killough

export PATH=/usr/bin:/bin

files=
rstcode=0
ascii=0
trailing=0
trailing_after=0

main()
{
    parse_args "$@"
    cleanup
}

cleanup()
{
    # iconv command to translate UTF8 to ASCII
    iconv="/usr/bin/iconv -s -f utf-8 -t ascii//TRANSLIT"

    set -ex

    git ls-files -z --exclude-standard "$files" | while read -rd '' file; do
        # Operate only on regular files of MIME type text/*
        if [[ -f $file && "$(file -b --mime-type "$file")" == text/* ]]; then
            # Add missing newline to end of file
            sed -i -e '$a\' "$file"

            # Remove trailing whitespace at end of lines
            if [[ $trailing -ne 0 ]]; then
                sed -i -e 's/[[:space:]]*$//' "$file"
            elif [[ $trailing_after -ne 0 ]]; then
                perl -pi -e 's/\S\K\s+$/\n/' "$file"
            fi

            # Temporary file
            temp=$(mktemp)

            # Replace non-ASCII text and/or RST code line with ASCII equivalents
            if [[ $ascii -ne 0 ]]; then
                $iconv "$file" > "$temp"
            elif [[ $rstcode -ne 0 && $file == *.rst ]]; then
                { set +x; } 2>/dev/null
                echo perl -e '$(rstcode_perl)' "\"$iconv\" \"$file\" > \"$temp\"" >&2
                perl -e "$(rstcode_perl)" "$iconv" "$file" > "$temp"
                set -x
            fi

            # Preserve permissions and add file to Git if updated
            chmod --reference="$file" "$temp"
            mv -f "$temp" "$file"
            git add -u "$file"
        fi
        echo "" >&2
    done

    { set +x; } 2>/dev/null
    git status
    echo " All of the selected files in the repository have been cleaned up."
}

# Parse the command-line arguments
parse_args()
{
    while [[ $# -ne 0 ]]; do
        case "$1" in
            --files)
                shift
                [[ $# -ne 0 ]] || usage
                files="$1"
                ;;
            --rstcode)
                : ${files:=*.rst}
                rstcode=1
                ;;
            --ascii)
                ascii=1
                ;;
            --trailing)
                $trailing=1
                ;;
            --trailing-after)
                ;&
            --trailing_after)
                $trailing_after=1
                ;;
            *)
                usage
        esac
        shift
    done

    [[ -n $files ]] || usage
}

# Perl script to handle code sections of RST files
rstcode_perl()
{
    cat <<'EOF'
use strict;
use warnings;
my ($iconv, $code, $code_indent) = (shift, 0);
while(<>)
{
    my ($indent) = /^(\s*)/;
    if($code) {
      $code = 0 if /\S/ && length($indent) <= length($code_indent);
      open ICONV, "|-", $iconv or die "$!";
      print ICONV;
      close ICONV;
    } else {
      ($code, $code_indent) = (1, $indent) if /::(\s+\S+)?\s*$/;
      print;
    }
}
EOF
}

# Help message
usage()
{
    cat<<EOF
Usage:

    $0
       [ --files <wildcard or path> ]
       [ --rstcode ]
       [ --ascii ]
       [ --trailing | --trailing-after ]

Description:

    Replaces non-ASCII Unicode characters with their ASCII equivalents in
    selected text files, or in the code sections of reStructuredText (RST)
    files.

    Adds missing newlines at the ends of selected text files.

    Optionally removes trailing whitespace at the ends of lines in selected
    text files.

    Code sections of RST files are critically important, because they are
    often copied-and-pasted to a user's terminal, and if they contain
    non-ASCII characters, then they will not work.

Options:

    --files <wildcard or path>

                       Clean up all text files matching wildcard or path,
                       e.g.:

                       --files "*.md"
                       --files "*.rst"
                       --files "*"
                       --files README.md

                       (Wildcard may need to be quoted, to prevent shell
                       wildcard expansion.)

    --rstcode          Clean up only the code sections of selected RST
                       files, or all RST files if --files is not specified.

    --ascii            Replace non-ASCII UTF-8 characters in selected text
                       files with their ASCII equivalents.

    --trailing         Remove trailing whitespace at the ends of lines in
                       selected files. This includes converting CR-LF to LF.

    --trailing-after   Remove trailing whitespace at the ends of lines in
                       selected files, but only after non-space characters.
                       This prevents removing indentation from otherwise
                       blank lines.

EOF
    exit 1
}

main "$@"
