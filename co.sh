#!/bin/sh

RUN=FALSE
NEKO=FALSE
DEBUG=FALSE
OPT=
VALUE=

while getopts ang OPT
    do
        case $OPT in
            a) RUN=TRUE ;;
            n) NEKO=TRUE ;;
            g) DEBUG=TRUE ;;
            /?) echo "Usage: $0 [-ang] parameter" 1>&2
               exit 1 ;;
        esac
    done
    shift `expr $OPTIND - 1`

if [ "$RUN" = "TRUE" ]; then
    g++ $1 -std=c++11 && echo "input" && ./a.out
elif [ "$NEKO" = "TRUE" ]; then
    cat $1 | pbcopy
elif [ "$DEBUG" = "TRUE" ]; then
    g++ $1 -g -std=c++11 && lldb a.out
else
    g++ $1 -std=c++11
fi
