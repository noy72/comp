    set -u
    RUN=FALSE
    NEKO=FALSE
    DEBUG=FALSE
    OPT=""
    OPTIND_SAVE=$OPTIND
    VALUE=""

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
        g++ $1 -std=c++14 -Wall -fsanitize=undefined -D_GLIBCXX_DEBUG && echo "input" && ./a.out
    elif [ "$NEKO" = "TRUE" ]; then
        cat $1 | pbcopy
    elif [ "$DEBUG" = "TRUE" ]; then
        g++ $1 -g -std=c++14 && lldb a.out
    else
        echo "only"
        g++ $1 -std=c++11
    fi

    OPTIND=$OPTIND_SAVE
