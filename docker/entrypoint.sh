#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    source activate InterpAny
    cd /InterpAny-Clearer/webapp/backend && nohup flask run -p 5001 --host "0.0.0.0" &
    cd /InterpAny-Clearer/webapp/webapp && yarn && nohup yarn start &
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null