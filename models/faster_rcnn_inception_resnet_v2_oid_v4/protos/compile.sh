#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo Compiling files in directory: $DIR
protoc -I=$DIR --python_out=$DIR $DIR/string_int_label_map.proto