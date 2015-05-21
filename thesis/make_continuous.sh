#!/usr/bin/env sh
latexmk -pvc --interaction=nonstopmode -pdf -bibtex main 2>&1 | egrep -i --color=auto "undefined|error|$"

