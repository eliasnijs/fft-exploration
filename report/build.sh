#!/bin/sh

groff -ms -s -e -t -p -U -R verslag.ms -T ps > verslag.ps
ps2pdf verslag.ps > verslag.pdf
