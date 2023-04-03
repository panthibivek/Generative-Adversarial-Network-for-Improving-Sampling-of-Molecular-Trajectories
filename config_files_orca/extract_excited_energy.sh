#!/bin/bash
cat molecule.log | grep "Lowest Energy" | tail -n1 | cut -d' ' -f20

