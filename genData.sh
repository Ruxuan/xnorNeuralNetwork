#!/bin/bash

gcc xnorDataGenerator.c
./a.out > data.txt
rm a.out
