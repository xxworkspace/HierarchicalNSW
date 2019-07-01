#!/bin/bash
for ((a=1; a<= 25 ;a++))
do
  echo "$a"
   ./avxgemv_test $a
done
