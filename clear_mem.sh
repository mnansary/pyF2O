#!/bin/sh
while true
do
free -m
echo 1 > /proc/sys/vm/drop_caches
free -m
sleep 30
done
