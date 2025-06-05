#!/usr/bin/env bash

sudo chmod o+rw /dev/bus/usb/001/*
sudo chmod o+rw /dev/bus/usb/002/*

python3 detection_server.py
