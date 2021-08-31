cmake_minimum_required(VERSION 3.17)
project(spkmeans C)

set(CMAKE_C_STANDARD 90)

add_library(spkmeans spkmeans.c spkmeans.h main.c main.h)