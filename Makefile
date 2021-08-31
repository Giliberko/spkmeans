.PHONY: kiki build all

all: kiki build

kiki:
	gcc -ansi -Wall -Wextra -Werror -pedantic-errors spkmeans.c -lm -o spkmeans

build:
	python setup.py build_ext --inplace

debug:
	gcc -ansi -Wall -Wextra -Werror -pedantic-errors spkmeans.c -lm -o spkmeans -D DEBUG