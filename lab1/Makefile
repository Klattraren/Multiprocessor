all: clean quicksort gaussian instructions

clean:
	rm -f quicksort
	rm -f gaussian

quicksort:
	gcc -O2 -o quicksort qsortpar.c -lpthread

gaussian:
	gcc -O2 -o gaussian multicore_gaussian.c -lpthread

instructions:
	@echo "\n\033[1m To run quicksort, use 'time ./quicksort'\033[0m"
	@echo " \033[1mTo run gaussian, use 'time ./gaussian'\033[0m"