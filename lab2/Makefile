all: clean gaussian

clean:
	rm -f gaussian

gaussian:
	nvcc cuda_gaussjordan.cu -o gaussian && time ./gaussian
