# Default target to clean, compile, and run the main program
all: clean run_gaussian

# Clean target
clean:
	rm -f gaussian oddeven1 oddeven2

# Compile and run Gaussian
gaussian:
	nvcc cuda_gaussjordan.cu -o gaussian

run_gaussian: gaussian
	@echo "Running Gaussian elimination..."
	time ./gaussian

# Compile and run Odd-Even Sort Task 1
oddeven1:
	nvcc oddevensort_task1.cu -o oddeven1

run_oddeven1: oddeven1
	@echo "Running Odd-Even Sort Task 1..."
	./oddeven1

# Compile and run Odd-Even Sort Task 2
oddeven2:
	nvcc oddevensort_task2.cu -o oddeven2

run_oddeven2: oddeven2
	@echo "Running Odd-Even Sort Task 2..."
	./oddeven2

# Target to compile and run all programs
run_all: run_gaussian run_oddeven1 run_oddeven2
	@echo "All tasks complete!"
