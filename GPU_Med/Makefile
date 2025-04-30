# Source files
C_SOURCES   = $(wildcard matrix/*.c neural/*.c util/*.c *.c)
CU_SOURCES  = $(wildcard matrix/*.cu neural/*.cu util/*.cu *.cu)
HEADERS     = $(wildcard matrix/*.h neural/*.h util/*.h *.h)

# Object files
C_OBJ  = ${C_SOURCES:.c=.o}
CU_OBJ = ${CU_SOURCES:.cu=.o}
OBJ    = ${C_OBJ} ${CU_OBJ}

# Compiler and flags
CC     = gcc
NVCC   = nvcc
CFLAGS =
NVFLAGS =

MAIN   = main

# Default target
run: ${MAIN}
	./${MAIN}

${MAIN}: ${OBJ}
	${NVCC} ${NVFLAGS} $^ -o $@ -lm

# Compile C sources
%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $< -o $@

# Compile CUDA sources
%.o: %.cu ${HEADERS}
	${NVCC} ${NVFLAGS} -c $< -o $@

# Cleanup
clean:
	rm -f matrix/*.o neural/*.o util/*.o *.o ${MAIN}

