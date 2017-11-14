CC=g++-6
CFLAGS=-std=c++14 -Ilibs/
OPT=-O3 -march=native -flto
# OPT=-pg -O2 -march=native -flto -I../libs/
# OPT=-g -march=native

BIN_D=../bin
OBJ_D=../obj
MNIST_D=../mnist-data

.PHONY: all makedir clean

all: mnist

mnist: format.o Network.o mnist.o mnist_utils.o 
	$(CC) $(OBJ_D)/format.o $(OBJ_D)/Network.o $(OBJ_D)/mnist.o $(OBJ_D)/mnist_utils.o $(OPT) -o $(BIN_D)/mnist
	ln -sf $(MNIST_D)/* $(BIN_D)


mnist.o: mnist.cpp Network.h Network.tpp makedir
	$(CC) $(CFLAGS) $(OPT) -c mnist.cpp -o $(OBJ_D)/mnist.o

mnist_utils.o: mnist_utils.cpp makedir
	$(CC) $(CFLAGS) $(OPT) -c mnist_utils.cpp -o $(OBJ_D)/mnist_utils.o

Network.o: Network.cpp Network.h Network.tpp makedir
	$(CC) $(CFLAGS) $(OPT) -c Network.cpp -o $(OBJ_D)/Network.o

format.o: libs/fmt/format.cc makedir
	$(CC) $(CFLAGS) $(OPT) -c libs/fmt/format.cc -o $(OBJ_D)/format.o


makedir:
	mkdir -p $(BIN_D)
	mkdir -p $(OBJ_D)


clean:
	rm -rf $(BIN_D) $(OBJ_D)
