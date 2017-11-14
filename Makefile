.PHONY: all clean run

all:
	cd src && make all

clean:
	cd src && make clean

run:
	cd bin && ./mnist