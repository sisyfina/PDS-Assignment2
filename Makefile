CC=gcc
MPICC=mpicc

default: all

main:
	$(CC) -o main main.c V0.c reader.c -lopenblas -lpthread -lm

distrmain:
	$(MPICC)  -o distrmain distrmain.c V1.c reader.c -lopenblas -lpthread -lm 

.PHONY: clean

all: main distrmain

clean:
	rm -f main distrmain
