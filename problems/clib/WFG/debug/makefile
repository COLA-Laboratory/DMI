CC=gcc

CFLAGS= -lm -I.

DEPS=../WFG.h

SRC=../WFG.c ../WFG1.c ../WFG2.c ../WFG3.c ../WFG4.c ../WFG5.c ../WFG6.c ../WFG7.c ../WFG8.c ../WFG9.c ../WFG2x.c ../WFG4x.c debug.c

OBJ=obj/WFG1.o obj/WFG2.o obj/WFG3.o obj/WFG4.o obj/WFG5.o obj/WFG6.o obj/WFG7.o obj/WFG8.o obj/WFG9.o obj/WFG2x.o obj/WFG4x.o obj/WFG.o obj/debug.o

obj/%.o: ../%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: ${OBJ}
	$(CC) -o $@ $^ $(CFLAGS)

obj/debug.o: debug.c ${DEPS}
	$(CC) -c -o obj/debug.o $< $(CFLAGS)

clean:
	rm -rf obj/*.o