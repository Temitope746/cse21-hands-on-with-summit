COMP = xlc_r
FLAGS = -qsmp=omp -qoffload

vec_add: vector_addition.o
	$(COMP) $(FLAGS) vector_addition.o -o vec_add

vector_addition.o: vector_addition.c
	$(COMP) $(FLAGS) -c vector_addition.c

.PHONY: clean cleanall

clean:
	rm -f vec_add *.o

cleanall:
	rm -f vec_add *.o add_vec_ompGPU*
