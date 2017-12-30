#ifndef RANDOM_H
#define RANDOM_H
#include <cstdlib>

unsigned long long next_random;

extern "C"
void randReset() {
	next_random = rand();
}

unsigned long long randd(int id) {
	next_random = next_random * (unsigned long long)25214903917 + 11;
	return next_random;
}

int rand_max(int id, int x) {
	int res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

#endif
