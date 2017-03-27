#include <stdio.h>
#include <stdlib.h>

const int threshold = 50;
const int mod = 100;
const int m = 500;
const unsigned int seed = 1234;

int main() {
	srand(seed);

	for (int i = 0; i < m; i++) {
		int a = rand() % mod;
		int b = rand() % mod;

		int activation = 0;
		if ((a >= threshold && b >= threshold) || (a < threshold && b < threshold)) {
			activation = 1;
		}

		printf("%d %d %d\n", a, b, activation);
	}
}
