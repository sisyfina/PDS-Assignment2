#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <string.h>
#include <cblas.h>
#include "V0_helper.h"
#include "reader_helper.h"


int main(int argc, char*argv[])
{
  struct timespec ts_start, ts_end;
  knnresult knn;
  int n, m, d, k;
  double *X, *Y;

  X = read_X(&n, &d, argv[1]);
  Y = read_X(&n, &d, argv[2]);
  k = atoi(argv[3]);

  clock_gettime(CLOCK_MONOTONIC, &ts_start); // get the start
  knn = kNN(X, Y, n, m, d, k);
  clock_gettime(CLOCK_MONOTONIC, &ts_end); // get the finishing
  long delta_sec = ts_end.tv_nsec - ts_start.tv_nsec;
  printf("Execution time: %ld ns\n", delta_sec);

  return 0;
}
