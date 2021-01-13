#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include "V1_helper.h"
#include "reader_helper.h"


int main(int argc, char* argv[])
{
  struct timespec ts_start, ts_end;
  knnresult knn;
  int n, d, k;
  double *X;

  MPI_Init( &argc, &argv );

  int SelfTID, NumTasks;
  MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );
  MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
  printf("NumTasks = %d\n", NumTasks);

  X = read_X(&n, &d, argv[1]);
  k = atoi(argv[2]); // # neighbors

  clock_gettime(CLOCK_MONOTONIC, &ts_start); // get the start
  knn = distrAllkNN(X, n, d, k);
  clock_gettime(CLOCK_MONOTONIC, &ts_end); // get the finishing
  long delta_sec = ts_end.tv_nsec - ts_start.tv_nsec;
  printf("Execution time: %ld ns\n", delta_sec);

  MPI_Finalize();

  free(X);


  return 0;
}
