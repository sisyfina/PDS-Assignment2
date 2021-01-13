#ifndef _V0_HELPER_H_
#define _V0_HELPER_H_

// Definition of the kNN result struct
typedef struct knnresult{
  int    * nidx;    //!< Indices (0-based) of nearest neighbors [m-by-k]
  double * ndist;   //!< Distance of nearest neighbors          [m-by-k]
  int      m;       //!< Number of query points                 [scalar]
  int      k;       //!< Number of nearest neighbors            [scalar]
} knnresult;


void hadamard(double * X, double * hadam, int n, int d);
void sqrtd(double * D, int n, int m);
void distance(double * X, double * Y, int n, int m, int d, double *D);
void block_distance(double * X, double * Y, int n, int m, int d, double *D, int startidx);
void swap(double *list, int *index, int one, int two);
int partition(double *list, int *index, int left, int right);
knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
void initialize(knnresult * sptr);
void allocate(knnresult * sptr);
void quicksort(double * D, int * idx, int lo, int hi);
//double quickSelect(double *list, int *index, int left, int right, int k);
int chooseChunk(int m);

#endif
