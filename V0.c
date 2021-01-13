#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <math.h>
#include "V0_helper.h"
#include "reader_helper.h"


void initialize(knnresult * sptr){
  sptr = (knnresult*)malloc(sizeof(knnresult));
  sptr->nidx = NULL;
  sptr->ndist = NULL;
}

void allocate(knnresult * sptr){
  sptr->nidx = (int *)malloc((sptr->m)*(sptr->k)*sizeof(int));
  sptr->ndist = (double *)malloc((sptr->m)*(sptr->k)*sizeof(double));
}

void hadamard(double * X, double * hadam, int n, int d){
  double tmp;
  int rm = n*d;

  for(int i=0; i<rm; i++){
    tmp = X[i];
    tmp = tmp*tmp;
    hadam[i] = tmp;
  }
}

void sqrtd(double * D, int n, int m){
  double tmp;
  for(int i=0; i<n*m; i++){
    tmp = D[i];
    tmp = sqrt(tmp);
    D[i] = tmp;
  }
}

void distance(double * X, double * Y, int n, int m, int d, double *D){
  double *e1 = (double *)malloc(m*d *sizeof(double));
  double *XH = (double *)malloc(n*d *sizeof(double));

  int tmp;
  tmp = m*d;
  for(int i=0; i<tmp; i++){
    e1[i] = 1.0;
  }

  hadamard(X, XH, n, d);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n,m,d,1,XH,d,e1,d,0,D,m);
  free(e1);
  free(XH);

  double *e2 = (double *)malloc(n*d *sizeof(double));
  tmp = n*d;
  for(int i=0; i<tmp; i++){
    e2[i] = 1.0;
  }
  double *YH = (double *)malloc(m*d *sizeof(double));

  hadamard(Y, YH, m, d);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n,m,d,1,e2,d,YH,d,1,D,m);
  free(e2);
  free(YH);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              n,m,d,-2,X,d,Y,d,1,D,m);

  sqrtd(D, n, m);
/*
  printf("D = ");
  for(int i=0; i<n*m; i++)
    printf("%lf ", D[i]);
  printf("\n");
  */
}


void block_distance(double * X, double * Y, int n, int m, int d, double *D, int startidx){

  if(n>=50 && (n%2 == 0)){
    //printf("Hi from block!!!\n");
    int half = n/2;
    block_distance( X,       Y, half, m, d, &D[startidx],        startidx);
    block_distance(&X[half], Y, half, m, d, &D[startidx+half*m], startidx+half*m);
  }
  else{
     distance(X, Y, n, m, d, &D[startidx]);
  }

  distance(X, Y, n, m, d, &D[startidx]);

}



void swap(double *list, int *index, int one, int two){
  //printf("Hi from swap!\n");

  int tmpidx;
  double tmp;
  tmpidx = index[one];
  tmp = list[one];
  index[one] = index[two];
  list[one] = list[two];
  index[two] = tmpidx;
  list[two] = tmp;
  //printf("Bye from swap!\n");

}
/*
double quickSelect(double *list, int *index, int left, int right, int k){
  int pivotIndex;
  // if the list contains only one element
  if(left == right)
    return list[left]; // return that element

  // select a pivotIndex between left and right
  pivotIndex = partition(list, index, left, right);

  if(k == pivotIndex)
    return list[k];
  else if(k < pivotIndex)
    return quickSelect(list, index, left, pivotIndex - 1, k);
  else
    return quickSelect(list, index, pivotIndex + 1, right, k);
}
*/

int partition(double *list, int *index, int left, int right){
  double pivot = list[(left+right)/2];
  int i = left - 1;
  int j = right + 1;
  for(;;){
    do{
      i++;
    }while(list[i] < pivot);
    do{
      j--;
    }while(list[j] > pivot);
    if(i>=j)
      return j;
    swap(list, index, i, j);
  }
}

void quicksort(double * D, int * idx, int lo, int hi){
  int p;
  if (lo < hi){
    p = partition(D, idx, lo, hi);
    quicksort(D, idx, lo, p);
    quicksort(D, idx, p+1, hi);
  }
}

int chooseChunk(int m){
  if(m/2>=50 && m%2==0){
    return chooseChunk(m/2);
  }
  return m;
}

knnresult kNN(double * X, double * Y, int n, int m, int d, int k){

  int chunk = m;
  int i;
  knnresult result;
  initialize(&result);

  result.m = m;
  result.k = k;

  // choose chunk
  chunk = chooseChunk(m);

  double *kdist = (double *)malloc(k*m *sizeof(double));
  int *kindexD = (int *)malloc(k*m *sizeof(int));

  if(m == chunk){

    double *D = (double *)malloc(n*m *sizeof(double));
    int *indexD = (int *)malloc(n*m *sizeof(int));

    for(i=0; i<m; i++){
      for(int j=0; j<n; j++)
        indexD[i*n+j] = j;
    }

    //distance(Y, X, m, n, d, D);
    block_distance(Y, X, m, n, d, D, 0);

    for(i=0; i<m; i++){
      quicksort(&D[i*n], &indexD[i*n], 0, n-1);
    }
/*
    printf("Dquicksort = ");
    for(int j = 0; j<k; j++){
      printf(" %lf", D[(m-1)*n+j]);
    }
    printf("\n");

    printf("indexDquicksort = ");
    for(int j = 0; j<k; j++){
      printf(" %d", indexD[(m-1)*n+j]);
    }
    printf("\n");
*/

    // CHOOSE Kneighbours

    for(i=0; i<m; i++){
      for(int j=0; j<k; j++){
        kdist[i*k+j] = D[i*n+j];
      }
    }

    for(i=0; i<m; i++){
      for(int j=0; j<k; j++){
        kindexD[i*k+j] = indexD[i*n+j];
      }
    }

    free(D);
    free(indexD);
  }
  else {
    double *D = (double *)malloc(n*chunk *sizeof(double));
    int *indexD = (int *)malloc(n*chunk *sizeof(int));

    // first chunk

    for(i=0; i<chunk; i++){
      for(int j=0; j<n; j++)
        indexD[i*n+j] = j;
    }

    //block_distance(Y, X, m, n, d, D, 0);
    distance(Y, X, chunk, n, d, D);

    for(i=0; i<chunk; i++){
      quicksort(&D[i*n], &indexD[i*n], 0, n-1);
    }

    for(i=0; i<chunk; i++){
      for(int j=0; j<k; j++){
        kdist[i*k+j] = D[i*n+j];
      }
    }

    for(i=0; i<chunk; i++){
      for(int j=0; j<k; j++){
        kindexD[i*k+j] = indexD[i*n+j];
      }
    }

    // for rest of chunks
    for(int ch=1; ch<(m/chunk); ch++){

      for(i=0; i<chunk; i++){
        for(int j=0; j<n; j++)
          indexD[i*n+j] = j;
      }

      //block_distance(&Y[ch*chunk*d], X, m, n, d, D, 0);
      distance(&Y[ch*chunk*d], X, chunk, n, d, D);
      for(i=0; i<chunk; i++){
        quicksort(&D[i*n], &indexD[i*n], 0, n-1);
      }

      for(i=0; i<chunk; i++){
        for(int j=0; j<k; j++){
          kdist[(ch*chunk+i)*k+j] = D[i*n+j];
        }
      }

      for(i=0; i<chunk; i++){
        for(int j=0; j<k; j++){
          kindexD[(ch*chunk+i)*k+j] = indexD[i*n+j];
        }
      }

    }

    free(D);
    free(indexD);

  }

  allocate(&result);

  result.nidx = &kindexD[0];
  result.ndist = &kdist[0];

  return result;

}
