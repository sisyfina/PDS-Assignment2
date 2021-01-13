#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "V1_helper.h"

// mpicc -o run TESTV1.c testerV1.c -lopenblas -lpthread -lm
// mpirun -np 2 run


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
  //printf("Bye from distance!\n");
}


void block_distance(double * X, double * Y, int n, int m, int d, double *D, int startidx){

  if(n>=50 && (n%2 == 0)){
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

  int tmpidx;
  double tmp;
  tmpidx = index[one];
  tmp = list[one];
  index[one] = index[two];
  list[one] = list[two];
  index[two] = tmpidx;
  list[two] = tmp;

}

/*
double quickSelect(double *list, int *index, int left, int right, int k){
  //printf("Hi from quickSelect!\n");
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
//  printf("Hi from quickSelect!\n")
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

  //printf("Hi from kNN!\n");
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

  //printf("m = %d\n", m);
  //printf("chunk = %d\n", chunk);
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

  //printf("Bye from kNN\n");

  return result;

}

void update(knnresult *knnnew, knnresult *knnold){

  int k = knnold->k;
  int m = knnold->m;

  int *indexD = (int *)malloc(2*k*m *sizeof(int));
  double *D = (double *)malloc(2*k*m *sizeof(double));

  for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      indexD[i*2*k+j] = knnold->nidx[i*k+j];
      D[i*2*k+j] = knnold->ndist[i*k+j];
    }
    for(int j=k; j<2*k; j++){
      indexD[i*2*k+j] = knnnew->nidx[i*k-k+j];
      D[i*2*k+j] = knnnew->ndist[i*k-k+j];
    }
  }

  for(int i=0; i<m; i++){
    quicksort(&D[i*2*k], &indexD[i*2*k], 0, 2*k-1);
  }

  for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      knnold->nidx[i*k+j] = indexD[i*2*k+j];
      knnold->ndist[i*k+j] = D[i*2*k+j];
    }
  }

  //free(indexD);
  //free(D);
}

void shiftRight(int *offset, int n){
  int tmp = offset[n-1];
  for(int i=1; i<n; i++){
    offset[i] = offset[i-1];
  }
  offset[0] = tmp;
}

knnresult distrAllkNN(double * X, int n, int d, int k){

// initialize to infinity
  knnresult result;
  //initialize(&result);
  //initialize(&knn);

  int SelfTID, NumTasks, t;
  MPI_Status mpistat;
  MPI_Request mpireq;

  MPI_Comm_size( MPI_COMM_WORLD, &NumTasks );
  int p = NumTasks;
  int num_points = n*d/p; // elements with dimensions per process
  int m = n/p; // elements per process
  if(m/p<=k) exit(1);
  MPI_Comm_rank( MPI_COMM_WORLD, &SelfTID );
  printf("SelfTID = %d\n", SelfTID);


  // offset for indexD
  int *offset  = (int *)malloc(p * sizeof(int));
  for(int i=0; i<p; i++)
    offset[i] = i;

  double *Xlocal = (double *)malloc(num_points * sizeof(double));
  double *Ylocal = (double *)malloc(num_points * sizeof(double));
  double *Zlocal = (double *)malloc(num_points * sizeof(double));
  int yreceiv = 0;
  int *nidx = NULL;
  double *ndist = NULL;
  int *recvcnts = NULL;
  int *displs = NULL;
  double *distbuf = NULL;
  int *idxbuf = NULL;


  // DISTRIBUTION
  if( SelfTID == 0 ) {
    Xlocal = X;
    for(int t=1;t<NumTasks;t++) {
       MPI_Isend(&X[t*num_points],num_points,MPI_DOUBLE,t,0,MPI_COMM_WORLD, &mpireq);
    }
  } else {
    MPI_Recv(Xlocal,num_points,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&mpistat);
  }
  // END DISTRIBUTION

  // BARRIER
  MPI_Barrier(MPI_COMM_WORLD);

  // first iteration
  if(SelfTID != p-1){
    for(int ch=0; ch<num_points; ch++)
      MPI_Isend(Xlocal,num_points,MPI_DOUBLE,SelfTID+1,1,MPI_COMM_WORLD, &mpireq);
  } else {
    for(int ch=0; ch<num_points; ch++)
      MPI_Isend(Xlocal,num_points,MPI_DOUBLE,0,1,MPI_COMM_WORLD, &mpireq);
  }

  // compute previous knn
  knnresult knn = kNN(Xlocal, Xlocal, m, m, d, k);

  for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      knn.nidx[i*k+j] = knn.nidx[i*k+j] + offset[SelfTID]*m;
    }
    printf("\n");
  }

  if(SelfTID != 0){
    MPI_Recv(Ylocal,num_points,MPI_DOUBLE,SelfTID-1,1,MPI_COMM_WORLD, &mpistat);
  } else {
    MPI_Recv(Ylocal,num_points,MPI_DOUBLE,p-1,1,MPI_COMM_WORLD, &mpistat);
  }

  yreceiv = 1;
  MPI_Barrier(MPI_COMM_WORLD);
  // SYNCHRONIZE

  if(NumTasks>2){
    // next iterations except from the last one
    for(int t=1; t<p-1; t++){
      if(yreceiv == 1){

        if(SelfTID != p-1){
          MPI_Isend(Ylocal,num_points,MPI_DOUBLE,SelfTID+1,t+1,MPI_COMM_WORLD,&mpireq);
        } else {
          MPI_Isend(Ylocal,num_points,MPI_DOUBLE,0,t+1,MPI_COMM_WORLD,&mpireq);
        }

        knnresult knnnew = kNN(Xlocal, Ylocal, m, m, d, k);

        shiftRight(offset, p);
        //  index offset
        for(int i=0; i<m; i++){
          for(int j=0; j<k; j++){
            knn.nidx[i*k+j] = knn.nidx[i*k+j] + offset[SelfTID]*m;
          }
        }

        update(&knnnew, &knn);

        if(SelfTID != 0){
          MPI_Recv(Zlocal,num_points,MPI_DOUBLE,SelfTID-1,t+1,MPI_COMM_WORLD, &mpistat);
        } else {
          MPI_Recv(Zlocal,num_points,MPI_DOUBLE,p-1,t+1,MPI_COMM_WORLD, &mpistat);
        }

        yreceiv = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        // SYNCHRONIZE
      }
      else {

        if(SelfTID != p-1){
          MPI_Isend(Zlocal,num_points,MPI_DOUBLE,SelfTID+1,t+1,MPI_COMM_WORLD,&mpireq);
        } else {
          MPI_Isend(Zlocal,num_points,MPI_DOUBLE,0,t+1,MPI_COMM_WORLD,&mpireq);
        }

        knnresult knnnew = kNN(Xlocal, Zlocal, m, m, d, k);

        shiftRight(offset, p);
        //  index offset
        for(int i=0; i<m; i++){
          for(int j=0; j<k; j++){
            knn.nidx[i*k+j] = knn.nidx[i*k+j] + offset[SelfTID]*m;
          }
        }

        update(&knnnew, &knn);

        if(SelfTID != 0){
          MPI_Recv(Ylocal,num_points,MPI_DOUBLE,SelfTID-1,t+1,MPI_COMM_WORLD, &mpistat);
        } else {
          MPI_Recv(Ylocal,num_points,MPI_DOUBLE,p-1,t+1,MPI_COMM_WORLD, &mpistat);
        }

        yreceiv = 1;
        MPI_Barrier(MPI_COMM_WORLD);
        // SYNCHRONIZE
      }
    }
  }

  // COMPUTE LAST KNN
  if(yreceiv == 1){
    knnresult knnnew = kNN(Xlocal, Ylocal, m, m, d, k);

    shiftRight(offset, p);
    //  index offset
    for(int i=0; i<m; i++){
      for(int j=0; j<k; j++){
        knn.nidx[i*k+j] = knn.nidx[i*k+j] + offset[SelfTID]*m;
      }
    }

    update(&knnnew, &knn);
  }
  else{
    knnresult knnnew = kNN(Xlocal, Zlocal, m, m, d, k);

    shiftRight(offset, p);
    //  index offset
    for(int i=0; i<m; i++){
      for(int j=0; j<k; j++){
        knn.nidx[i*k+j] = knn.nidx[i*k+j] + offset[SelfTID]*m;
      }
    }

    update(&knnnew, &knn);
  }

  // deallocate
  free(offset);

  // FILL SEND BUFFERS
  idxbuf = (int *)malloc(sizeof(int) * n*k);
  distbuf = (double *)malloc(sizeof(double) * n*k);

  for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      idxbuf[i*k+j] = knn.nidx[i*k+j];
      distbuf[i*k+j] = knn.ndist[i*k+j];
    }
  }


/*
  for(int i=0; i<m; i++){
    for(int j=0; j<k; j++){
      printf("(%d)%d ", SelfTID, idxbuf[i*k+j]);
    }
    printf("\n");
  }
*/
  MPI_Barrier(MPI_COMM_WORLD);

  // GATHER kNNs
  if(SelfTID == 0){
    nidx = (int *)malloc(sizeof(int) * n*k);
    ndist = (double *)malloc(sizeof(double) * n*k);

    displs = (int *)malloc(sizeof(int) * p);
    recvcnts = (int *)malloc(sizeof(int) * p);
    for(int t=0; t<NumTasks; t++){
      displs[t] = k*m*t;
      recvcnts[t] = k*m;
    }
  }
  // ERRORS
  MPI_Gatherv(idxbuf,   k*m, MPI_INT, nidx, recvcnts, displs, MPI_INT, 0, MPI_COMM_WORLD);

  if(SelfTID == 0){
  for(int i=0; i<n; i++){
    for(int j=0; j<k; j++){
      printf("(%d)%d ", SelfTID, idxbuf[i*k+j]);
    }
    printf("\n");
  }}

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gatherv(distbuf,  k*m, MPI_DOUBLE, ndist, recvcnts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(distbuf);
  free(idxbuf);
/*
  if(SelfTID != 0){
    MPI_Isend(knn.ndist,  k*m,  MPI_DOUBLE, 0,  150,  MPI_COMM_WORLD, &mpireq);
    MPI_Isend(knn.nidx,   k*m,     MPI_INT, 0,  140,  MPI_COMM_WORLD, &mpireq);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if(SelfTID == 0){
    for(int i=0; i<m; i++){
      for(int j=0; j<k; j++){
        ndist[i*k+j] = knn.ndist[i*k+j];
        nidx[i*k+j] = knn.nidx[i*k+j];
      }
    }

    for(int t=1; t<p; t++){
      MPI_Irecv(&ndist[t*m*k],  k*m,  MPI_DOUBLE, t,  150,  MPI_COMM_WORLD, &mpireq);
      MPI_Irecv(&nidx[t*m*k],   k*m,     MPI_INT, t,  140,  MPI_COMM_WORLD, &mpireq);
    }
  }
*/

  //free(nidx_buf);
  //free(ndist_buf);


  if(SelfTID == 0){
    result.m = n;
    result.k = k;
    allocate(&result);
    result.ndist = ndist;
    result.nidx = nidx;
    free(displs);
    free(recvcnts);
  } else {
    result.ndist = NULL;
    result.nidx = NULL;
    result.m = 0;
    result.k = 0;

  }

  free(Xlocal);
  free(Ylocal);
  free(Zlocal);
  printf("Bye from distrAllkNN\n");
  return result;


}
