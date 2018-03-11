/* errno */
#include <errno.h>

/* fopen, fscanf, fprintf, fclose */
#include <stdio.h>

/* EXIT_SUCCESS, EXIT_FAILURE, malloc, free */
#include <stdlib.h>

#include <omp.h>
#include <math.h>

static int load_mat(char const * const fname, size_t * const np,
                    size_t * const mp, double ** const matp)
{
  size_t i, j, n, m;
  double * mat=NULL;
  FILE * fp;

  if (!(fp=fopen(fname, "r"))) {
    goto failure;
  }

  if (2 != fscanf(fp, "%zu %zu", &n, &m)) {
    goto cleanup;
  }

  if (!(mat=malloc(n*m*sizeof(*mat)))) {
    goto cleanup;
  }

  for (i=0; i<n; ++i) {
    for (j=0; j<m; ++j) {
      if (feof(fp)) {
        goto cleanup;
      }
      fscanf(fp, "%lf", mat+i*m+j);
    }
  }

  if (fclose(fp)) {
    goto failure;
  }

  *np   = n;
  *mp   = m;
  *matp = mat;

  return 0;

  cleanup:
  free(mat);
  fclose(fp);

  failure:
  return -1;
}


static int save_mat(char const * const fname, size_t const n,
                    size_t const m, double const * const mat)
{
  size_t i, j;
  FILE * fp;

  if (!(fp=fopen(fname, "w"))) {
    goto failure;
  }

  fprintf(fp, "%zu %zu\n", n, m);

  for (i=0; i<n; ++i) {
    for (j=0; j<m; ++j) {
      fprintf(fp, "%10.4lf ", mat[i*m+j]);
    }
    fprintf(fp, "\n");
  }

  if (fclose(fp)) {
    goto failure;
  }

  return 0;

  failure:
  return -1;
}

void small_matrix_multiplication(double const * const A, double const * const B, double *C, size_t const i, size_t const j, size_t const k, size_t const tileX, size_t const tileY, size_t const tileZ, size_t const n, size_t const m, size_t const p)
{
	double mysum;

	for (size_t x = i; x < i + tileX - 1 && x < n; ++x) {
		for (size_t y = j; y < j + tileY - 1 && y < p; ++y) {
			mysum = 0.0;
			for (size_t z = k; z < k + tileZ - 1 && z < m; ++z) {
				mysum += A[x*m + z] * B[z*p + y];
			}			
			C[x*p + y] += mysum;
		}
	}
}

void strassen() {

	//strassen, why is it slower?
	/*
	x = i;
	y = j;
	z = k;

	double M1, M2, M3, M4, M5, M6, M7;
	double C11, C12, C21, C22;
	double A11, A12, A21, A22, B11, B12, B21, B22;


	//cout <<"x y z"<< x << " " << y << " " << z<<endl;

	A11 = A[x*m + y];
	A12 = A[x*m + y + 1];
	A21 = A[(x + 1)*m + y];
	A22 = A[(x + 1)*m + y + 1];

	B11 = B[y*p + z];
	B12 = B[y*p + z + 1];
	B21 = B[(y + 1)*p + z];
	B22 = B[(y + 1)*p + z + 1];

	M1 = (A11 + A22)*(B11 + B22);
	M2 = (A21 + A22)*B11;
	M3 = A11 * (B12 - B22);
	M4 = A22 * (B21 - B11);
	M5 = (A11 + A12)*B22;
	M6 = (A21 - A11)*(B11 + B12);
	M7 = (A12 - A22)*(B21 + B22);

	C11 = M1 + M4 - M5 + M7;
	C12 = M3 + M5;
	C21 = M2 + M4;
	C22 = M1 - M2 + M3 + M6;

	C[x*p + z] += C11;
	C[x*p + z + 1] += C12;
	C[(x + 1)*p + z] += C21;
	C[(x + 1)*p + z + 1] += C22;

	//slightly faster
	//C[x*p + z] += M1 + M4 - M5 + M7;
	//C[x*p + z + 1] += M3 + M5;
	//C[(x + 1)*p + z] += M2 + M4;
	//C[(x + 1)*p + z + 1] += M1 - M2 + M3 + M6;

	*/
}

static int mult_mat(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  size_t i, j, k;
  double sum;
  double * C=NULL;
  double time_used, point1,point2, point3;
  unsigned short threads;
  double * D = NULL; 
  
  _Bool correct;
  size_t tileX, tileY, tileZ;
  size_t x, y, z;
  double mysum;
  int count,count_another;

  omp_lock_t * C_locks = NULL;
  

  if (!(C=malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  if (!(C_locks = malloc(n*p * sizeof(*C_locks)))) {
	  goto cleanup;
  }  

  if (!(D = (double *)malloc(n*p * sizeof(*C)))) {
  goto cleanup;
  }

  point1 = omp_get_wtime();

  //original serial code

 /* for (i=0; i<n; ++i) {
    for (j=0; j<p; ++j) {
      for (k=0,sum=0.0; k<m; ++k) {
        sum += A[i*m+k] * B[k*p+j];
      }
      D[i*p+j] = sum;
    }
  }*/
  
  

  
  
//this works, easiest version
//#pragma omp parallel for private(j,k,sum) 
//  for (i = 0; i<n; ++i) {
//	  for (j = 0; j<p; ++j) {
//		  for (k = 0, sum = 0.0; k<m; ++k) {
//			  sum += A[i*m + k] * B[k*p + j];
//		  }
//		  D[i*p + j] = sum;
//	  }
//  }

 

  //time_used = point2 - point1;
  //printf("serial time:  %f  \n", time_used);
  

 
  tileY = 200; 
 
  tileX = n * (4096 / tileY) / (n + p); //to fit L1
  if (tileX < 1) { tileX = 1; }
  tileZ = p * (4096 / tileY) / (n + p); //to fit L1
  if (tileZ < 1) { tileZ = 1; }

  //printf("tile size %d, %d, %d  \n", tileX, tileY, tileZ);

  
#pragma omp parallel for 
  for (int w = 0;w < n*p;w++) {
	  omp_init_lock(&C_locks[w]);
	  C[w] = 0.0;
  }

point2 = omp_get_wtime();

#pragma omp parallel  
 {
#pragma omp master	
	 {
		 for (j = 0; j < m; j = j + tileY)
		 {  //for matrix A, do it vertically, when you move down, all the data needed for B is still in cache, you only need to read a small chunk of data for A)


	//#pragma omp parallel
	//#pragma omp for nowait schedule(auto) private(i,k,x,y,z,mysum) firstprivate(j)   //no wait schedule(static,1)

//#pragma omp task private(i,k,x,y,z,mysum) firstprivate(j) 
			 for (i = 0; i < n; i = i + tileX)
			 {
				 //#pragma omp for nowait private(x,y,z,mysum) firstprivate(i,j)
#pragma omp task private(k,x,y,z,mysum) firstprivate(i,j) 

				 for (k = 0;k < p; k = k + tileZ) {   
					 
					 //matrix multiplication calculation for two small tiles
					 for (x = i; x < i + tileX && x < n; ++x) {
						
						 for (y = k; y < k + tileZ && y < p; ++y) {  
							 
							 mysum = 0;
							 for (z = j; z < j + tileY && z < m; ++z) {   
								  
								 mysum += A[x*m + z] * B[z*p + y];
							 }
							 //should put a lock here, race condition unlikely to happen
							 omp_set_lock(&C_locks[x*p + y]);
							 C[x*p + y] += mysum;
							 omp_unset_lock(&C_locks[x*p + y]);

							 
						 }
					 } //end of small tile calculation loop					

				 }
			 }
		 }
	 }
 }
 
 
 

  point3 = omp_get_wtime();
  time_used = point3 - point2;
  printf("multiplication time used: %f  \n", time_used);


#pragma omp parallel for
  for (int w = 0;w < n*p;w++) {
	  omp_destroy_lock (&C_locks[w]);
  }

  


 //for comparing results because diff does not seem to work well


 //correct = 1;
 //count = 0; //wrong number count

 // for (i = 0; i < n*p; ++i) {  //n*p size

	//  if (fabs(C[i] - D[i]) > 0.0001) {
	//	  correct = 0;
	//	  count++;
	//	  if (count < 100) {
	//		  fprintf(stderr, "wrong? i %d ", i);			
	//		  fprintf(stderr, "difference: %f: \n", D[i] - C[i]);
	//	  }
	//  }	
 // }
 // 
 // fprintf(stderr, "correct? %d\n", correct);
 // fprintf(stderr, "wrong number count? %d\n", count);
 //



  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}


int main(int argc, char * argv[])
{
  // size_t stored an unsigned integer
  size_t n, m, p, mm;
  double * A=NULL, * B=NULL, * C=NULL;
  double start, end, time;

  if (argc != 4) {
    fprintf(stderr, "usage: matmult A.mat B.mat C.sol\n");
    goto failure;
  }
  
  start = omp_get_wtime();

  if (load_mat(argv[1], &n, &m, &A)) {
    perror("error");
    goto failure;
  }
  

  if (load_mat(argv[2], &mm, &p, &B)) {
    perror("error");
    goto failure;
  }
  
  if (m != mm) {
    fprintf(stderr, "dimensions do not match: %zu x %zu, %zu x %zu\n",
      n, m, mm, p);
  }  

  if (mult_mat(n, m, p, A, B, &C)) {
    perror("error");
    goto failure;
  }

 
  
  if (save_mat(argv[3], n, p, C)) {
    perror("error");
    goto failure;
  }
  
    
  end = omp_get_wtime();
  time = end - start;
  printf("total time used: %f  \n", time);
  printf("\n");

  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;

  failure:
  free(A);
  free(B);
  free(C);
  return EXIT_FAILURE;
}
