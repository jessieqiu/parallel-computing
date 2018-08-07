/* assert */
#include <assert.h>
/* INFINITY */
#include <math.h>
/* FILE, fopen, fclose, fscanf, rewind */
#include <stdio.h>
/* EXIT_SUCCESS, malloc, calloc, free */
#include <stdlib.h>
/* time, CLOCKS_PER_SEC */
#include <time.h>
#include <omp.h>
//NR is number of rows, NC is number of columns, R is row id, C is column id
#define ROWMJR(R,C,NR,NC) (R*NC+C)
#define COLMJR(R,C,NR,NC) (C*NR+R)
/* define access directions for matrices */
#define a(R,C) a[ROWMJR(R,C,ln,n)]
#define b(R,C) b[ROWMJR(R,C,nn,n)]

static void
load(
  char const * const filename,
  int * const np,
  float ** const ap
)
{
  int i, j, n, ret;
  FILE * fp=NULL;
  float * a;

  /* open the file */
  fp = fopen(filename, "r");
  assert(fp);

  /* get the number of nodes in the graph */
  ret = fscanf(fp, "%d", &n);
  assert(1 == ret);

  /* allocate memory for local values */
  a = malloc(n*n*sizeof(*a));
  assert(a);

  /* read in roots local values */
  for (i=0; i<n; ++i) {
    for (j=0; j<n; ++j) {
      ret = fscanf(fp, "%f", &a(i,j));
      assert(1 == ret);
    }
  }

  /* close file */
  ret = fclose(fp);
  assert(!ret);

  /* record output values */
  *np = n;
  *ap = a;
}

static void
dijkstra(
  int const s,   //source
  int const n,   //number or nodes
  float const * const a,
  float ** const lp
)
{
	double t1, t2, t3, t4,t5;
	t1 = omp_get_wtime();

  int i, j;
  struct float_int {
    float l;
    int u;
  } min;
  char * m;
  float * l;
  

  

  m = calloc(n, sizeof(*m));
  assert(m);

  l = malloc(n*sizeof(*l));
  assert(l);

 
//#pragma omp parallel for
  for (i=0; i<n; ++i) {
	  l[i] = a(s, i); // = a(i,s);	
  }

  //if (i == 0 && omp_get_thread_num() == 0)
	 // printf("number of threads: %d \n", omp_get_num_threads());

  m[s] = 1;
  min.u = -1; /* avoid compiler warning */

  t2 = omp_get_wtime();
  float temp;
  int threads;
  int *js;
  int p = 0;
  js = calloc(28, sizeof(*js));
  assert(js);

  for (i = 1; i < n; ++i) {
	  //min.l = INFINITY;
	  temp = INFINITY;
	  /* find local minimum */
	  #pragma omp parallel for reduction(min:temp) firstprivate(p)
	  for (j = 0; j < n; ++j) {
		  if (i==1 && j == 0) {
			  if (omp_get_thread_num() == 0) {
				  threads = omp_get_num_threads();
				 // printf("number of threads in loop 1: %d \n", threads);
				  			 
			  }
			  
		  }
		  if (!m[j] && l[j] < temp) {
			  //	if(!m[j]){
			  //		min.l = min(min.l, l[j]);
			  temp = l[j];
			 // min.u = j;

			  p = omp_get_thread_num();
			 // printf("what is p %d \n", p);
			  js[p] = j;
			  
		  }

	  }
	 /* if (i == 1)
		  printf("min of temp %f \n", temp);*/
	  t4 = omp_get_wtime();


	  for (j = 0; j < threads;++j) {
		  
		  if (!m[js[j]] && l[js[j]] == temp) {
			  min.u = js[j];
			  
		  }
	  }
	  t5 = omp_get_wtime();
	  if(i==10) //check different i
		//printf("time used to look for j: %f \n", t5-t4);

	  m[min.u] = 1;
	  min.l = temp;

	  //float temp;
#pragma omp parallel for 
	  for (j = 0; j < n; ++j) {
		  if (!m[j] && min.l + a(j, min.u) < l[j])
			  l[j] = min.l + a(j, min.u);
		 
	  }  
  }
  t3 = omp_get_wtime();
  
  free(m);

  *lp = l;

  
//  printf("time used: %f %f \n", t2 - t1, t3 - t2);
}

static void
print_time(double const seconds)
{
  printf("Operation Time: %0.04fs\n", seconds);
}

static void
print_numbers(
  char const * const filename,
  int const n,
  float const * const numbers)
{
  int i;
  FILE * fout;

  /* open file */
  if(NULL == (fout = fopen(filename, "w"))) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write numbers to fout */
  for(i=0; i<n; ++i) {
    fprintf(fout, "%10.4f\n", numbers[i]);
  }

  fclose(fout);
}

int
main(int argc, char ** argv)
{
  int n;
  clock_t ts, te;
  float * a, * l;

  if(argc < 4){
     printf("Invalid number of arguments.\nUsage: dijkstra <graph> <source> <output_file>.\n");
     return EXIT_FAILURE;
  }


  load(argv[1], &n, &a);
  double omp_ss, omp_ee;
// source node, number of nodes, a is adjacency matrix, l is your output
 // ts = clock();
  omp_ss = omp_get_wtime();
  dijkstra(atoi(argv[2]), n, a, &l);
//  te = clock();
  omp_ee = omp_get_wtime();
//  print_time((double)(te-ts)/CLOCKS_PER_SEC);
 // print_time(omp_ee - omp_ss);
  printf("omp time %f \n", omp_ee - omp_ss);
  print_numbers(argv[3], n, l);
  
  //arg 3 is output file 

  free(a);
  free(l);

  return EXIT_SUCCESS;
}
