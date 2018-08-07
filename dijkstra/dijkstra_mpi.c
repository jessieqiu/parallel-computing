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
#include <mpi.h>

#define ROWMJR(R,C,NR,NC) (R*NC+C)
#define COLMJR(R,C,NR,NC) (C*NR+R)
/* define access directions for matrices */
#define a(R,C) a[ROWMJR(R,C,ln,n)]
//#define b(R,C) b[ROWMJR(R,C,nn,n)]
#define d(R,C) d[ROWMJR(R,C,ln,n)]


static void
load(
	char const * const filename,
	int * const np,
	float ** const ap
)
{
	int i, j, n, ret;
	FILE * fp = NULL;
	float * a;

	/* open the file */
	fp = fopen(filename, "r");
	assert(fp);

	/* get the number of nodes in the graph */
	ret = fscanf(fp, "%d", &n);
	assert(1 == ret);

	/* allocate memory for local values */
	a = malloc(n*n * sizeof(*a));
	assert(a);

	/* read in roots local values */
	for (i = 0; i<n; ++i) {
		for (j = 0; j<n; ++j) {
			ret = fscanf(fp, "%f", &a(i, j));
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
my_load(
	char const * const filename,
	int * const np,
	float ** const ap,
	const int n_nodes,
	const int my_rank
)
{
 
  int n;
  float * a;

  if (my_rank == 0) {
	  int i, j, ret;
	  
	  float * d;
	  /* open the file */
	  FILE * fp = NULL;
	  fp = fopen(filename, "r");
	  assert(fp);

	  /* get the number of nodes in the graph */
	  ret = fscanf(fp, "%d", &n);
	  assert(1 == ret);
	  //broadcast n
	  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	  int remainder = n % n_nodes;
	  int len = n / n_nodes;
	  /* allocate memory for local values */
	  a = malloc((len+ remainder) * n * sizeof(*a));
	  assert(a);
	  d = malloc((len) * n * sizeof(*d));
	  assert(d);
	  int count = 0;
	  for (i = 0; i < len + remainder; ++i) {
		  for (j = 0; j < n; ++j) {
			  ret = fscanf(fp, "%f", &a(i, j));
			  assert(1 == ret);
		  }
		  count++;
	  }
	  //printf("count1 %d \n", count);
	  for (int p = 1;p < n_nodes;++p) {
		 
		 // printf("what is p %d \n", p);

		  for (i = 0; i < len; ++i) {
			  for (j = 0; j < n; ++j) {
				  ret = fscanf(fp, "%f", &d(i, j));
				  assert(1 == ret);
			  }
			  //printf("p, i %d %d \n",p, i);
			  count++;
		  }
		  
		  //printf("send to %d %f \n", p, d(10, 10));
		  MPI_Send(d, len*n, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
	  }
	 // printf("count2 %d \n", count);
	  
	  *ap = a;
	  free(d);

	  /* close file */
	  ret = fclose(fp);
	  assert(!ret);
  }
  else {
//get n first
	  MPI_Status status;
	  
	  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	  int len = n / n_nodes;

	  /*if (my_rank == 1) {
		  printf("node 1, n is %d, len is %d \n", n, len);
	  }*/

	  /* allocate memory for local values */
	  a = malloc(len * n * sizeof(*a));
	  assert(a);  
	  
	  MPI_Recv(a, len*n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
	  // MPI_STATUS_IGNORE
	  //printf("receive %d %f \n", my_rank, a(10, 10));
	  *ap = a;
  }
  /* record output values */
  *np = n;
  
}

static void print_array(int const n, int const * const numbers) {

	for (int i = 0; i<n; ++i) {
		printf("array i %d %d \n", i, numbers[i]);
	}
}


static void
dijkstra(
  int const s,
  int const n,
  float const * const a,
  float ** const lp,
  const int n_nodes,
  const int my_rank
)
{
  int i, j;
  struct float_int {
    float l;
    int u;
  } min;
  int * m;
  float * l;
  double t1, t2, t3, t4, t5;
  
  //t3 = 0.0;t4 = 0.0;t5 = 0.0;

  t1 = MPI_Wtime();

  m = calloc(n, sizeof(*m));
  assert(m);

  l = malloc(n*sizeof(*l));
  assert(l);

  float* global_l;
  global_l = malloc(n * sizeof(*global_l));
  assert(global_l);
  
  float * local_min;
  local_min = malloc(2 * sizeof(*local_min));

  float * local_min_list;
  local_min_list = malloc(n_nodes* 2 * sizeof(*local_min_list));
 
  float * global_min;
  global_min = malloc(2 * sizeof(*global_min));

  int remainder = n % n_nodes;
  int len = n / n_nodes;
 
  int size = len;
  int m_j = -1;
  if (my_rank == 0) {
	  size = len + remainder;
	  m_j = 0;

  }
  else { m_j = remainder + len * my_rank; }

 
  //printf("my rank %d my size %d my m_j %d \n", my_rank, size, m_j);


  for (i=0; i<size; ++i) {
    l[i] = a(i,s);
	//printf("l %f \n", l[i]);
  }

  m[s] = 1;
  min.u = -1; /* avoid compiler warning */

  for (i=1; i<n; ++i) {
    min.l = INFINITY;


	if(i==5)
		t3 = MPI_Wtime();

    /* find local minimum */
    for (j=0; j<size; ++j) {
		//m_j = ((bool)my_rank)*remainder + len * my_rank;
		/*if(my_rank ==0 &&i==1 || my_rank == 2 && i == 5)
			printf("index %d \n", j+m_j);*/

		if (!m[j+ m_j] && l[j] < min.l) {
        min.l = l[j];  //float
        min.u = j + m_j;   //int  global index !!!
      }
    }

	local_min[0] = min.l;
	local_min[1] = (float) min.u;

	MPI_Gather(local_min, 2, MPI_FLOAT, local_min_list, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	if (my_rank == 0) {
		//for (int tp = 0; tp < n_nodes ; ++tp) {
			//printf("i: %d, gather %f %f\n",i, local_min_list[tp*2], local_min_list[tp * 2+1]);
		//}
	
		
		//find global minimum
		global_min[0] = local_min_list[0];  //float
		global_min[1] = local_min_list[1];   //int, j
		for (int g = 1;g < n_nodes;++g) {
			if (local_min_list[g * 2] < global_min[0]) {
				global_min[0] = local_min_list[g * 2];
				global_min[1] = local_min_list[g * 2 + 1];
			}
		}
		//printf("global min %f %f \n", global_min[0], global_min[1]);
		MPI_Bcast(global_min, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
			
	}
	else {
		MPI_Bcast(global_min, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	//assign min.u and min.l

	min.l = global_min[0];
	min.u =(int) global_min[1];
	
	//printf("min.u %d \n", min.u);
	

    m[min.u] = 1;

	//printf("m char %d \n", m[min.u]);
	

	if (i == n-1 && my_rank==1) {
		
		//print_array(10, m);
		t4 = MPI_Wtime();
	}

    for (j=0; j<size; ++j) {
      if (!m[j + m_j] && min.l+a(j,min.u) < l[j])
        l[j] = min.l+a(j,min.u);
    }

	/*if (i == n-1 && my_rank == 2) {
		for (int tee = 0;tee < size;++tee) {
			printf("L array i %d %f \n", tee, l[tee]);
		}
	}*/
	// gather l
	

	

	if (i == 5) {
		t5 = MPI_Wtime();
		//printf("inside for time used: %f %f %f \n",t4-t3,t5-t4,t5-t3);
		//first one only use 5% time of second one
	}

  }//end of for loop

   /*MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
   void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype
   recvtype,
   int root, MPI_Comm comm)*/
  /*int *rcounts, *displs;
  
  for (int q = 0; q<n_nodes; ++q) {
	  displs[q] = 0;
	  rcounts[q] = len;
  }

  rcounts[0] = len + remainder;*/

  //MPI_Gatherv(l, size, MPI_FLOAT, global_l, rcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Gather(l, size, MPI_FLOAT, global_l, size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  //MPI_Gather(l, len, MPI_FLOAT, global_l, len, MPI_FLOAT, 0, MPI_COMM_WORLD); //then change first one
  /*if (my_rank == 0) {
	  for (int te = 0;te < 100;++te) {
		  printf("final array i %d %f \n", te, global_l[te]);
	  }
  }*/

  t2 = MPI_Wtime();
  //printf("Total function time used: %f  from rank %d \n", t2 - t1, my_rank);
    

  *lp = global_l;

  free(m);

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

  MPI_Init(NULL, NULL);

  int world_rank;
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /*if (world_rank == 0 || world_rank == 1)
  printf("in main n is %d \n", n);*/

  //load(argv[1], &n, &a);
  my_load(argv[1], &n, &a, world_size, world_rank);

 
  double starttime, endtime;
  //ts = clock();
  starttime = MPI_Wtime();
 
  
  dijkstra(atoi(argv[2]), n, a, &l, world_size, world_rank);
 
 // te = clock();
  endtime = MPI_Wtime();

  if (world_rank == 0) {

	 // print_time((double)(te - ts) / CLOCKS_PER_SEC);
	  printf("using mpi_wtime: ");
	  print_time(endtime - starttime);
	  //put it back here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  print_numbers(argv[3], n, l);
  }


  MPI_Finalize();
  free(a);
  free(l);
  
  return EXIT_SUCCESS;
}
