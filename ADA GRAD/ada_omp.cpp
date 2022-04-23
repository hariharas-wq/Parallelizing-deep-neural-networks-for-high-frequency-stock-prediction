#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string.h>

#ifdef OPENMP
#include <omp.h>
#endif

using namespace std;


#define NUM_NODES 256


#define VISUALIZE 1


#define FREQUENCY 50


#define TOLERANCE 1e-12



void allocate_arrays(double **A, int **rowA, int **colA, double **x, double **b,
                     int n_nodes);


void conjugate_gradient(double *A, int *rowA, int *colA, double *x, double *b,
                        int n_nodes, double tol, bool viz, int freq);


void sp_mv_product(double *result, double *A, int *rowA, int *colA, double *x,
                   int n);


void daxpy(double *result, double alpha, double *x, double *y, int n);


double dot_product(double *x, double *y, int n);


double vector_norm(double *vec, int n);


void vector_copy(double *vec_in, double *vec_out, int n);

double write_output(double *x, int n_nodes);


void deallocate_arrays(double *A, int *rowA, int *colA, double *x, double *b);



int main(int argc, char* argv[]) {



  bool viz = VISUALIZE;
  int freq = FREQUENCY, n_nodes = NUM_NODES, n = (n_nodes-2)*(n_nodes-2);
  double tol = TOLERANCE, center_sol, StartTime, StopTime, UsedTime;



#ifdef OPENMP
  int n_threads = omp_get_num_procs();
  if (argc != 2){
    printf( "\n   !!! Error !!!\n" );
    printf( " Incorrect thread number specification.\n");
    printf( " Proper usage: ./cg_openmp [n_threads]\n\n");
    exit(1);
  } else {
    n_threads = atoi(argv[1]);
  }
  omp_set_num_threads(n_threads);
#endif



  printf("\n");
  printf("#============================================#\n");
  printf("# ADA Gradient (AG) Poisson solver.    #\n");
  printf("#============================================#\n\n");



  double *A, *x, *b;
  int *rowA, *colA;



  allocate_arrays(&A, &rowA, &colA, &x, &b, n_nodes);



  StartTime = double(clock())/double(CLOCKS_PER_SEC);
#ifdef OPENMP
  StartTime = omp_get_wtime();
#endif



  conjugate_gradient(A, rowA, colA, x, b, n, tol, viz, freq);



#ifdef OPENMP
  StopTime = omp_get_wtime();
  UsedTime = StopTime-StartTime;
#else
  StopTime = double(clock())/double(CLOCKS_PER_SEC);
  UsedTime = StopTime-StartTime;
#endif



  if (viz) center_sol = write_output(x, n_nodes);



  deallocate_arrays(A, rowA, colA, x, b);



#ifdef OPENMP
  cout << "\nCompleted in " << fixed << UsedTime << " seconds with ";
  cout << n_threads << " threads." << endl;
#else
  cout << "\nCompleted in " << fixed << UsedTime;
  cout << " seconds in parallel." << endl;
#endif


  if (viz) {
    printf("Solution at domain center: %1.15f\n", center_sol);
    printf("#============================================#\n\n");
  } else printf("#============================================#\n\n");

  return 0;
}

void allocate_arrays(double **A, int **rowA, int **colA, double **x,
                     double **b, int n_nodes) {



  int offset = (n_nodes-2);
  int n_unknowns = offset*offset;
  int *rowA_temp, *colA_temp;
  double *A_temp, *x_temp, *b_temp;



  double h2 = pow(1.0/((double)n_nodes-1.0),2.0);


  x_temp = new double[n_unknowns];
  b_temp = new double[n_unknowns];



  for (int i = 0; i < n_unknowns; i++) {
    x_temp[i]  = 0.0;
    b_temp[i]  = 1.0*h2;
  }



  rowA_temp = new int[n_unknowns+1];
  int i_unknown = 0, nnz = 0;
  rowA_temp[0] = nnz;

  for (int i = 0; i < offset; i++) {
    for (int j = 0; j < offset; j++) {



      if (i == 0 && (j == 0 || j == offset-1))
        nnz += 3;
      else if (i == offset-1 && (j == 0 || j == offset-1))
        nnz += 3;
      else if (i == 0 || i == offset-1)
        nnz += 4;
      else if (j == 0 || j == offset-1)
        nnz += 4;
      else
        nnz += 5;



      rowA_temp[i_unknown+1] = nnz;
      i_unknown++;

    }
  }



  A_temp    = new double[nnz];
  colA_temp = new int[nnz];


  int row_start, k = 0;

  for (int i = 0; i < offset; i++) {
    for (int j = 0; j < offset; j++) {


      row_start = rowA_temp[k];


      if (i == 0 && j == 0) {

        A_temp[row_start]   = -4;
        A_temp[row_start+1] = 1;
        A_temp[row_start+2] = 1;

        colA_temp[row_start]   = k;
        colA_temp[row_start+1] = k+1;
        colA_temp[row_start+2] = k+offset;

      } else if (i == 0 && j == offset-1) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = -4;
        A_temp[row_start+2] = 1;

        colA_temp[row_start]   = k-1;
        colA_temp[row_start+1] = k;
        colA_temp[row_start+2] = k+offset;

      } else if (i == offset-1 && j == 0) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = -4;
        A_temp[row_start+2] = 1;

        colA_temp[row_start]   = k-offset;
        colA_temp[row_start+1] = k;
        colA_temp[row_start+2] = k+1;

      } else if (i == offset-1 && j == offset-1) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = 1;
        A_temp[row_start+2] = -4;

        colA_temp[row_start]   = k-offset;
        colA_temp[row_start+1] = k-1;
        colA_temp[row_start+2] = k;

      } else if (i == 0) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = -4;
        A_temp[row_start+2] = 1;
        A_temp[row_start+3] = 1;

        colA_temp[row_start]   = k-1;
        colA_temp[row_start+1] = k;
        colA_temp[row_start+2] = k+1;
        colA_temp[row_start+3] = k+offset;

      } else if (i == offset-1) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = 1;
        A_temp[row_start+2] = -4;
        A_temp[row_start+3] = 1;

        colA_temp[row_start]   = k-offset;
        colA_temp[row_start+1] = k-1;
        colA_temp[row_start+2] = k;
        colA_temp[row_start+3] = k+1;

      } else if (j == 0) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = -4;
        A_temp[row_start+2] = 1;
        A_temp[row_start+3] = 1;

        colA_temp[row_start]   = k-offset;
        colA_temp[row_start+1] = k;
        colA_temp[row_start+2] = k+1;
        colA_temp[row_start+3] = k+offset;

      } else if ( j == offset-1) {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = 1;
        A_temp[row_start+2] = -4;
        A_temp[row_start+3] = 1;

        colA_temp[row_start]   = k-offset;
        colA_temp[row_start+1] = k-1;
        colA_temp[row_start+2] = k;
        colA_temp[row_start+3] = k+offset;

      } else {

        A_temp[row_start]   = 1;
        A_temp[row_start+1] = 1;
        A_temp[row_start+2] = -4;
        A_temp[row_start+3] = 1;
        A_temp[row_start+4] = 1;

        colA_temp[row_start]   = k-offset;
        colA_temp[row_start+1] = k-1;
        colA_temp[row_start+2] = k;
        colA_temp[row_start+3] = k+1;
        colA_temp[row_start+4] = k+offset;

      }



      k++;

    }
  }


  *A    = A_temp;
  *rowA = rowA_temp;
  *colA = colA_temp;
  *x    = x_temp;
  *b    = b_temp;

}

void conjugate_gradient(double *A, int *rowA, int *colA, double *x, double *b,
                        int n, double tol, bool viz, int freq) {



  int iter = 0;
  double resid, resid0, resid2, alpha, beta;



  double *r = new double[n];
  double *p = new double[n];
  double *mv_product = new double[n];



  sp_mv_product(mv_product, A, rowA, colA, x, n);
  daxpy(r, -1.0, mv_product, b, n);
  vector_copy(r, p, n);



  resid  = vector_norm(r,n)/vector_norm(b,n); resid0 = resid;
  if (viz) {
    printf("  Iteration         Residual \n");
    printf("  -------------------------- \n");
    printf("    %6d     %13e \n", iter, resid/resid0);
  }

  while ( resid/resid0 > tol && iter < n ) {



    iter  = iter + 1;

    sp_mv_product(mv_product, A, rowA, colA, p, n);


    resid2 = dot_product(r, r, n);
    alpha  = resid2/dot_product(p, mv_product, n);


    daxpy(x, alpha, p, x, n);


    daxpy(r, -1.0*alpha, mv_product, r, n);


    beta = dot_product(r, r, n)/resid2;


    daxpy(p, beta, p, r, n);



    resid = vector_norm(r,n)/vector_norm(b,n);


    if (viz && (iter%freq == 0))
      printf("    %6d     %13e \n", iter, resid/resid0);

  }



  if (viz && (iter%freq != 0))
    printf("    %6d     %13e \n", iter, resid/resid0);



  delete [] r;
  delete [] p;
  delete [] mv_product;

}

void sp_mv_product(double *result, double *A, int *rowA, int *colA, double *x,
                   int n) {


  int row_ind, index;

#ifdef OPENMP
#pragma omp parallel for default(none) \
private(row_ind,index) shared(result,A,rowA,colA,x,n)
#endif
  for (row_ind = 0; row_ind < n; row_ind++) {
    result[row_ind] = 0.0;
    for (index = rowA[row_ind]; index < rowA[row_ind+1]; index++) {
      result[row_ind] += A[index]*x[colA[index]];
    }
  }

}

void daxpy(double *result, double alpha, double *x, double *y, int n) {



  int i;

#ifdef OPENMP
#pragma omp parallel for default(none) \
private(i) shared(result,alpha,x,y,n)
#endif
  for (i = 0; i < n; i++) {
    result[i] = alpha*x[i] + y[i];
  }

}


double dot_product(double *x, double *y, int n) {



  double result = 0.0; int i;

#ifdef OPENMP
#pragma omp parallel for default(none) \
private(i) shared(x,y,n) ordered reduction(+:result)
#endif
  for (i = 0; i < n; i++) {
    result = result + x[i]*y[i];
  }

  return result;
}


double vector_norm(double *vec, int n) {



  double norm = 0.0; int i;

#ifdef OPENMP
#pragma omp parallel for default(none) \
private(i) shared(vec,n) ordered reduction(+:norm)
#endif
  for (i = 0; i < n; i++) {
    norm += vec[i]*vec[i];
  }
  norm = sqrt(norm);

  return norm;
}


void vector_copy(double *vec_in, double *vec_out, int n) {


  int i;

#ifdef OPENMP
#pragma omp parallel for default(none) \
private(i) shared(vec_out,vec_in,n)
#endif
  for (i = 0; i < n; i++) {
    vec_out[i] = vec_in[i];
  }

}

double write_output(double *x, int n_nodes) {



  char cstr[200];
  double center_sol = 0.0;

  strcpy(cstr, "solution.dat");
  ofstream sol_file;
  sol_file.precision(15);
  sol_file.open(cstr, ios::out);
  sol_file << "TITLE = \"Visualization of a Poisson solution\"" << endl;
  sol_file << "VARIABLES = \"x\", \"y\", \"f\"";
  sol_file << endl;
  sol_file << "ZONE DATAPACKING=BLOCK I=" << n_nodes << ", J=" << n_nodes;
  sol_file << endl;



  for (int j = 0; j < n_nodes; j++) {
    for (int i = 0; i < n_nodes; i++) {
      sol_file << (double)i/(double)(n_nodes-1) << " ";
    }
    sol_file << endl;
  }


  for (int j = n_nodes-1; j >= 0; j--) {
    for (int i = 0; i < n_nodes; i++) {
      sol_file <<  (double)j/(double)(n_nodes-1) << " ";
    }
    sol_file << endl;
  }


  double **solution = new double*[n_nodes];
  for (int i = 0; i < n_nodes; i++) {
    solution[i] = new double[n_nodes];
  }



  int k = 0, center = n_nodes/2;
  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      if (i == 0 || j == 0 || i == n_nodes-1 || j == n_nodes-1) {
        solution[i][j] = 0.0;
      } else {
        solution[i][j] = x[k];
        if (i == center && j == center) center_sol = x[k];
        k++;
      }
    }
  }



  for (int i = 0; i < n_nodes; i++) {
    for (int j = 0; j < n_nodes; j++) {
      sol_file << solution[i][j] << endl;
    }
  }
  sol_file.close();


  for (int i = 0; i < n_nodes; i++)
    delete [] solution[i];
  delete [] solution;


  return center_sol;

}

void deallocate_arrays(double *A, int *rowA, int *colA, double *x, double *b) {


  delete [] A;
  delete [] rowA;
  delete [] colA;
  delete [] x;
  delete [] b;

}
