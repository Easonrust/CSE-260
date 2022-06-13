/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#ifdef _MPI_
#include <mpi.h>
#endif
using namespace std;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

extern control_block cb;

#define LEFT 0
#define RIGHT 1
#define UP 2
#define DOWN 3

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve_baseline(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){
  
      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /* 
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */
    
    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;

    // Fills in the TOP Ghost Cells
    for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
    }

    // Fills in the RIGHT Ghost Cells
    for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
    }

    // Fills in the LEFT Ghost Cells
    for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
    }	

    // Fills in the BOTTOM Ghost Cells
    for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
    }

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  stats(E_prev,m,n,&Linf,&sumSq);
  L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

#ifdef _MPI_
double *alloc1D(int m, int n);

static inline int max(int x, int y) {
    return x <= y ? y : x;
}

static inline int min(int x, int y) {
    return x <= y ? x : y;
}

void distribute(const double *__restrict__ A, double *__restrict__ B, const int M, const int N, const int Mcount, const int Ncount, const int stride) {
    // memset B
    int X = cb.py * M, Y = cb.px * N;
    for (int i = 0; i < X; i++)
        for (int j = 0; j < Y; j++)
            B[i * Y + j] = 0;
    int idxB = 0, actualM, actualN, x1, x2, y1, y2, idxA, offsetB, offsetA;
    for (int i = 0; i < cb.py; i++) {
        for (int j = 0; j < cb.px; j++) {
            actualM = i < Mcount ? M : M - 1;
            actualN = j < Ncount ? N : N - 1;
            x1 = min(i, Mcount);
            y1 = min(j, Ncount);
            x2 = max(0, i - Mcount);
            y2 = max(0, j - Ncount);
            idxA = (x1 * M + x2 * (M - 1) + 1) * stride + (y1 * N + y2 * (N - 1) + 1);
            offsetB = 0;
            offsetA = 0;
            for (int ii = 0; ii < actualM; ii++) {
                for (int jj = 0; jj < actualN; jj++) {
                    B[idxB + offsetB + jj] = A[idxA + offsetA + jj];
                }
                offsetA += stride;
                offsetB += N;
            }
            idxB += M * N;
        }
    }
}

void set_matrix(const double *__restrict__ A, double *__restrict__ B, const int M, const int N) {
    // memset B
    int actualM = M + 2, actualN = N + 2, offsetB = 0, offsetA = 0;
    for (int i = 0; i < actualM; i++) {
        for (int j = 0; j < actualN; j++) {
            B[offsetB + j] = 0;
        }
        offsetB += actualN;
    }
    offsetB = actualN;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[offsetB + j + 1] = A[offsetA + j];
        }
        offsetA += N;
        offsetB += actualN;
    }
        
}

static inline void stat_helper(const double *E, int myrank, int m, int n, int stride, double *_mx, double *_l2) {
    double mx = -1, _sumSq = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double x = E[(i + 1) * stride + (1 + j)];
            _sumSq += x * x;
            double fe = fabs(x);
            if (fe > mx) {
                mx = fe;
            }
        }
    }
    MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : &mx, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(myrank == 0 ? MPI_IN_PLACE : &_sumSq, &_sumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    *_mx = mx;
    *_l2 = L2Norm(_sumSq);
}

void solve_mpi(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
    double *_E_prev_input = *_E_prev, *_E_input = *_E;

    //PDE loop fused or not
    #define FUSED 1

    // get my rank
    int myrank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // get 2D index of current subblock
    int x = myrank/cb.px;
    int y = myrank%cb.px;

    // assume cb.m is divisible by cb.py and cb.n is divisible by cb.px
    // get the dimension of current subblock
    int m = cb.m/cb.py;
    int n = cb.n/cb.px;
    int Mcount = cb.m % cb.py != 0 ? (cb.m % cb.py) : cb.py;
    int Ncount = cb.n % cb.px != 0 ? (cb.n % cb.px) : cb.px;
    // get the dimension of larger subblock
    int M = Mcount == cb.py ? m : m + 1;
    int N = Ncount == cb.px ? n : n + 1;
    m += (M != m && x < Mcount);
    n += (N != n && y < Ncount);

    // data owned by the current block 
    double *E = alloc1D(M + 2, N + 2);
    double *E_prev = alloc1D(M + 2, N + 2);
    double *r = alloc1D(M + 2, N + 2);

    // get e_prev and r
    double *E_scatter = alloc1D(cb.py * M, cb.px * N);
    double *R_scatter = alloc1D(cb.py * M, cb.px * N);
    double *E_recv = alloc1D(M, N);
    double *R_recv = alloc1D(M, N);

    // scatter data
    if (myrank == 0) {
        distribute(_E_prev_input, E_scatter, M, N, Mcount, Ncount, cb.n + 2);
        distribute(R, R_scatter, M, N, Mcount, Ncount, cb.n + 2);
    }

    if (!cb.noComm) {
        MPI_Scatter(E_scatter, M * N, MPI_DOUBLE, E_recv, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(R_scatter, M * N, MPI_DOUBLE, R_recv, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    set_matrix(E_recv, E_prev, M, N);
    set_matrix(R_recv, r, M, N);

    int stride = N + 2;
    MPI_Datatype col;
    MPI_Type_vector(m, 1, stride, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);


    //  double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = r;
    double *E_tmp = E;
    double *E_prev_tmp = E_prev;
    double mx, sumSq;
    int niter;
    //  int m = cb.m, n=cb.n;
    int innerBlockRowStartIndex = stride+1;
    int innerBlockRowEndIndex = m*stride+1;
    int i, j;

    // We continue to sweep over the mesh until the simulation has reached
    // the desired number of iterations
    for (niter = 0; niter < cb.niters; niter++){
        if(cb.debug && (niter==0)){
	        stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
	        repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	        if (cb.plot_freq){
	            plotter->updatePlot(E,  -1, m+1, n+1);
            }
        }
        if (!cb.noComm){
            MPI_Request requests[8] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL,MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
            //MPI_Request requests[8];
            // communication on the top
            if (x==0){
                for (i = 1; i <= n; i++) {
                    E_prev[i] = E_prev[i + stride*2];
                }
            }else{
                int top_rank = (x-1)*cb.px+y;
                MPI_Isend(E_prev+1+stride, n, MPI_DOUBLE, top_rank, UP, MPI_COMM_WORLD, &requests[0]);
                MPI_Irecv(E_prev+1, n, MPI_DOUBLE, top_rank, DOWN, MPI_COMM_WORLD, &requests[1]);
            }

        // communication on the right
        if (y==cb.px-1){
            for (i = (n+1)+stride; i <= n+1+m*stride; i+=stride){
                E_prev[i] = E_prev[i-2];
            }
        }else{
            int right_rank = x*cb.px+y+1;
            MPI_Isend(E_prev+stride+n, 1, col, right_rank, RIGHT, MPI_COMM_WORLD, &requests[2]);
            MPI_Irecv(E_prev+stride+n+1, 1, col, right_rank, LEFT, MPI_COMM_WORLD, &requests[3]);
        }

        // communication on the left
        if (y==0){
            for (i = stride; i <= m*stride; i+=stride) {
                E_prev[i] = E_prev[i+2];
            }	
        }else{
            int left_rank = x*cb.px+y-1;
            MPI_Isend(E_prev+stride+1, 1, col, left_rank, LEFT, MPI_COMM_WORLD, &requests[4]);
            MPI_Irecv(E_prev+stride, 1, col, left_rank, RIGHT, MPI_COMM_WORLD, &requests[5]);
        }

        // communication on the bottom
        if (x==cb.py-1){
            for (i = ((m+2)*stride-stride+1); i <= (m+1)*stride+n; i++) {
                E_prev[i] = E_prev[i - stride*2];
            }
        }else{
            int bottom_rank = (x+1)*cb.px+y;
            MPI_Isend(E_prev+m*stride+1, n, MPI_DOUBLE, bottom_rank, DOWN, MPI_COMM_WORLD, &requests[6]);
            MPI_Irecv(E_prev+(m+1)*stride+1, n, MPI_DOUBLE, bottom_rank, UP, MPI_COMM_WORLD, &requests[7]);
        }
        
        // Compute inner block while we waiting
        int innerStart = 2*stride+2;
        int innerEnd = (m-1)*stride+2;

#ifdef FUSED
        // Solve for the excitation, a PDE
        for(j = innerStart; j <= innerEnd; j+=stride) {
            E_tmp = E + j;
	        E_prev_tmp = E_prev + j;
            R_tmp = r + j;
	        for(i = 0; i < n-2; i++) {
	            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+stride]+E_prev_tmp[i-stride]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
#else
        // Solve for the excitation, a PDE
        for(j = innerStart; j <= innerEnd; j+=stride) {
            E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n-2; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+stride]+E_prev_tmp[i-stride]);
            }
        }

        /* 
         * Solve the ODE, advancing excitation and recovery variables
         *     to the next timtestep
         */

        for(j = innerStart; j <= innerEnd; j+=stride) {
            E_tmp = E + j;
            R_tmp = r + j;
	        E_prev_tmp = E_prev + j;
            for(i = 0; i < n-2; i++) {
	            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
            }
        }
#endif


        MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
    }
      




    //////////////////////////////////////////////////////////////////////////////
    int i1,j1,i2,j2;

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(int j = innerBlockRowStartIndex+stride; j <= innerBlockRowEndIndex-stride; j+=stride) {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
        R_tmp = r + j;
	    
        i1 = 0;
        i2 = n-1;
	    E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+stride]+E_prev_tmp[i1-stride]);
        E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
        R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
        E_tmp[i2] = E_prev_tmp[i2]+alpha*(E_prev_tmp[i2+1]+E_prev_tmp[i2-1]-4*E_prev_tmp[i2]+E_prev_tmp[i2+stride]+E_prev_tmp[i2-stride]);
        E_tmp[i2] += -dt*(kk*E_prev_tmp[i2]*(E_prev_tmp[i2]-a)*(E_prev_tmp[i2]-1)+E_prev_tmp[i2]*R_tmp[i2]);
        R_tmp[i2] += dt*(epsilon+M1* R_tmp[i2]/( E_prev_tmp[i2]+M2))*(-R_tmp[i2]-kk*E_prev_tmp[i2]*(E_prev_tmp[i2]-b-1));
    
    }
    // top & bot boundary with corner
    j1 = innerBlockRowStartIndex;
    j2 = innerBlockRowEndIndex;

    E_tmp = E + j1;
	E_prev_tmp = E_prev + j1;
    R_tmp = r + j1;
	for(i1 = 0; i1 < n; i1++) {
	    E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+stride]+E_prev_tmp[i1-stride]);
        E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
        R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
    }
    E_tmp = E + j2;
	E_prev_tmp = E_prev + j2;
    R_tmp = r + j2;
	for(i1 = 0; i1 < n; i1++) {
	    E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+stride]+E_prev_tmp[i1-stride]);
        E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
        R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
    }
    
#else
    // Solve for the excitation, a PDE
    for(int j = innerBlockRowStartIndex+stride; j <= innerBlockRowEndIndex-stride; j+=stride) {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
	    
        i1 = 0;
        i2 = n-1;
	    E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+stride]+E_prev_tmp[i1-stride]);
        
        E_tmp[i2] = E_prev_tmp[i2]+alpha*(E_prev_tmp[i2+1]+E_prev_tmp[i2-1]-4*E_prev_tmp[i2]+E_prev_tmp[i2+stride]+E_prev_tmp[i2-stride]);

    }
    // top & bot boundary with corner
    j1 = innerBlockRowStartIndex;
    j2 = innerBlockRowEndIndex;

    E_tmp = E + j1;
	E_prev_tmp = E_prev + j1;
	for(i1 = 0; i1 < n; i1++) {
	    E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+stride]+E_prev_tmp[i1-stride]);    
    }
    E_tmp = E + j2;
	E_prev_tmp = E_prev + j2;
	for(i1 = 0; i1 < n; i1++) {
	    E_tmp[i1] = E_prev_tmp[i1]+alpha*(E_prev_tmp[i1+1]+E_prev_tmp[i1-1]-4*E_prev_tmp[i1]+E_prev_tmp[i1+stride]+E_prev_tmp[i1-stride]);
    }


    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */
    for(int j = innerBlockRowStartIndex+stride; j <= innerBlockRowEndIndex-stride; j+=stride) {
        E_tmp = E + j;
	    E_prev_tmp = E_prev + j;
        R_tmp = r + j;
	    
        i1 = 0;
        i2 = n-1;
	    
        E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
        R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
        
        E_tmp[i2] += -dt*(kk*E_prev_tmp[i2]*(E_prev_tmp[i2]-a)*(E_prev_tmp[i2]-1)+E_prev_tmp[i2]*R_tmp[i2]);
        R_tmp[i2] += dt*(epsilon+M1* R_tmp[i2]/( E_prev_tmp[i2]+M2))*(-R_tmp[i2]-kk*E_prev_tmp[i2]*(E_prev_tmp[i2]-b-1));
    
    }

    // top & bot boundary with corner
    E_tmp = E + j1;
	E_prev_tmp = E_prev + j1;
    R_tmp = r + j1;
	for(i1 = 0; i1 < n; i1++) {
        E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
        R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
    }
    E_tmp = E + j2;
	E_prev_tmp = E_prev + j2;
    R_tmp = r + j2;
	for(i1 = 0; i1 < n; i1++) {
        E_tmp[i1] += -dt*(kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-a)*(E_prev_tmp[i1]-1)+E_prev_tmp[i1]*R_tmp[i1]);
        R_tmp[i1] += dt*(epsilon+M1* R_tmp[i1]/( E_prev_tmp[i1]+M2))*(-R_tmp[i1]-kk*E_prev_tmp[i1]*(E_prev_tmp[i1]-b-1));
    }

#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
        if (!(niter % cb.plot_freq)){
	        int offsetA = stride, offsetB = 0;
            for (i = 0; i < M; i++) {
                for (j = 0; j < N; j++) {
                    E_recv[offsetB + j] = E[offsetA + j + 1]; 
                }
                offsetB += N;
                offsetA += stride;
            }
            // gather data into process 0(inverse process)
            MPI_Gather(E_recv, M * N, MPI_DOUBLE, E_scatter, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if (myrank == 0) {
                int idxB = 0, actualM, actualN, x1, x2, y1, y2, idxA;
                for (i = 0; i < cb.py; i++) {
                    for (j = 0; j < cb.px; j++) {
                        actualM = i < Mcount ? M : M - 1;
                        actualN = j < Ncount ? N : N - 1;
                        x1 = min(i, Mcount);
                        y1 = min(j, Ncount);
                        x2 = max(0, i - Mcount);
                        y2 = max(0, j - Ncount);
                        idxA = (x1 * M + x2 * (M - 1) + 1) * (cb.n + 2) + (y1 * N + y2 * (N - 1) + 1);
                        offsetB = 0;
                        offsetA = 0;
                        for (int ii = 0; ii < actualM; ii++) {
                            for (int jj = 0; jj < actualN; jj++) {
                                _E_input[idxA + offsetA + jj] = E_scatter[idxB + offsetB + jj];
                            }
                            offsetA += (cb.n + 2);
                            offsetB += N;
                        }
                        idxB += M * N;
                    }
                }
                plotter->updatePlot(_E_input, niter, cb.m, cb.n);
            }
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning
  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  MPI_Type_free(&col);
  stat_helper(E_prev, myrank, m, n, stride, &Linf, &L2);
//   stats(E_prev,m,n,&Linf,&sumSq);
//   L2 = L2Norm(sumSq);

  // Swap pointers so we can re-use the arrays
//   *_E = E;
//   *_E_prev = E_prev;
}
#endif

void printMat2(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
    #ifdef _MPI_
        solve_mpi(_E, _E_prev, R, alpha, dt,  plotter,  L2, Linf);
    #else
        solve_baseline(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
    #endif
}