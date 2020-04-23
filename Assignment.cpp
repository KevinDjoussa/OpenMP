#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <iostream>
using namespace std;

void naive_mul(float **MA, float **MB, float **MC, int arrsize);// parallel naive matrix

void naive_mul_non(float **MA, float **MB, float **MC, int arrsize);// parallel naive matrix

void matrix_tr(float **MA, float **BM, float **MC, int arrsize);// parallel transpose optimisation


int main ()
{
    int arrsize = 200; // we set the size to 128 can be change to test out different Matrices sizes
    // dealing only with square matrices

    float **MA= new float*[arrsize];
    float **MB = new float*[arrsize];
    float **BM = new float*[arrsize];
    float **MC = new float*[arrsize];

    for (int i = 0; i < arrsize; i++)
    {
        MA[i] = new float[arrsize];
        MB[i] = new float[arrsize];
        BM[i] = new float[arrsize];
        MC[i] = new float[arrsize];
        for (int j = 0; j < arrsize; j++)
        {
            MA[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);// piece of code gotten from https://stackoverflow.com/questions/686353/random-float-number-generation
            MB[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);// piece of code gotten from https://stackoverflow.com/questions/686353/random-float-number-generation
            MC[i][j] = 0.0;
        }
    }
    for (int i = 0; i < arrsize; i++)
    {
        for (int j = 0; j < arrsize; j++)
        {
            BM[i][j] = MB[j][i];
        }
    }

    for (int it = 0; it < 1; it++)
    {

        // parallel Naive here will be compared with transpose
        cout << "Parallel Naive has started" << endl;
        clock_t start = clock();

        naive_mul(MA, MB, MC, arrsize);// passes in Matrix A and Matrix B and C to store result

        clock_t end = clock();

        clock_t ft = end-start; // final timing

        cout << "Run Time: " << (double) (ft) << endl;

        // we do parallel transposition hereeeeeee
        cout << "Parallel transposition started" << endl;
        start = clock();

        matrix_tr(MA, BM, MC, arrsize);// passes in matrix A and the transpose of matrix B and C to store result

        end = clock();

        ft = end-start;
        cout << "Run Time: " << (double) (ft) << endl;

        // we do parallel transposition hereeeeeee
        cout << "Naive non Parallel started" << endl;
        start = clock();

        naive_mul_non(MA, MB, MC, arrsize);// passes in matrix A and the transpose of matrix B and C to store result

        end = clock();

        ft = end-start;
        cout << "Run Time: " << (double) (ft) << endl;


    }
    return 0;
}

void naive_mul_non(float **MA, float **MB, float **MC, int arrsize)
{
    int i, j, k;

    // naive is run in parallel allowing for optimal execution and computation

    for (i = 0; i < arrsize; i++)// i made private to avoid racing
    {
        for (j = 0; j < arrsize; j++)//j made private to avoid racing
        {
            for (k = 0; k < arrsize; k++)// k made private to avoid racing
            {
                MC[i][j] += MA[i][k] * MB[k][j];// multiplication
            }
        }
    }

}

void matrix_tr(float **MA, float **BM, float **MC, int arrsize)
{
    int i, j, k;

    //this algorithm uses the transpose matrix BM instead of MB which optimises Data Locality and performance
#pragma omp parallel shared(MA, BM, arrsize) private(i, j, k)
    {
#pragma omp parallel for
        for (i = 0; i < arrsize; i++)
        {
            for (j = 0; j < arrsize; j++)
            {
                for (k = 0; k < arrsize; k++)
                {
                    MC[i][j] += MA[i][k] * BM[j][k]; // transpose multiplication
                }
            }
        }
    }
}

void naive_mul(float **MA, float **MB, float **MC, int arrsize)
{
    int i, j, k;

    // naive is run in parallel allowing for optimal execution and computation
#pragma omp parallel shared(MA, MB, arrsize) private(i, j, k)

    {
#pragma omp parallel for
        for (i = 0; i < arrsize; i++)// i made private to avoid racing
        {
            for (j = 0; j < arrsize; j++)//j made private to avoid racing
            {
                for (k = 0; k < arrsize; k++)// k made private to avoid racing
                {
                    MC[i][j] += MA[i][k] * MB[k][j];// multiplication
                }
            }
        }
    }
}