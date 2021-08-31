
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define epsilon 1.0e-15
#define errorMsg "An Error Has Occured"
#define inputMsg "Invalid Input!"

static double ** scanInput(char * fileName);
static double ** buildWam(double ** vectors);
static double euclideanNormCalc (double * first, double * second);
static double weightCalc (double res);
static void printMat(double ** mat);
static void freeAll(double ** array, int k);
static double * buildDdg(double ** vectors);
/** static void printDiagonal(double * diag);**/
static double * buildDdgSqrt(double * ddg);
static void multiplyDiag(double ** res, double ** mat, double * diag, int flag);
static void IReduce(double ** mat);
static double ** buildLnorm(double ** wam, double * sqrtDdg);
static int * findMaxAbsValInd(double ** A);
static double calcTheta(double Ajj, double Aii, double Aij);
static double calcT(double theta);
static double calcC(double t);
static double ** initMat();
static void buildRotationMat(double ** mat, int i, int j, double c, double s);
static void buildCurA(double ** curA, double ** A, int i, int j, double c, double s);
static void copyMat(double ** orig, double ** dst);
static void calcV(double ** V, int i, int j, double c, double s);
static double ** initIdentityMat();
static void returnToIdentity(double ** P, int i, int j);
static double ** jacobi(double ** Lnorm);
static double frobeniusNormCalc(double ** mat);
static int convergence(double ** A, double ** curA);
static double * extractEigenValues(double ** mat);
static void swap(double *xp, double *yp);
static void bubbleSort(double * vector);
static double calcNorm(double * vector);
/** static double ** renormalize(double ** U, int k); **/


int n, D;



void freeAll(double ** array, int k){
    int i;
    for (i = 0; i < k; i++){
        free(array[i]);
    }
    free(array);
}

void printMat(double ** mat){
    int i, j;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            printf("%.4f", mat[i][j]);
            if(j < (n-1)){
                printf(",");
            }
        }
        printf("\n");
    }
}

/** void printDiagonal(double * diag){
    int i, j;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (i == j){
                printf("%.4f", diag[i]);
            }
            else {
                printf("%.4f", 0.0000);
            }
            if(j < (n-1)){
                printf(",");
            }
        }
        printf("\n");
    }
}
 **/

int main(int argc, char* argv[])
{
    int k;
    double ** vectors, ** wam, ** Lnorm, ** V;
    double * ddg, * sqrtDdg;
    char * fileName;

    assert(argc == 4 && inputMsg);
    k = atoi(argv[1]);
    /** goal = argv[2]; **/
    fileName = argv[3];

    vectors = scanInput(fileName);

    if (k >= n){
        printf(inputMsg);
        exit(-1);
    }

    wam = buildWam(vectors);
    ddg = buildDdg(wam);
    sqrtDdg = buildDdgSqrt(ddg);
    Lnorm = buildLnorm(wam, sqrtDdg);
    V = jacobi(Lnorm);

    /** printf("%s", "wam");
    printf("\n");
    printMat(wam);
    printf("%s", "ddg");
    printf("\n");
    printDiagonal(ddg);
    printf("%s", "sqrtddg");
    printf("\n");
    printDiagonal(sqrtDdg);
    printf("%s", "Lnorm");
    printf("\n");
    printMat(Lnorm);
    printf("%s", "V");
    printf("\n");
    printMat(V); **/

    freeAll(vectors, n);
    freeAll(wam, n);
    freeAll(Lnorm, n);
    freeAll(V, n);
    free(ddg);
    free(sqrtDdg);


    return 0;
}

/** parsing input into an array of vectors **/
double ** scanInput(char * fileName) {
    double val;
    char c;
    int index, i, d, maxD, maxN;
    double *currVector;
    double **vectors;
    FILE * inputFile;
    maxN = 50;
    maxD = 10;

    currVector = (double *) calloc(maxD , sizeof(double));
    assert(currVector != NULL && "Error in allocating memory!");
    vectors = (double **) calloc(maxN, sizeof(double *));
    assert(vectors != NULL && "Error in allocating memory!");

    inputFile = fopen(fileName, "r");

    index = 0;
    i = 0;
    while (fscanf(inputFile, "%lf%c", &val, &c) == 2) {
        currVector[i] = val;
        if (c != '\n') {
            i += 1;
        } else {
            d = i;
            vectors[index] = currVector;
            i = 0;
            currVector = (double *) calloc(d + 1, sizeof(double));
            assert(currVector != NULL && "Error in allocating memory!");
            index += 1;
        }
    }
    currVector[i] = val;
    vectors[index] = currVector;
    D = d + 1;
    n = index + 1;

    fclose(inputFile);

    return vectors;

}

double euclideanNormCalc (double * first, double * second){
    int i;
    double sum, res;
    sum = 0;

    for (i = 0; i < D; i++){
        sum += pow(first[i] - second[i], 2);
    }

    res = sqrt(sum);
    return res;
}

double weightCalc (double res){
    double weight;
    weight = -(res / 2);
    weight = exp(weight);
    return weight;
}

double ** buildWam(double ** vectors) {
    double ** wam;
    double * first, * second;
    int i, j;
    double res, weight;

    wam = (double **) calloc(n, sizeof(double *));
    assert(wam != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        wam[i] = (double *) calloc(n , sizeof(double));
        assert(wam[i] != NULL && "Error in allocating memory!");
    }

    for (i = 0; i < n - 1; i++){
        for (j = i + 1; j < n; j++){
            first = vectors[i];
            second = vectors[j];
            res = euclideanNormCalc(first, second);
            weight = weightCalc(res);
            wam[i][j] = weight;
            wam[j][i] = weight;
        }
    }

    return wam;
}

double sumRow(double * row){
    int i;
    double sum;
    sum = 0;
    for (i = 0; i < n; i++){
        sum += row[i];
    }
    return sum;
}

double * buildDdg(double ** wam) {
    double * ddg;
    int i;

    ddg = (double *) calloc(n, sizeof(double));
    assert(ddg != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        ddg[i] = sumRow(wam[i]);
    }
    return ddg;
}

double * buildDdgSqrt(double * ddg){
    double * sqrtDdg;
    int i;

    sqrtDdg = (double *) calloc(n, sizeof(double));
    assert(sqrtDdg != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        sqrtDdg[i] = 1 / sqrt(ddg[i]);
    }
    return sqrtDdg;
}

double ** buildLnorm(double ** wam, double * sqrtDdg){
    double ** Lnorm;
    int i;
    Lnorm = (double **) calloc(n, sizeof(double *));
    assert(Lnorm != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        Lnorm[i] = (double *) calloc(n , sizeof(double));
        assert(Lnorm[i] != NULL && "Error in allocating memory!");
    }

    multiplyDiag(Lnorm, wam, sqrtDdg, 1);
    multiplyDiag(Lnorm, Lnorm, sqrtDdg, 0);
    IReduce(Lnorm);

    return Lnorm;

}

void multiplyDiag(double ** res, double ** mat, double * diag, int flag){
    int i, j;

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (flag == 1) { /** 1 for row **/
                res[i][j] = mat[i][j] * diag[i];
            }
            else {
                res[j][i] = mat[j][i] * diag[i];
            }
        }
    }
}

void IReduce(double ** mat){
    int i, j;
    double cur;

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (j != i){
                cur = -mat[i][j];
                if (cur < 0 && cur > -0.00005){
                    mat[i][j] = 0.0000;
                }
                else {
                    mat[i][j] = cur;
                }
            }
            else{
                mat[i][j] = 1;
            }
        }
    }
}

double ** jacobi(double ** Lnorm){

    double ** A, ** curA, ** P, ** V;
    double theta, t, c, s;
    int maxI, maxJ, * maxInd, ind, isConvergence;

    A = initMat();
    curA = initMat();
    V = initMat();
    P = initIdentityMat();

    copyMat(Lnorm, A);
    ind = 0;
    isConvergence = 1;
    printf("%d", isConvergence);

    while (ind < 100 && isConvergence){
        printf("%s", "iter =");
        printf("%d", ind);
        printf("\n");

        maxInd = findMaxAbsValInd(A);
        maxI = maxInd[0];
        maxJ = maxInd[1];

        printf("%s", "i =");
        printf("%d", maxI);
        printf("\n");

        printf("%s", "j =");
        printf("%d", maxJ);
        printf("\n");

        theta = calcTheta(A[maxJ][maxJ], A[maxI][maxI], A[maxI][maxJ]);
        printf("%s", "theta =");
        printf("%f", theta);
        printf("\n");

        t = calcT(theta);
        printf("%s", "t =");
        printf("%f", t);
        printf("\n");

        c = calcC(t);
        printf("%s", "c =");
        printf("%f", c);
        printf("\n");

        s = t * c;
        printf("%s", "s =");
        printf("%f", s);
        printf("\n");

        buildRotationMat(P, maxI, maxJ, c, s);
        printf("%s", "P =");
        printMat(P);
        printf("\n");

        buildCurA(curA, A, maxI, maxJ, c, s);
        printf("%s", "curA =");
        printMat(curA);
        printf("\n");

        isConvergence = convergence(A, curA);
        copyMat(curA, A);
        if (ind == 0){
            copyMat(P, V);
        }
        else{
            calcV(V, maxI, maxJ, c, s);
        }
        returnToIdentity(P, maxI, maxJ);

        ind += 1;
    }

    printf("%s", "V =");
    printMat(V);
    printf("\n");

    return V;
}

int * findMaxAbsValInd(double ** A){
    int i, j, * maxInd;
    double maxAbs;
    maxAbs = -1;
    maxInd = (int *) calloc(2 , sizeof(int));

    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if ( i != j){
                if (fabs(A[i][j]) > maxAbs){
                    maxInd[0] = i;
                    maxInd[1] = j;
                    maxAbs=fabs(A[i][j]);
                }
            }
        }
    }
    printf("%s", "max =");
    printf("%f", maxAbs);
    printf("\n");

    return maxInd;
}

double calcTheta(double Ajj, double Aii, double Aij){
    double theta;
    theta = (Ajj - Aii) / (2 * Aij);

    return theta;
}

double calcT(double theta){
    double t;
    double tmp;
    tmp = pow(theta, 2);
    tmp += 1;
    t = 1 / (fabs(theta) + sqrt(tmp));
    if (theta < 0){
       t = -t;
    }
    return t;
}

double calcC(double t){
    double c, tmp;
    tmp = pow(t, 2);
    tmp += 1;
    c = 1 / sqrt(tmp);
    return c;
}

void buildRotationMat(double ** mat, int i, int j, double c, double s){
    mat[i][i] = c;
    mat[j][j] = c;
    mat[i][j] = s;
    mat[j][i] = -s;
}

double ** initMat(){
    double ** mat;
    int i;
    mat = (double **) calloc(n, sizeof(double *));
    assert(mat != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        mat[i] = (double *) calloc(n , sizeof(double));
        assert(mat[i] != NULL && "Error in allocating memory!");
    }

    return mat;
}

double ** initIdentityMat(){
    double ** mat;
    int i;
    mat = initMat();
    for (i = 0; i < n; i++){
        mat[i][i] = 1;
    }
    return mat;
}

void returnToIdentity(double ** P, int i, int j){
    P[i][i] = 1;
    P[j][j] = 1;
    P[i][j] = 0;
    P[j][i] = 0;
}

void buildCurA(double ** curA, double ** A, int i, int j, double c, double s){
    int l;
    double tmp;


    printf("%s", "A");
    printMat(A);
    printf("\n");

    copyMat(A, curA);

    for (l = 0; l < n; l++){
        tmp = (c * A[l][i]) - (s * A[l][j]);
        curA[l][i] = tmp;
        curA[i][l] = tmp;
        tmp = (c * A[l][j]) + (s * A[l][i]);
        curA[l][j] = tmp;
        curA[j][l] = tmp;
    }

    curA[i][i] = (pow(c, 2) * A[i][i]) + (pow(s, 2) * A[j][j]) - (2 * s * c * A[i][j]);
    curA[j][j] = (pow(s, 2) * A[i][i]) + (pow(c, 2) * A[j][j]) + (2 * s * c * A[i][j]);
    curA[i][j] = 0;
    curA[j][i] = 0;


    /**
    printf("%s", "curA");
    printf("\n");
    printMat(curA);
     **/
}

void calcV(double ** V, int i, int j, double c, double s){
    int l;
    double ** tmpV;
    tmpV = initMat();
    copyMat(V, tmpV);
    for (l = 0; l < n; l++){
        tmpV[l][i] = (c * V[l][i]) - (s * V[l][j]);
        tmpV[l][j] = (c * V[l][j]) + (s * V[l][i]);
    }
    copyMat(tmpV, V);
    freeAll(tmpV, n);
}

void copyMat(double ** orig, double ** dst){
    int i, j;
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            dst[i][j] = orig[i][j];
        }
    }
}

int convergence(double ** A, double ** curA){
    int i;
    double resA, resCurA, diff, sumA, sumCurA;


    resA = frobeniusNormCalc(A);
    resCurA = frobeniusNormCalc(curA);
    sumA = 0;
    sumCurA = 0;

    for (i = 0; i < n; i++) {
        sumA += (double )pow(A[i][i], 2);
        sumCurA += (double )pow(curA[i][i], 2);
    }

    resA = resA - sumA;
    resCurA = resCurA - sumCurA;

    diff = resA - resCurA;
    printf("%s", "diff =");
    printf("%f", diff);
    printf("\n");

    if (resA - resCurA <= epsilon){
        return 0;
    }
    else{
        return 1;
    }
}

double frobeniusNormCalc(double ** mat){
    int i, j;
    double res;
    res = 0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            res += pow(mat[i][j], 2);
        }
    }
    return res;
}

double * extractEigenValues(double ** mat){
    double * eigenValues;
    int i;
    eigenValues = (double *) calloc(n , sizeof(double));
    assert(eigenValues != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        eigenValues[i] = mat[i][i];
    }

    return eigenValues;
}

void swap(double *xp, double *yp)
{
    double temp = *xp;
    *xp = *yp;
    *yp = temp;
}


void bubbleSort(double * vector)
{
    int i, j;
    for (i = 0; i < n-1; i++)
        for (j = 0; j < n-i-1; j++)
            if (vector[j] > vector[j+1])
                swap(&vector[j], &vector[j+1]);
}


int eigengapHeuristic(double ** A){
    double * eigenValuesVector;
    double delta, maxDelta;
    int k, i;
    eigenValuesVector = extractEigenValues(A);
    bubbleSort(eigenValuesVector);

    maxDelta = fabs(eigenValuesVector[0] - eigenValuesVector[1]);
    k = 1;

    for (i = 1; i < n/2 ; i++){
        delta = fabs(eigenValuesVector[1] - eigenValuesVector[2]);
        if (delta > maxDelta){
            k = i + 1;
            maxDelta = delta;
        }
    }

    return k;
}

double ** extractKEigenVectors(double ** V, int k){
    double ** U;
    int i, j;

    U = (double **) calloc(n, sizeof(double *));
    assert(U != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        U[i] = (double *) calloc(k , sizeof(double));
        assert(U[i] != NULL && "Error in allocating memory!");
    }

    for (i = 0; i < n; i++){
        for (j = 0; j < k; j++){
            U[i][j] = V[i][j];
        }
    }

    return U;
}

double ** renormalize(double ** U, int k){
    double * curVector;
    double curNorm;
    double ** T;
    int i, j;

    T = (double **) calloc(n, sizeof(double *));
    assert(T != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        T[i] = (double *) calloc(k , sizeof(double));
        assert(T[i] != NULL && "Error in allocating memory!");
    }

    curVector = (double *) calloc(n , sizeof(double));
    assert(curVector != NULL && "Error in allocating memory!");

    for (i = 0; i < n; i++){
        for (j = 0; j < k; j++){
            curVector[j] = U[i][j];
        }
        curNorm = calcNorm(curVector);
        for (j = 0; j < k; j++){
            T[i][j] =  U[i][j]/curNorm;
        }
    }

    return T;
}

double calcNorm(double * vector){
    double res;
    int i;

    res = 0;

    for (i = 0; i < n; i++){
        res += pow(vector[i], 2);
    }

    return sqrt(res);
}

