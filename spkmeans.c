
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define epsilon 1.0e-15

static double ** scanInput();
static double ** buildWam(double ** vectors);
static double euclideanNormCalc (double * first, double * second);
static double weightCalc (double res);
static void printMat(double ** mat);
static int isInteger(char* number);
static void freeAll(double ** array, int k);
static double * buildDdg(double ** vectors);
static void printDiagonal(double * diag);
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



int n, D;

int isInteger(char* number)
{
    int len, i;
    len = strlen(number);
    for (i = 0; i < len; i++){
        if (!isdigit(number[i]))
            return 0;
    }

    return 1;
}

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

void printDiagonal(double * diag){
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

int main(int argc, char* argv[])
{
    int k;
    double ** vectors, ** wam, ** Lnorm, ** V;
    double * ddg, * sqrtDdg;

    if(argc != 2){
        printf("Incorrect number of arguments!");
        exit(-1);
    }
    if (isInteger(argv[1])){
        k = atoi(argv[1]);
    }
    else{
        printf("k must be a positive integer");
        exit(-1);
    }


    vectors = scanInput();

    if (k >= n){
        printf("k must be < n");
        exit(-1);
    }

    wam = buildWam(vectors);
    printf("%s", "wam");
    printf("\n");

    printMat(wam);
    ddg = buildDdg(wam);

    printf("%s", "ddg");
    printf("\n");
    printDiagonal(ddg);

    sqrtDdg = buildDdgSqrt(ddg);
    printf("%s", "sqrtddg");
    printf("\n");
    printDiagonal(sqrtDdg);

    Lnorm = buildLnorm(wam, sqrtDdg);
    printf("%s", "Lnorm");
    printf("\n");
    printMat(Lnorm);

    V = jacobi(Lnorm);
    printf("%s", "V");
    printf("\n");
    printMat(V);

    freeAll(vectors, n);
    freeAll(wam, n);
    freeAll(Lnorm, n);
    freeAll(V, n);
    free(ddg);
    free(sqrtDdg);


    return 0;
}

/** parsing input into an array of vectors **/
double ** scanInput() {
    double val;
    char c;
    int index, i, d, maxD, maxN;
    double *currVector;
    double **vectors;
    maxN = 50;
    maxD = 10;

    currVector = (double *) calloc(maxD , sizeof(double));
    assert(currVector != NULL && "Error in allocating memory!");
    vectors = (double **) calloc(maxN, sizeof(double *));
    assert(vectors != NULL && "Error in allocating memory!");

    index = 0;
    i = 0;
    while (scanf("%lf%c", &val, &c) == 2) {
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

    while (ind < 100 && isConvergence){
        maxInd = findMaxAbsValInd(A);
        maxI = maxInd[0];
        maxJ = maxInd[1];
        theta = calcTheta(A[maxJ][maxJ], A[maxI][maxI], A[maxI][maxJ]);
        t = calcT(theta);
        c = calcC(t);
        s = t * c;
        buildRotationMat(P, maxI, maxJ, c, s);
        buildCurA(curA, A, maxI, maxJ, c, s);
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

    printf("%s", "A");
    printf("\n");
    printMat(A);
    printf("%s", "num of iter");
    printf("%d", ind);
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

    return maxInd;
}

double calcTheta(double Ajj, double Aii, double Aij){
    double theta;
    theta = (Ajj - Aii) / (2 * Aij);

    return theta;
}

double calcT(double theta){
    double t;
    t = 1 / (fabs(theta) + sqrt((pow(theta, 2) + 1)));
    if (theta < 0){
       t = -t;
    }

    return t;
}

double calcC(double t){
    double c;
    c = 1 / sqrt((pow(t, 2) + 1));
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

    /**

    printf("%s", "curA before changes");
    printf("\n");
    printMat(curA);

     **/

    copyMat(A, curA);

    for (l = 0; l < n; l++){
        curA[l][i] = (c * A[l][i]) - (s * A[l][j]);
        curA[l][j] = (c * A[l][j]) - (s * A[l][i]);
        curA[i][l] = curA[l][i];
        curA[j][l] = curA[l][j];
    }

    curA[i][i] = (pow(c, 2) * A[i][i]) + (pow(s, 2) * A[j][j]) - (2 * s * c * A[i][j]);
    curA[j][j] = (pow(s, 2) * A[i][i]) + (pow(c, 2) * A[j][j]) + (2 * s * c * A[i][j]);
    curA[i][j] = 0;
    curA[j][i] = 0;


    /** for (l = 0; l < n; l++){
        for (k = 0; k < n; k++){
            if (l == i){
                if (k == i){
                    curA[l][k] = (pow(c, 2) * A[i][i]) + (pow(s, 2) * A[j][j]) - (2 * s * c * A[i][j]);
                }
                else if (k == j){
                    curA[l][k] = 0;
                }
                else{
                    curA[l][k] = A[l][k];
                }
            }
            else if (l == j){
                if (k == j){
                    curA[l][k] = (pow(s, 2) * A[i][i]) + (pow(c, 2) * A[j][j]) + (2 * s * c * A[i][j]);
                }
                else if (k == i){
                    curA[l][k] = 0;
                }
                else {
                    curA[l][k] = A[l][k];
                }
            }
            else if (k == i){
                curA[l][k] = (c * A[l][k]) - (s * A[l][j]);
            }
            else if (k == j){
                curA[l][k] = (c * A[l][k]) + (s * A[l][i]);
            }
            else{
                curA[l][k] = A[l][k];
            }
        }
    }
     **/

    /**
    printf("%s", "curA");
    printf("\n");
    printMat(curA);
     **/
}

void calcV(double ** V, int i, int j, double c, double s){
    int l;
    double tmp;
    for (l = 0; l < n; l++){
        tmp = V[l][i];
        V[l][i] = (c * V[l][i]) - (s * V[l][j]);
        V[l][j] = (c * V[l][j]) + (s * tmp);
    }
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
    double resA, resCurA, diff;

    resA = frobeniusNormCalc(A);
    resCurA = frobeniusNormCalc(curA);

    for (i = 0; i < n; i++) {
        resA -= pow(A[i][i], 2);
        resCurA -= pow(curA[i][i], 2);
    }

    diff = resA - resCurA;
    if (diff <= epsilon){
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
    return pow(res, 2);
}


