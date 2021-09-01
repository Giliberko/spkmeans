
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>

#define epsilon 1.0e-15
#define errorMsg "An Error Has Occured" /** TODO: replace all error msgs with this **/
#define inputMsg "Invalid Input!"

static double ** scanInput(char * fileName);
static double ** buildWam(double ** vectors);
static double euclideanNormCalc (double * first, double * second);
static double weightCalc (double res);
static void printMat(double ** mat, int r, int c);
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
static double ** initMat(int r, int c);
static void buildRotationMat(double ** mat, int i, int j, double c, double s);
static void buildCurA(double ** curA, double ** A, int i, int j, double c, double s);
static void copyMat(double ** orig, double ** dst);
static void calcV(double ** V, int i, int j, double c, double s);
static double ** initIdentityMat();
static void returnToIdentity(double ** P, int i, int j);
static double ** jacobiAlg(double ** Lnorm);
static double frobeniusNormCalc(double ** mat);
static int convergence(double ** A, double ** curA);
static double * extractEigenValues(double ** mat);
static void swap(double *xp, double *yp);
static void bubbleSort(double * vector, int * indices) ;
static double calcNorm(double * vector, int k);
static double ** renormalize(double ** U, int k);
static double ** combineAllEigen(double ** mat, double * vector);
static double ** separateVectors(double ** combined);
static double * separateValues(double ** combined);
static int eigenGapHeuristic(double * eigenValuesVector, int * indices);
static double ** extractKEigenVectors(double ** V, int * indices, int k);
static double * ddg(double ** inputMat);
static double ** transpose(double ** mat);
static double ** jacobi(double ** inputMat);
static void addVectorToCluster(const double* vector, double** clustersSum, int* clustersSize, int index, int k);
static void clustering(double ** vectors, double ** centroids, double ** clustersSum, int* clustersSize, int k);
static void avgCalc(double ** clustersSum, const int * clustersSize, int index, double * newCent, int k);
static void updateCentroid(int k, double ** clustersSum, int * clustersSize, double ** curCentroids);
static int diff(int k, double ** curCentroids, double ** centroids);
static void kmean(double ** vectors, int k);


int n, D;



void freeAll(double ** array, int k){
    int i;
    for (i = 0; i < k; i++){
        free(array[i]);
    }
    free(array);
}

void printMat(double ** mat, int r, int c){
    int i, j;
    for (i = 0; i < r; i++){
        for (j = 0; j < c; j++){
            printf("%.4f", mat[i][j]);
            if(j < (c-1)){
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


double * separateValues(double ** combined){
    double * vector;
    int j;
    vector = (double *) calloc(n , sizeof(double));
    assert(vector != NULL && errorMsg);
    for (j = 0; j < n; j++){
        vector[j] = combined[0][j];
    }

    printf("%s", "seperateValues func =");
    printDiagonal(vector);
    printf("\n");

    return vector;
}

double ** separateVectors(double ** combined){
    double ** vectorsMat;
    int i, j;
    vectorsMat = initMat(n, n);
    for (i = 1; i < n + 1; i++){
        for (j = 0; j < n; j++){
            vectorsMat[i - 1][j] = combined[i][j];
        }
    }

    printf("%s", "seperateVectors func =");
    printMat(vectorsMat, n, n);
    printf("\n");

    return vectorsMat;
}

double ** spk(double ** inputMat, int k){
    double ** wam, * ddg, * sqrtDdg, ** Lnorm, ** combinedEigen, ** eigenVectors, ** U, ** T;
    double * eigenVals;
    int K, i, * indices;
    K = k;

    wam = buildWam(inputMat);
    ddg = buildDdg(wam);
    sqrtDdg = buildDdgSqrt(ddg);
    Lnorm = buildLnorm(wam, sqrtDdg);
    combinedEigen = jacobiAlg(Lnorm);
    eigenVals = separateValues(combinedEigen);
    eigenVectors = separateVectors(combinedEigen);

    indices= (int *) calloc(n , sizeof(int));
    assert(indices != NULL && errorMsg);

    for (i = 0; i < n; i++){
        indices[i] = i;
    }
    if (k == 0){
        K = eigenGapHeuristic(eigenVals, indices);
    }
    else{
        eigenGapHeuristic(eigenVals, indices);
    }
    U = extractKEigenVectors(eigenVectors, indices, K);

    T = renormalize(U, K);
    printf("%s", "T =");
    printMat(T, n, K);
    printf("\n");

    printf("%s", "kmeans =");
    printf("\n");
    kmean(T, K);
    printf("\n");


    return T;

}

double * ddg(double ** inputMat){
    double ** wam, * ddg;
    wam = buildWam(inputMat);
    ddg = buildDdg(wam);
    return ddg;
}

double ** lnorm(double ** inputMat){
    double ** wam, * ddg, ** lnorm, * sqrtDdg;
    wam = buildWam(inputMat);
    ddg = buildDdg(wam);
    sqrtDdg = buildDdgSqrt(ddg);
    lnorm = buildLnorm(wam, sqrtDdg);
    return lnorm;
}

double ** jacobi(double ** inputMat){
    double ** combined, ** transposedJac;
    combined = jacobiAlg(inputMat);
    transposedJac = transpose(combined);
    return transposedJac;
}

double ** transpose(double ** mat){
    double ** transposedMat;
    int i, j;
    transposedMat = initMat(n + 1, n);
    for (i = 0; i < n + 1; i++){
        for (j = 0; j < n; j++){
            if (i == 0){
                transposedMat[i][j] = mat[i][j];
            }
            else{
                transposedMat[j + 1][i - 1] = mat[i][j];
            }
        }
    }

    return transposedMat;
}

int main(int argc, char* argv[])
{
    int k;
    double ** inputMat;
    char * fileName, * goal;
    char first;

    assert(argc == 4 && inputMsg);
    k = atoi(argv[1]);
    goal = argv[2];
    fileName = argv[3];

    inputMat = scanInput(fileName);

    if (k >= n){
        printf(inputMsg);
        exit(-1);
    }

    first = goal[0];

    if (first == 's'){
       spk(inputMat, k);
    }
    else if (first == 'w'){
        printMat(buildWam(inputMat), n, n);
    }
    else if (first == 'd'){
        printDiagonal(ddg(inputMat));
    }
    else if (first == 'l'){
        printMat(lnorm(inputMat), n, n);
    }
    else if (first == 'j'){
        printMat(jacobi(inputMat), n + 1, n);
    }
    else{
        printf(inputMsg);
        exit(-1);
    }

    freeAll(inputMat, n);

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
    assert(currVector != NULL && errorMsg);
    vectors = (double **) calloc(maxN, sizeof(double *));
    assert(vectors != NULL && errorMsg);

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
            assert(currVector != NULL && errorMsg);
            index += 1;
        }
    }
    currVector[i] = val;
    vectors[index] = currVector;
    D = d + 1;
    n = index + 1;

    printf("%s", "n =");
    printf("%d", n);
    printf("\n");
    printf("%s", "d =");
    printf("%d", D);
    printf("\n");

    fclose(inputFile);

    printMat(vectors, n, D);

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

double ** jacobiAlg(double ** Lnorm){

    double ** A, ** curA, ** P, ** V, ** eigenRes, * eigenValues;
    double theta, t, c, s;
    int maxI, maxJ, * maxInd, ind, isConvergence;

    A = initMat(n, n);
    curA = initMat(n, n);
    V = initMat(n, n);
    P = initIdentityMat();

    copyMat(Lnorm, A);
    ind = 0;
    isConvergence = 1;

    while (ind < 100 && isConvergence){

        maxInd = findMaxAbsValInd(A);
        maxI = maxInd[0];
        maxJ = maxInd[1];

        /** printf("%s", "i =");
        printf("%d", maxI);
        printf("\n"); **/

        /** printf("%s", "j =");
        printf("%d", maxJ);
        printf("\n");**/

        theta = calcTheta(A[maxJ][maxJ], A[maxI][maxI], A[maxI][maxJ]);
        /** printf("%s", "theta =");
        printf("%f", theta);
        printf("\n");**/

        t = calcT(theta);
        /** printf("%s", "t =");
        printf("%f", t);
        printf("\n");**/

        c = calcC(t);
        /** printf("%s", "c =");
        printf("%f", c);
        printf("\n");**/

        s = t * c;
        /** printf("%s", "s =");
        printf("%f", s);
        printf("\n");**/

        buildRotationMat(P, maxI, maxJ, c, s);
       /**  printf("%s", "P =");
        printMat(P);
        printf("\n");**/

        buildCurA(curA, A, maxI, maxJ, c, s);
       /**  printf("%s", "curA =");
        printMat(curA);
        printf("\n");**/

        isConvergence = convergence(A, curA);
        /** if (!isConvergence){
            continue;
        } **/

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

    printf("%s", "last iter =");
    printf("%d", ind);
    printf("\n");

    printf("%s", "V =");
    printMat(V, n, n);
    printf("\n");

    eigenValues = extractEigenValues(A);
    eigenRes = combineAllEigen(V, eigenValues);



    return eigenRes;
}

double ** combineAllEigen(double ** mat, double * vector){
    double ** combinedMat;
    int i, j;

    combinedMat = initMat(n + 1, n);

    for (i = 0; i < n + 1; i++){
        for (j = 0; j < n; j++){
            if (i == 0){
                combinedMat[i][j] = vector[j];
            }
            else{
                combinedMat[i][j] = mat[i - 1][j];
            }
        }
    }
    printf("%s", "combined from jacobi algo =");
    printMat(combinedMat, n + 1, n);
    printf("\n");
    return combinedMat;
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
    /** printf("%s", "max =");
    printf("%f", maxAbs);
    printf("\n"); **/

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

double ** initMat(int r, int c){
    double ** mat;
    int i;
    mat = (double **) calloc(r, sizeof(double *));
    assert(mat != NULL && "Error in allocating memory!");

    for (i = 0; i < r; i++){
        mat[i] = (double *) calloc(c , sizeof(double));
        assert(mat[i] != NULL && "Error in allocating memory!");
    }

    return mat;
}

double ** initIdentityMat(){
    double ** mat;
    int i;
    mat = initMat(n, n);
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


    /** printf("%s", "A");
    printMat(A);
    printf("\n"); **/

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
    tmpV = initMat(n, n);
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
    double resA, resCurA, sumA, sumCurA;


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
    assert(eigenValues != NULL && errorMsg);

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


void bubbleSort(double * vector, int * indices) {
    int i, j, temp;

    for (i = 0; i < n - 1; i++){
        for (j = 0; j < n - i - 1; j++){
            if (vector[j] > vector[j + 1]){
                swap(&vector[j], &vector[j + 1]);
                temp = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp;
            }
        }
    }
}


int eigenGapHeuristic(double * eigenValuesVector, int * indices){
    double delta, maxDelta;
    int k, i;

    bubbleSort(eigenValuesVector, indices);

    maxDelta = fabs(eigenValuesVector[0] - eigenValuesVector[1]);
    k = 1;

    for (i = 1; i < n/2 ; i++){
        delta = fabs(eigenValuesVector[i] - eigenValuesVector[i + 1]);
        if (delta > maxDelta){
            k = i + 1;
            maxDelta = delta;
        }
    }

    return k;
}

double ** extractKEigenVectors(double ** V, int * indices, int k){
    double ** U;
    int i, j;

    U = initMat(n, k);

    for (i = 0; i < n; i++){
        for (j = 0; j < k; j++){
            U[i][j] = V[i][indices[j]];
        }
    }
    printf("%s", "U =");
    printMat(U, n, k);
    printf("\n");

    return U;
}

double ** renormalize(double ** U, int k){
    double curNorm;
    double ** T;
    int i, j;

    T = initMat(n, k);


    for (i = 0; i < n; i++){
        curNorm = calcNorm(U[i], k);
        for (j = 0; j < k; j++){
            T[i][j] =  U[i][j]/curNorm;
        }
    }

    return T;
}

double calcNorm(double * vector, int k){
    double res;
    int i;

    res = 0;

    for (i = 0; i < k; i++){
        res += pow(vector[i], 2);
    }

    return sqrt(res);
}

/** adding vector to the right cluster - increase cluster size and sum accordingly **/
void addVectorToCluster(const double* vector, double** clustersSum, int* clustersSize, int index, int k){
    int i, j;
    if (clustersSize[index] == 0){
        for (j = 0; j < k; j++){
            clustersSum[index][j] = vector[j];
        }
    }
    else{
        for (i = 0; i < k; i++){
            clustersSum[index][i] += vector[i];
        }
    }
    clustersSize[index] += 1;
}

/** assigning each vector to the closest cluster **/
void clustering(double ** vectors, double ** centroids, double ** clustersSum, int* clustersSize, int k){
    double *currVector;
    int i, j, l, p, isFirst,clusterIndex ;
    double currSum, minDis;
    currVector=(double *) calloc(k , sizeof (double));
    assert(currVector != NULL && "Error in allocating memory!");
    for (i = 0; i < n ; i++){
        for (p = 0; p < k; p++){
            currVector[p] = vectors[i][p];
        }
        isFirst = 1;
        minDis = 0;
        clusterIndex = 0;
        for (j = 0; j < k; j++){
            currSum = 0;
            for (l = 0; l < k; l++){
                currSum += ((currVector[l] - centroids[j][l]) * (currVector[l] - centroids[j][l]));
            }
            if (isFirst | (currSum < minDis)){
                minDis = currSum;
                clusterIndex = j;
                isFirst = 0;
            }
        }
        addVectorToCluster(currVector, clustersSum, clustersSize, clusterIndex, k);
    }

    free(currVector);
}

/** calculating centroids - avg of all clusters' vectors **/
void avgCalc(double ** clustersSum, const int * clustersSize, int index, double * newCent, int k){
    int i;
    for(i = 0; i < k; i++){
        newCent[i] = ((clustersSum[index][i])/((double)clustersSize[index]));
    }
}

/** update all centroids according to the assigned clusters **/
void updateCentroid(int k, double ** clustersSum, int * clustersSize, double ** curCentroids ) {
    int i, j;
    double * avg;
    avg = (double *) calloc(k, sizeof (double));
    assert(avg != NULL && "Error in allocating memory!");
    for (i = 0; i < k; i++) {
        avgCalc(clustersSum, clustersSize, i, avg, k);
        for (j = 0; j < k; j++) {
            curCentroids[i][j] = avg[j];
        }
        /** zero current cluster **/
        clustersSize[i] = 0;
    }

    free(avg);
}

/** check if centroids have changed during the past iteration **/
int diff(int k, double ** curCentroids, double ** centroids){
    int i, j;
    for (i = 0; i < k; i++){
        for (j = 0; j < k; j++){
            if (centroids[i][j] != curCentroids[i][j]){
                return 1;
            }
        }
    }
    return 0;
}

/** main function of the kmeans algorithm **/
void kmean(double ** vectors, int k) {
    double ** centroids;
    double ** clustersSum;
    int * clustersSize;
    double ** curCentroids;
    int i, j, p, t, m, counter, isChanged;
    clustersSum = (double **) calloc(k , sizeof (double *));
    assert(clustersSum != NULL && "Error in allocating memory!");
    clustersSize = (int*) calloc(k, sizeof (int));
    assert(clustersSize != NULL && "Error in allocating memory!");
    curCentroids=(double **) calloc(k , sizeof (double*));
    assert(curCentroids != NULL && "Error in allocating memory!");
    centroids = (double **) calloc(k , sizeof(double *));
    assert(centroids != NULL && "Error in allocating memory!");

    counter = 0;
    isChanged = 1;

    for (p = 0; p < k; p++){
        centroids[p] = (double *) calloc(k, sizeof(double ));
        curCentroids[p] = (double *) calloc(k, sizeof(double ));
        clustersSum[p] = (double *) calloc(k, sizeof(double ));
    }
    /** initializing centroids with first k vectors **/
    for (i = 0; i < k; i++){
        for (j = 0; j < k; j++){
            centroids[i][j] = vectors[i][j];
        }
    }

    /** loop until convergence **/
    while ((counter < 300) && isChanged){
        clustering(vectors, centroids, clustersSum, clustersSize, k);
        updateCentroid(k, clustersSum, clustersSize, curCentroids);
        isChanged = diff(k, curCentroids, centroids);
        for (t = 0; t < k; t++){
            for (m = 0; m < k; m++){
                centroids[t][m] = curCentroids[t][m]; /** update centroids **/
            }
        }

        counter += 1;
    }
    printMat(centroids, k, k);
    /** free all in use memory **/
    free(clustersSize);
    freeAll(centroids, k);s
    freeAll(clustersSum, k);
    freeAll(curCentroids, k);
}




