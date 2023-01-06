#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <assert.h>
#include "Matrix.h"
#include <xmmintrin.h>
//#include <vector>
#include <thread>

#define SHOW_MATRIX_PTR

using namespace std;

void test(double *a){

}

int main(){
//    double START,END;
//    Matrix a(9, 9, 0);
//    Matrix b(9, 9, 0);
//    a.set_matrix_1_to_x();
//    b.set_matrix_1_to_x();
//
//    START = clock();
//
//    Matrix::dot(a, b).print_matrix();
//    Matrix::dot2(a, b).print_matrix();
//
//    END = clock();
//    cout << (END - START) / CLOCKS_PER_SEC << endl;



    double START,END;
    Matrix a(1000, 1000, 0);
    Matrix b(1000, 1000, 0);
    a.set_matrix_1_to_x();
    b.set_matrix_1_to_x();

    START = clock();

    Matrix::dot(a, b);

    END = clock();
    cout << (END - START) / CLOCKS_PER_SEC << endl;


    START = clock();

    Matrix::dot2(a, b);

    END = clock();
    cout << (END - START) / CLOCKS_PER_SEC << endl;


    return 0;
}
