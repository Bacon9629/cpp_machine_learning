//
// Created by Bacon on 2022/9/22.
//

#include <iostream>

void dot(double *a[], double *b[], double *result[], const int row_a, const int col_a,  const int row_b, const int col_b){

    if (col_a != row_b){
        std::cout << "shape wrong" << std::endl;
    }

    const int row_result = row_a;
    const int col_result = col_b;

    for (int r = 0; r < row_result; r++) {
        for (int c = 0; c < col_result; c++) {
            // 指定在result內的哪個位置
            // 接下來依照指定的result位置取出a、b的值來做計算

            for (int i = 0; i < col_a; i++) {
                result[r][c] += a[r][i] * b[i][c];
            }

        }
    }
}
//
//int main(){
//    double *a[10];
//    double *b[10];
//    double *result[10];
//    dot(a, b, result, 10, 10, 10, 10);
//}

