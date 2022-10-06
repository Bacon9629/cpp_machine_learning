#include <iostream>
#include <time.h>
#include <vector>

int check() {
    std::cout << "Hello, World!" << std::endl;

    const int _size = 200;
    const int _count = 10;

    const int row_a = _size;
    const int col_a = _size;
    const int rol_b = _size;
    const int col_b = _size;
    const int row_result = row_a;
    const int col_result = col_b;

    if (row_a != col_b){
        std::cout << "shape WRONG！！！！！" << std::endl;
    }

    std::vector<std::vector<int>>

    int a[row_a][col_a];
    int b[rol_b][col_b];

    int result[row_result][col_result];

    double start, end;


    for (int COUNT=0;COUNT<_count;COUNT++) {

        start = clock();  // 計算執行效率 start

        for (int r = 0; r < row_result; r++) {
            for (int c = 0; c < col_result; c++) {
                // 指定在result內的哪個位置
                // 接下來依照指定的result位置取出a、b的值來做計算

                for (int i = 0; i < row_a; i++) {
                    result[r][c] += a[r][i] * b[i][c];
                }

            }
        }

        end = clock();  // 計算執行效率 end

        std::cout << (end - start) / 1000 << std::endl;


//    // print data
//    for (int r=0;r<row_result;r++) {
//        for (int c=0;c<col_result;c++){
//
//            std::cout << result[r][c] << ", ";
//
//        }
//        std::cout << std::endl;
//    }
    }
    return 0;
}
