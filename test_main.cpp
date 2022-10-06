#include <iostream>

using namespace std;

int main() {

    const int row = 3;
    const int col = 3;

    int a[row][col] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

    int result[col][row];

    for (int i=0;i<row;i++){
        for (int j=0;j<col;j++){
            result[j][i] = a[i][j];
        }
    }

    for (int i=0;i<row;i++){
        for (int j=0;j<col;j++){
            cout << result[i][j] << ", ";
        }
        cout << endl;
    }


    return 0;
}


