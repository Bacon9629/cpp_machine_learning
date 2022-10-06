#include <iostream>
#include <vector>
#include <cmath>
#include <time.h>

using namespace std;

class Matrix{
public:
    vector<vector<double>> matrix;

    Matrix() {
        init(0, 0, 0);
    }

    Matrix(vector<vector<double>> _vector){
        init(_vector);
    }

    Matrix(size_t row, size_t col, double init_val){
        init(row, col, init_val);
    }

    inline static size_t row(Matrix *_matrix){
        return _matrix->matrix.size();
    }

    inline static size_t col(Matrix *_matrix){
        return _matrix->matrix[0].size();
    }

    static Matrix dot(Matrix *matrix_a, Matrix *matrix_b){
        size_t row_a = Matrix::row(matrix_a);
        size_t col_a = Matrix::col(matrix_a);
        size_t row_b = Matrix::row(matrix_b);
        size_t col_b = Matrix::col(matrix_b);

        if (col_a != row_b){
            std::cout << "shape wrong" << std::endl;
        }

        const size_t row_result = row_a;
        const size_t col_result = col_b;

        vector<vector<double>> result(row_result, vector<double>(col_result));

        for (int r = 0; r < row_result; r++) {
            for (int c = 0; c < col_result; c++) {
                // 指定在result內的哪個位置
                // 接下來依照指定的result位置取出a、b的值來做計算

                for (int i = 0; i < col_a; i++) {
                    result[r][c] += matrix_a->matrix[r][i] * matrix_b->matrix[i][c];
                }

            }
        }

        Matrix temp(result);
        return temp;
    }

    inline static Matrix transpose(Matrix *_matrix) {
        vector<vector<double>> result(_matrix->col(), vector<double>(_matrix->row()));

        for (size_t i = 0; i < _matrix->row(); i++){
            for (size_t j = 0; j < _matrix->col(); j++){
                result[j][i] = _matrix->matrix[i][j];
            }
        }

        Matrix temp(result);
        return temp;
    }

    static Matrix add(Matrix *matrix_a, Matrix *matrix_b){
        if (matrix_a->row() != matrix_b->row() || matrix_a->col() != matrix_b->col()){
            std::cout << "shape wrong" << std::endl;
        }
        Matrix result(matrix_a->row(), matrix_b->col(), 0);
        for (int i=0;i<result.row();i++){
            for (int j=0;j<result.col();j++){
                result.matrix[i][j] = matrix_a->matrix[i][j] + matrix_b->matrix[i][j];
            }
        }
        return result;
    }

    inline static Matrix add(Matrix *matrix, double val){
        Matrix val_matrix(matrix->row(), matrix->col(), val);
        Matrix result_matrix(Matrix::add(matrix, &val_matrix));
        return result_matrix;
    }

    static Matrix reduce(Matrix *matrix_a, Matrix *matrix_b){
        if (matrix_a->row() != matrix_b->row() || matrix_a->col() != matrix_b->col()){
            std::cout << "shape wrong" << std::endl;
        }
        Matrix result(matrix_a->row(), matrix_b->col(), 0);
        for (int i=0;i<result.row();i++){
            for (int j=0;j<result.col();j++){
                result.matrix[i][j] = matrix_a->matrix[i][j] - matrix_b->matrix[i][j];
            }
        }
        return result;
    }

    inline static Matrix reduce(Matrix *matrix, double val){
        Matrix val_matrix(matrix->row(), matrix->col(), val);
        Matrix result_matrix(Matrix::reduce(matrix, &val_matrix));
        return result_matrix;
    }

    inline static Matrix multiplication(Matrix *matrix_a, Matrix *matrix_b){
        Matrix result(matrix_a->row(), matrix_a->col(), 0);
        for (int i=0;i<result.row();i++){
            for (int j=0;j<result.col();j++){
                result.matrix[i][j] = matrix_a->matrix[i][j] * matrix_b->matrix[i][j];
            }
        }
        return result;
    }

    inline static Matrix multiplication(Matrix *matrix, double val){
        Matrix val_matrix(matrix->row(), matrix->col(), val);
        Matrix result_matrix(Matrix::multiplication(matrix, &val_matrix));
        return result_matrix;
    }

    inline static Matrix division(Matrix *matrix_a, Matrix *matrix_b){
        Matrix result(matrix_a->row(), matrix_a->col(), 0);
        for (int i=0;i<result.row();i++){
            for (int j=0;j<result.col();j++){
                result.matrix[i][j] = matrix_a->matrix[i][j] / matrix_b->matrix[i][j];
            }
        }
        return result;
    }

    inline static Matrix division(Matrix *matrix, double val){
        Matrix val_matrix(matrix->row(), matrix->col(), val);
        Matrix result_matrix(Matrix::division(matrix, &val_matrix));
        return result_matrix;
    }

    void init(vector<vector<double>> _vector){
        this->matrix = {_vector.begin(), _vector.end()};
    }

    void init(size_t row, size_t col, double init_val){
        vector<vector<double>> _vector = vector<vector<double>>(row, vector<double>(col, init_val));
        init(_vector);
    }

    inline size_t row() {
        return Matrix::row(this);
    }

    inline size_t col() {
        return Matrix::col(this);
    }

    inline Matrix dot(Matrix *matrix_b) {
        return Matrix::dot(this, matrix_b);
    }

    inline Matrix transpose() {
        return Matrix::transpose(this);
    }

    inline Matrix add(Matrix *_matrix) {
        return Matrix::add(this, _matrix);
    }

    inline Matrix add(double val) {
        return Matrix::add(this, val);
    }

    inline Matrix reduce(Matrix *_matrix) {
        return Matrix::reduce(this, _matrix);
    }

    inline Matrix reduce(double val) {
        return Matrix::reduce(this, val);
    }

    inline Matrix multiplication(Matrix *_matrix) {
        return Matrix::multiplication(this, _matrix);
    }

    inline Matrix multiplication(double val) {
        return Matrix::multiplication(this, val);
    }

    inline Matrix division(Matrix *_matrix) {
        return Matrix::division(this, _matrix);
    }

    inline Matrix division(double val) {
        return Matrix::division(this, val);
    }

    inline void random_matrix(){
        srand(time(NULL));
        for (int i=0;i<this->row();i++){
            for (int j=0;j<this->col();j++){
                matrix[i][j] = rand() / (RAND_MAX + 1.0);
                matrix[i][j] -= 0.5;
                matrix[i][j] *= 3;
            }
        }
    }

    inline void print_matrix(){
        for (int i=0;i<this->row();i++){
            for (int j=0;j<this->col();j++){
                cout << matrix[i][j] << ", ";
            }
            cout << endl;
        }
    }

};

Matrix Sigmoid(Matrix x){
    Matrix result(x.row(), x.col(), 0);

    for (int i = 0; i< x.row(); i++){
        for (int j = 0; j< x.col(); j++) {
            result.matrix[i][j] = 1 / (1 + exp(-x.matrix[i][j]));
        }
    }
    return result;
}


Matrix DeltaSGD(Matrix &W, Matrix X, /*target*/ Matrix D){
    double alpha = 0.9;
    Matrix dWsum = Matrix(W.row(), W.col(), 0);
    size_t N = 4;

    for (size_t k=0;k<N;k++){

        Matrix x = Matrix(1, X.col(), 0);
        x.matrix[0] = X.matrix[k];
        Matrix d = Matrix(1, D.col(), 0);
        d.matrix[0] = D.matrix[k];

        Matrix v = Matrix::dot(&x, &W);

        Matrix y = Sigmoid(v);

        Matrix e = Matrix::reduce(&y, &d);

        Matrix _y = y.reduce(1.0);  // y-1
        _y = _y.multiplication(-1.0); // -1*(y-1)
        Matrix delta = Matrix::multiplication(&y, &_y);
        delta = Matrix::multiplication(&delta, &e);

        Matrix temp = Matrix::transpose(&x);
        Matrix dW = Matrix::dot(&temp, &delta);

        W = W.reduce(&dW);
    }

    return W;

}

Matrix DeltaBatch(Matrix &W, Matrix X, /*target*/ Matrix D){
    double alpha = 0.9;
    Matrix dWsum = Matrix(W.row(), W.col(), 0);
    size_t N = 4;

    for (size_t k=0;k<N;k++){

        Matrix x = Matrix(1, X.col(), 0);
        x.matrix[0] = X.matrix[k];
        Matrix d = Matrix(1, D.col(), 0);
        d.matrix[0] = D.matrix[k];

        Matrix v = Matrix::dot(&x, &W);

        Matrix y = Sigmoid(v);

        Matrix e = Matrix::reduce(&y, &d);

        Matrix _y = y.reduce(1.0);  // y-1
        _y = _y.multiplication(-1.0); // -1*(y-1)
        Matrix delta = Matrix::multiplication(&y, &_y);
        delta = Matrix::multiplication(&delta, &e);

        Matrix temp = Matrix::transpose(&x);
        Matrix dW = Matrix::dot(&temp, &delta);

        dWsum = dWsum.add(&dW);

    }

    Matrix dWavg = dWsum.division(N);
    dWavg = dWavg.multiplication(alpha);
    W = W.reduce(&dWavg);

    return W;

}

int main(){

    /*
     * x:
     *  dim = 4 * 3; 四筆資料
     *
     * Target:
     *  dim = 4 * 1
     *
     * input:
     *  dim = 3
     *
     * out:
     *  dim = 1
     *
     * epoch = 40000
     * alpha = 0.9
     * batch_size = 4
     * */

    std::cout << "start" << endl;

    int epoch = 4000;
    double alpha = 0.9;
    int batch_size = 4;

    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    vector<vector<double>> temp_target = {{0}, {0}, {1}, {1}};

    Matrix X = Matrix(temp_x);
    Matrix D = Matrix(temp_target);
    Matrix W = Matrix(X.col(), D.col(), 0);
    W.random_matrix();

    // training
    for (int i=0;i<epoch;i++){
//        W = DeltaBatch(W, X, D);
        W = DeltaSGD(W, X, D);
        if (!(i % 5000)){
            cout << "epoch" << " : " << i << endl;
        }
    }

    cout << "end of training\n\n";


    // show result and 驗證
    Matrix v = X.dot(&W);
    Matrix y = Sigmoid(v);

    cout << "input:" << endl;
    X.print_matrix();

    cout << "output:" << endl;
    y.print_matrix();



}
