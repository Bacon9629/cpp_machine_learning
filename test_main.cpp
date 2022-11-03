#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
//#include <vector>

using namespace std;

class Matrix{
public:
    double *matrix;
    size_t shape[4] = {0, 0, 0, 0};
    size_t _a[4] = {0, 0, 0, 0};

    Matrix() {
        init(1, 1, 0);
    }

//    Matrix(vector<vector<double>> _vector){
//        init(std::move(_vector));
//    }

    Matrix(size_t row, size_t col, double init_val){
        init(row, col, init_val);
    }

    inline static double get(Matrix &_matrix, size_t a, size_t b, size_t c, size_t d){
        size_t *_a = _matrix._a;
        return _matrix.matrix[a * _a[0] + b * _a[1] + c * _a[2] + d * _a[3]];
    }

    inline static double get(Matrix &_matrix, size_t c, size_t d){
        size_t *_a = _matrix._a;
        return _matrix.matrix[c * _a[2] + d * _a[3]];
    }

    inline static size_t row(Matrix *_matrix){
        return _matrix->shape[2];
    }

    inline static size_t col(Matrix *_matrix){
        return _matrix->shape[3];
    }

    // 取 start 到 end - 1 的row
    inline static Matrix getRow(Matrix &_matrix, size_t start, size_t end){
        if (end > _matrix.row() || start < 0){
            cout << "shape_wrong: Matrix getRow, start: " << start
                 << "  , end: " << end << endl;
        }
        vector<vector<double>> result(_matrix.matrix.begin() + start, _matrix.matrix.begin() + end);
        return Matrix(result);
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

    inline static Matrix expand_row(Matrix *matrix_a, Matrix *matrix_b){
        Matrix _temp_b;
//        if(matrix_a->row() != matrix_b->row()){
        _temp_b = Matrix(matrix_a->row(), matrix_b->col(), 0);
        for (int i=0;i<matrix_a->row();i++){
            _temp_b.matrix[i] = matrix_b->matrix[0];
        }
//        }
        return _temp_b;
    }

    static Matrix add(Matrix *matrix_a, Matrix *matrix_b){
        Matrix _temp_b;
        if(matrix_a->row() != matrix_b->row()){
            _temp_b = expand_row(matrix_a, matrix_b);
            matrix_b = &_temp_b;
        }

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
        Matrix _temp_b;
        if(matrix_a->row() != matrix_b->row()){
            _temp_b = expand_row(matrix_a, matrix_b);
            matrix_b = &_temp_b;
        }

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

    inline static Matrix times(Matrix *matrix_a, Matrix *matrix_b){
        Matrix _temp_b;
        if(matrix_a->row() != matrix_b->row()){
            _temp_b = expand_row(matrix_a, matrix_b);
            matrix_b = &_temp_b;
        }

        Matrix result(matrix_a->row(), matrix_a->col(), 0);
        for (int i=0;i<result.row();i++){
            for (int j=0;j<result.col();j++){
                result.matrix[i][j] = matrix_a->matrix[i][j] * matrix_b->matrix[i][j];
            }
        }
        return result;
    }

    inline static Matrix times(Matrix *matrix, double val){
        Matrix val_matrix(matrix->row(), matrix->col(), val);
        Matrix result_matrix(Matrix::times(matrix, &val_matrix));
        return result_matrix;
    }

    void init(double* _vector){
//        this->matrix = {_vector.begin(), _vector.end()};
//        shape[0] = _vector.size();
//        shape[1] = _vector[0].size();
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
        return Matrix::times(this, _matrix);
    }

    inline Matrix multiplication(double val) {
        return Matrix::times(this, val);
    }

    inline void random_matrix(){
        srand(time(NULL));
        for (int i=0;i<this->row();i++){
            for (int j=0;j<this->col();j++){
                matrix[i][j] = rand() / (RAND_MAX + 1.0) - 0.5;
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

    inline void print_shape(){
        cout << "rol: " << row() << " col: " << col() << endl;
    }

};

int main(){
    return 0;
}