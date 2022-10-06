#include <iostream>
#include <time.h>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cmath>

using namespace std;

class Matrix{
public:
    vector<vector<double>> matrix;

    Matrix() {
        init(0, 0, 0);
    }

    Matrix(vector<vector<double>> _vector){
        init(std::move(_vector));
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

    inline static Matrix multiplication(Matrix *matrix_a, Matrix *matrix_b){
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

    inline static Matrix multiplication(Matrix *matrix, double val){
        Matrix val_matrix(matrix->row(), matrix->col(), val);
        Matrix result_matrix(Matrix::multiplication(matrix, &val_matrix));
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

class LossFunc{
public:
    virtual double forward(Matrix &y, Matrix &target) = 0;
    virtual Matrix backward(Matrix &y, Matrix &target) = 0;
};

class MSE: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override{
        double result = 0;
        Matrix temp = Matrix::reduce(&y, &target);
        temp = Matrix::multiplication(&temp, &temp);

        for(size_t r=0;r<y.row();++r)
            for(size_t c=0;c<y.col();++c)
                result += temp.matrix[r][c];

        result /= y.row() * y.col();
        return result;
    }

    Matrix backward(Matrix &y, Matrix &target) override {
        return Matrix::reduce(&y, &target);
    }
};

class ActiveFunc{
public:
    virtual Matrix func_forward(Matrix x) = 0;
    virtual Matrix func_backward(Matrix x) = 0;

};

class Sigmoid: public ActiveFunc{
public:
    Matrix func_forward(Matrix x) override {
        Matrix result(x.row(), x.col(), 0);

        for (int i = 0; i< x.row(); i++){
            for (int j = 0; j< x.col(); j++) {
                result.matrix[i][j] = 1 / (1 + exp(-x.matrix[i][j]));
            }
        }
        return result;
    }

    Matrix func_backward(Matrix x) override {
        Matrix a = func_forward(x);
        Matrix b = Matrix::reduce(&a, 1);
        b = Matrix::multiplication(&b, -1);
        return Matrix::multiplication(&a, &b);
    }
};

class Layer{
public:
    Matrix x;  // 輸入
    Matrix y;  // y = xw+b
    Matrix u;  // u = active_func(y)；此層輸出(下一層的輸入)
    Matrix w, b;
    Matrix delta;

    Matrix grad_w;
    Matrix grad_b;

    ActiveFunc *active_func;

    virtual Matrix forward(Matrix x) = 0;
    virtual Matrix backward(Matrix _delta) = 0;
    virtual void update(double eta) = 0;
};

class DenseLayer : public Layer{
public:
//    Matrix x;  // 輸入
//    Matrix y;  // y = xw+b
//    Matrix u;  // u = active_func(y)；此層輸出(下一層的輸入)
//    Matrix w, b;

//    Matrix (*active_func)(Matrix x);

//    double alpha;  // 學習率

    DenseLayer(size_t input_size, size_t output_size, ActiveFunc *activeFunc){
        init(input_size, output_size, activeFunc);
    }

    void init(size_t input_size, size_t output_size, ActiveFunc *activeFunc){
        w = Matrix(input_size, output_size, 0);
        w.random_matrix();
        b = Matrix(1, output_size, 0);
        b.random_matrix();
        grad_w = Matrix(w.row(), w.col(), 0);
        grad_b = Matrix(b.row(), b.col(), 0);
        active_func = activeFunc;
    }

    Matrix forward(Matrix _x) override{
        x = _x;
        u = Matrix::dot(&x, &w);
        u = Matrix::add(&u, &b);
        y = active_func->func_forward(u);
        return y;
    }

    Matrix backward(Matrix _delta) override{
        // 反向傳播的active function該如何處理好呢
//         x.T dot (_delta * active_func->func_backward(x))
        Matrix active_func_back = active_func->func_backward(u);
        Matrix my_delta = Matrix::multiplication(&_delta, &active_func_back);

        Matrix x_t = x.transpose();
        Matrix _grad_w = Matrix::dot(&x_t, &my_delta);
        grad_w = Matrix::add(&grad_w, &_grad_w);

        Matrix _grad_b = my_delta;
        grad_b = Matrix::add(&grad_b, &_grad_b);
//1*30 3*30 =
        Matrix w_t = w.transpose();
        delta = Matrix::dot(&my_delta, &w_t);
//        grad_w.print_matrix();
        return delta;
    }

    void update(double eta) override{
//        w - a(w)
        double _temp = eta / x.row();

//        grad_w.print_matrix();
        Matrix temp_w = Matrix::multiplication(&grad_w, _temp);

//        cout << "temp_w:" << endl;
//        temp_w.print_matrix();

        w = Matrix::reduce(&w, &temp_w);

        Matrix temp_b = Matrix::multiplication(&grad_b, _temp);
        b = Matrix::reduce(&b, &temp_b);

        grad_w = Matrix(grad_w.row(), grad_w.col(), 0);
        grad_b = Matrix(grad_b.row(), grad_b.col(), 0);


//        w.print_matrix();

    }

};

class MyFrame{
    vector<Layer*> layers = vector<Layer*>(0);
    LossFunc *lossFunc;
    double eta;  // 學習率
    int batch;

public:

    MyFrame(LossFunc *_lossFun, double _eta, int _batch){
        lossFunc = _lossFun;
        eta = _eta;
        batch = _batch;
    }

    ~MyFrame(){
        size_t size = layers.size();
        for (size_t i = 0; i < size; ++i) {
            delete layers[i];
        }
        layers.clear();

    }

    void add(Layer *layer){
        layers.push_back(layer);
    }

    void train(size_t epoch, Matrix &x, Matrix &target){
        for (size_t time=0;time<epoch;time++){
            train_one_time(x, target);
        }
    }

    inline void train_one_time(Matrix &x, Matrix &target){
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y);
        }

        Matrix delta = lossFunc->backward(y, target);
        for (int i = layers.size() - 1; i >= 0; --i){
            delta = layers[i]->backward(delta);
        }

        for (size_t i = 0; i < layers.size(); ++i){
            layers[i]->update(eta);
        }
    }

    void show(Matrix &x, Matrix &target){
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y);
        }

        cout << "\nlabel: " << endl;
        target.print_matrix();

        cout << "\nresult: " << endl;
        y.print_matrix();

        cout << "\nloss: " << lossFunc->forward(y, target) << endl;

    }

};

int main() {
//    ALayer aLayer = ALayer(3, 1, new Sigmoid);

//    vector<vector<double>> temp_x = {{0, 1}, {1, 1}};
//    vector<vector<double>> temp_target = {{0}, {1}};

    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    vector<vector<double>> temp_target = {{0}, {1}, {1}, {0}};
//    vector<vector<double>> temp_target = {{0, 1}, {0, 1}, {1, 0}, {1, 0}};

    Matrix x = Matrix(temp_x);
    Matrix target = Matrix(temp_target);

    Sigmoid sigmoid = Sigmoid();
    MSE loss_func = MSE();

    MyFrame frame = MyFrame(&loss_func, 0.9, -1);

    frame.add(new DenseLayer(3, 5, &sigmoid));
    frame.add(new DenseLayer(5, 1, &sigmoid));

    frame.train(4000, x, target);

    frame.show(x, target);
    return 0;
}
