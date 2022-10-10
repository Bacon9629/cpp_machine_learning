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


// loss function - start

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
        temp = Matrix::times(&temp, &temp);

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

// loss function - end


// active function - start

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
        b = Matrix::times(&b, -1);
        return Matrix::times(&a, &b);
    }
};

// active function - end


// Optimizer - start

class Optimizer{
public:
    virtual void gradient_decent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) = 0;
};

class MMT: public Optimizer{
public:
    double eta;
    double beta = 0.9;
    Matrix last_grad_w = Matrix();
    Matrix last_grad_b = Matrix();

    MMT(double _eta){
        init(_eta, 0.9);
    }

    MMT(double _eta, double _beta){
        init(_eta, _beta);
    }

    void init(double _eta, double _beta){
        eta = _eta;
        beta = _beta;
    }

    void gradient_decent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
//        last_grad_w =  alpha * grad_w + beta * last_grad_w;
//        w -= last_grad_w;
        if (last_grad_w.row() == 0){
            last_grad_w = Matrix(grad_w.row(), grad_w.col(), 0);
            last_grad_b = Matrix(grad_b.row(), grad_b.col(), 0);
        }

        Matrix temp_a = Matrix::times(&grad_w, eta);
        Matrix temp_b = Matrix::times(&last_grad_w, beta);
        last_grad_w = Matrix::add(&temp_a, &temp_b);
        w = Matrix::reduce(&w, &last_grad_w);

        temp_a = Matrix::times(&grad_b, eta);
        temp_b = Matrix::times(&last_grad_b, beta);
        last_grad_b = Matrix::add(&temp_a, &temp_b);
        b = Matrix::reduce(&b, &last_grad_b);

    }

};

// Optimizer - end


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
    Optimizer *optimizer;

    virtual Matrix forward(Matrix x) = 0;
    virtual Matrix backward(Matrix _delta) = 0;
    virtual void update() = 0;
};

class DenseLayer : public Layer{
public:
//    Matrix x;  // 輸入
//    Matrix y;  // y = xw+b
//    Matrix u;  // u = active_func(y)；此層輸出(下一層的輸入)
//    Matrix w, b;
//    Matrix delta;
//
//    Matrix grad_w;
//    Matrix grad_b;
//
//    ActiveFunc *active_func;
//    Optimizer *optimizer;

    DenseLayer(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        init(input_size, output_size, _activeFunc, _optimizer);
    }

    void init(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        w = Matrix(input_size, output_size, 0);
        w.random_matrix();
        b = Matrix(1, output_size, 0);
        b.random_matrix();
        grad_w = Matrix(w.row(), w.col(), 0);
        grad_b = Matrix(b.row(), b.col(), 0);
        active_func = _activeFunc;
        optimizer = _optimizer;
    }

    Matrix forward(Matrix _x) override{
        x = _x;
        u = Matrix::dot(&x, &w);
        u = Matrix::add(&u, &b);
        y = active_func->func_forward(u);
        return y;
    }

    Matrix backward(Matrix _delta) override{
//         x.T dot (_delta * active_func->func_backward(x))

        // 初始化gradient_w and b
        grad_w = Matrix(grad_w.row(), grad_w.col(), 0);
        grad_b = Matrix(grad_b.row(), grad_b.col(), 0);

        Matrix active_func_back = active_func->func_backward(u);
        Matrix my_delta = Matrix::times(&_delta, &active_func_back);

        Matrix x_t = x.transpose();
        Matrix _grad_w = Matrix::dot(&x_t, &my_delta);
        grad_w = Matrix::add(&grad_w, &_grad_w);

        Matrix _grad_b = my_delta;
        grad_b = Matrix::add(&grad_b, &_grad_b);

        Matrix w_t = w.transpose();
        delta = Matrix::dot(&my_delta, &w_t);
        return delta;
    }

    void update() override{
        optimizer->gradient_decent(w, b, grad_w, grad_b);
    }

};

class MyFrame{
    vector<Layer*> layers = vector<Layer*>(0);
    LossFunc *lossFunc;
    int batch;

public:

    MyFrame(LossFunc *_lossFun, int _batch){
        lossFunc = _lossFun;
        batch = _batch;
    }

    ~MyFrame(){
        // 這裡是這樣寫嗎?
        layers.clear();
        layers = vector<Layer*>();
    }

    void add(Layer *layer){
        layers.push_back(layer);
    }

    void train(size_t epoch, Matrix &x, Matrix &target){  // 這裡擴充batch size
        size_t _batch = batch == -1 ? x.row() : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.row();  // 存著還有幾筆資料需要訓練
//            train_one_time(x, target);
//            continue;

            for (size_t j = 0; data_left_size > 0 ; j++){
                if (data_left_size < _batch){
                    // 如果資料量"不足"填滿一個batch
                    Matrix _x = Matrix::getRow(x, j * _batch, j * _batch + data_left_size);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + data_left_size);

                    train_one_time(_x, _target);
                    data_left_size = 0;
                }else{
                    // 如果資料量"足夠"填滿一個batch
                    Matrix _x = Matrix::getRow(x, j * _batch, j * _batch + _batch);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + _batch);

                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }

            }
        }
    }

    inline void train_one_time(Matrix &x, Matrix &target){  // 這裡客製化網路輸入
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y);
        }
        Matrix delta = lossFunc->backward(y, target);
        for (int i = layers.size() - 1; i >= 0; --i){
            delta = layers[i]->backward(delta);
        }

        for (size_t i = 0; i < layers.size(); ++i){
            layers[i]->update();
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
    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    vector<vector<double>> temp_target = {{0}, {1}, {1}, {0}};

    // data init
    Matrix x = Matrix(temp_x);
    Matrix target = Matrix(temp_target);

    // active func
    Sigmoid sigmoid = Sigmoid();

    // define network
    /**
     * loss function: MSE
     * */
    MyFrame frame = MyFrame(new MSE, -1);


    /**
     * active function: sigmoid
     * optimizer: MMT
     * */
    frame.add(new DenseLayer(3, 5, &sigmoid, new MMT(0.9)));
    frame.add(new DenseLayer(5, 1, &sigmoid, new MMT(0.9)));


    frame.train(4000, x, target);

    frame.show(x, target);
    return 0;
}
