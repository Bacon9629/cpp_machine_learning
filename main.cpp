#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>

using namespace std;


#include "Matrix.h"

// loss function - start

class LossFunc{
public:
    virtual double forward(Matrix &y, Matrix &target) = 0;
    virtual Matrix& backward(Matrix &y, Matrix &target) = 0;
};

class MSE: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override{
        double result = 0;
        Matrix temp = y - target;
        temp = temp * temp;
        result = temp.sum();
        result /= double (temp.shape[2] * temp.shape[3]);
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(true);
        *result = y - target;
        return *result;
    }
};

class CrossEntropy: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override {
        double result = 0;
        for (size_t i = 0; i< y.shape[2]; i++){
            for (size_t j = 0; j < y.shape[3]; j++){
                double _y = y.get(i, j);
                double _target = target.get(i, j);
                result += -_target * log(_y) - (1 - _target) * log(1 - _y);
            }
        }
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(y.shape[2], y.shape[3], 0, true);

        for (size_t i = 0; i< y.shape[2]; i++){
            for (size_t j = 0; j < y.shape[3]; j++){
                double _y = y.get(i, j);
                double _target = target.get(i, j);
                result->get(i, j) = (_y - _target) / (_y * (1 - _y));
            }
        }

        return *result;
    }
};

class CrossEntropy_SoftMax: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override {
        double result = 0;
        for (size_t i = 0; i< y.shape[2]; i++){
            for (size_t j = 0; j < y.shape[3]; j++){
                if (target.get(i, j) == 1){
                    result -= log(y.get(i, j));
                    break;
                }else{
                    continue;
                }
            }
        }
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(y.matrix, y.shape[2], y.shape[3],  true);
        double *result_matrix = result->matrix;
        double *target_matrix = target.matrix;
        for (size_t i = 0; i< result->size_1d; i++){
            if (target_matrix[i] != 0){
                result_matrix[i] -= 1;
            }
        }
        return *result;
    }
};

// loss function - end


// active function - start

class ActiveFunc{
public:
    virtual Matrix& func_forward(Matrix &x) = 0;
    virtual Matrix& func_backward(Matrix &x) = 0;

};

class Relu: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        double* result_matrix_pointer = result->matrix;
        double* x_matrix_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_matrix_pointer[i];
            result_matrix_pointer[i] = temp > 0 ? temp : 0;
        }

//        for (size_t i = 0; i < x.shape[2]; i++){
//            for (size_t j = 0; j < x.shape[3]; j++){
//                double temp = x.get(i, j);
//                result->get(i, j) = temp > 0 ? temp : 0;
//            }
//        }
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        double* result_matrix_pointer = result->matrix;
        double* x_matrix_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_matrix_pointer[i];
            result_matrix_pointer[i] = temp > 0 ? 1 : 0;
        }

//        for (size_t i = 0; i < x.shape[2]; i++){
//            for (size_t j = 0; j < x.shape[3]; j++){
//                result->get(i, j) = x.get(i, j) > 0 ? 1 : 0;
//            }
//        }
        return *result;
    }
};

class Sigmoid: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape, 0, true);
        double* result_pointer = result->matrix;
        double* x_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_pointer[i];
            result_pointer[i] = 1 / (1 + exp(-x_pointer[i]));
        }


//        for (size_t i = 0; i < x.shape[2]; i++){
//            for (size_t j = 0; j < x.shape[3]; j++){
//                result->get(i, j) = 1 / (1 + exp(-x.get(i, j)));
//
//            }
//        }
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix a = func_forward(x);
        Matrix *result = new Matrix(true);
        *result = (a - 1) * -1 * a;
        return *result;
    }
};

class SoftMax_CrossEntropy: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(true);
        Matrix x_exp = x.exp_();
        Matrix x_total = x_exp.row_sum();
        *result = x_exp / x_total;
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        return *(new Matrix(x.shape[2], x.shape[3], 1, true));
    }
};

class Tanh:public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(true);
        Matrix a = x.exp_();
        Matrix b = a * -1;
        *result = (a - b) / (b - a);
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix *result = new Matrix(true);
        Matrix a = x.exp_();
        Matrix b = a * -1;
        Matrix c = (a - b) / (b - a);
        *result = (c * c - 1) * -1;
        return *result;
    }

};

// active function - end


// Optimizer - start

class Optimizer{
public:
    virtual void gradient_descent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) = 0;
};

class SGD: public Optimizer{
public:
    double eta;

    SGD(double _eta){
        eta = _eta;
    }

    void gradient_descent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
        double _eta = eta / double (w.shape[2]);
        w = w - (grad_w * _eta);
        b = b - (grad_b * _eta);
    }

};

class MMT: public Optimizer{
public:
    double eta;
    double beta = 0.2;
    Matrix last_grad_w;
    Matrix last_grad_b;

    MMT(double _eta){
        init(_eta, 0.2);
    }

    MMT(double _eta, double _beta){
        init(_eta, _beta);
    }

    void init(double _eta, double _beta){
        eta = _eta;
        beta = _beta;
    }

    void gradient_descent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
//        last_grad_w =  alpha * grad_w + beta * last_grad_w;
//        w -= last_grad_w;
        if (last_grad_w.shape[2] == 0){
            last_grad_w = *(new Matrix(grad_w.shape[2], grad_w.shape[3], 0, true));
            last_grad_b = *(new Matrix(grad_b.shape[2], grad_b.shape[3], 0, true));
        }

        last_grad_w = (grad_w * eta) + (last_grad_w * beta);
        w = w - last_grad_w;
        last_grad_b = (grad_b * eta) + (last_grad_b * beta);
        b = b - last_grad_b;
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

    virtual Matrix& forward(Matrix &_x, bool is_train) = 0;
    virtual Matrix& backward(Matrix &_delta, bool is_train) = 0;
    virtual void update() = 0;
};

class DropoutLayer:public Layer{
public:

    double dropout_probability;
    Matrix dropout_matrix;

    DropoutLayer(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer, double _dropout_probability){
        init(input_size, output_size, _activeFunc, _optimizer);
        dropout_probability = _dropout_probability;
    }

    Matrix& construct_random_bool_list(size_t row, size_t col, double probability){
        Matrix *result = new Matrix(row, col, 0, true);
        double a = 0;
        double b = 0;
        for(size_t i = 0; i < row; i++){
            for(size_t j = 0; j < col; j++){
                result->get(i, j) = (double(rand()) / RAND_MAX < probability) ? 0 : 1;
            }
        }
        return *result;
    }

    void init(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
    }

    Matrix& forward(Matrix &_x, bool is_train) override {
        x = _x;
        if (!is_train){
            return x * (1 - dropout_probability);
        }

        dropout_matrix = construct_random_bool_list(x.shape[2], x.shape[3], dropout_probability);
        x = x * dropout_matrix;
        return x;
    }

    Matrix& backward(Matrix &_delta, bool is_train) override {
        if (!is_train){
            return _delta;
        }
        delta = _delta * dropout_matrix;

        return delta;
    }

    void update() override {
    }
};

class ConvLayer: public Layer{
public:

    /***
     * 捲積
     * @param img shape = (img_account, img_row, img_col, img_channel)
     * @param filter shape = (filter_size, kernel_row, kernel_col, kernel_channel)，kernel size 必須要是奇數，filter 必須是正方形
     * @return feature_img, shape = (img_account, img_row, img_col, img_channel)
     */
    static Matrix &convolution(Matrix &img, Matrix &filter){
        assert(filter.shape[1] < img.shape[1]);
        assert(filter.shape[2] < img.shape[2]);
        if (filter.shape[3] != img.shape[3]){
            cout << endl;
        }
        assert(filter.shape[3] == img.shape[3]);  // filter 與 img 的 channel 要一樣
//        assert(filter.shape[1] == img.shape[2]);  // filter 必須正方形

        size_t channel_size = img.shape[3];
        size_t filter_size = filter.shape[0];
        const size_t kernel_high = filter.shape[1];
        const size_t kernel_width = filter.shape[2];
        Matrix *result = new Matrix(img.shape[0], img.shape[1] - kernel_high + 1, img.shape[2] - kernel_width + 1, filter.shape[0], 0, true);
        size_t kernel_area_size = kernel_high * kernel_width;

        size_t* kernel_reflect_img_pos_idx = new size_t[kernel_high * kernel_width];

        // 建立kernel_reflect_img_pos_idx
        kernel_reflect_img_pos_idx[0] = 0;
        for (size_t col = 1; col < kernel_width; col++){
            kernel_reflect_img_pos_idx[col] = kernel_reflect_img_pos_idx[0] + col * channel_size;
        }

        size_t source_img_width = img.shape[2];
        for (size_t row = 1; row < kernel_high; row++){  // 最左排完成
            kernel_reflect_img_pos_idx[row * kernel_width + 0] = kernel_reflect_img_pos_idx[(row-1) * kernel_width] + channel_size * source_img_width;
            for (size_t col = 1; col < kernel_width; col++){
                kernel_reflect_img_pos_idx[row * kernel_width + col] = kernel_reflect_img_pos_idx[row * kernel_width] + col * channel_size;
            }
        }


        // 捲積
//        size_t col_jump = kernel_width * channel_size;
        for (size_t k = 0; k < img.shape[0]; k++){  // 對一張圖片

            for (size_t j = 0; j < filter_size; j++){  // 一張圖片對一個filter

                for (size_t result_row = 0; result_row < result->shape[1]; result_row++){
                    double *img_target_ptr = &(img.get(k, result_row, 0, 0));  // 現在的目標像素點是哪個

                    for (size_t result_col = 0; result_col < result->shape[2]; result_col++) {  // 一個filter對一張圖片中的一個像素點做捲機計算

                        // 對原始照片的一個像素點做卷積運算後存入result
                        double *filter_ptr = &(filter.get(j, 0, 0, 0));  // 現在要做卷積的filter是哪個
                        double temp = 0;
                        for (size_t i = 0; i < kernel_area_size; i++){

                            double *pixel_a = img_target_ptr + kernel_reflect_img_pos_idx[i];  // 目標像素點對映到捲積計算時要與kernel相乘的像素點位置
                            double *kernel_a =  filter_ptr + i * channel_size;
                            for (size_t z = 0; z < channel_size; z++){
                                temp += pixel_a[z] * kernel_a[z];
                            }

                        }
//                        temp += bias.matrix[j];
                        result->get(k, result_row, result_col, j) = temp;  // 把「一個捲積核」的計算結果存進result matrix
                        img_target_ptr += channel_size;
                    }
                }
            }
        }

        delete[] kernel_reflect_img_pos_idx;
        return *result;
    }


    /***
     * 捲積
     * @param img shape = (img_account, img_row, img_col, img_channel)
     * @param filter shape = (filter_size, kernel_row, kernel_col, kernel_channel)，kernel size 必須要是奇數，filter 必須是正方形，kernel_channel與img_channel相同
     * @param bias shape = (1, 1, 1, filter_size)
     * @return feature_img, shape = (img_account, img_row, img_col, img_channel)
     */
    static Matrix &convolution(Matrix &img, Matrix &filter, Matrix &bias){
        assert(filter.shape[1] < img.shape[1]);
        assert(filter.shape[2] < img.shape[2]);
        assert(filter.shape[3] == img.shape[3]);  // filter 與 img 的 channel 要一樣
//        assert(filter.shape[1] == img.shape[2]);  // filter 必須正方形

        size_t channel_size = img.shape[3];
        size_t filter_size = filter.shape[0];
        const size_t kernel_high = filter.shape[1];
        const size_t kernel_width = filter.shape[2];
//        size_t temp_a = (kernel_width - 1) / 2;  // 因為kernel size關係，img必須從temp_a個像素點開始做捲積計算
        Matrix *result = new Matrix(img.shape[0], img.shape[1] - kernel_high + 1, img.shape[2] - kernel_width + 1, filter.shape[0], 0, true);
        size_t kernel_area_size = kernel_high * kernel_width;
//        size_t kernel_reflect_img_pos_idx[kernel_high][kernel_width];
//        size_t *kernel_reflect_img_pos_idx_1D = kernel_reflect_img_pos_idx[0];  // filter中每一個kernel對應到圖片上需要加減多少個點

        size_t *kernel_reflect_img_pos_idx = new size_t [kernel_high * kernel_width];
//        size_t *kernel_reflect_img_pos_idx_1D = kernel_reflect_img_pos_idx.matrix;

//        size_t *(kernel_reflect_img_pos_idx[kernel_width]);
//        kernel_reflect_img_pos_idx[0] = kernel_reflect_img_pos_idx_1D;


        // 建立kernel_reflect_img_pos_idx
        kernel_reflect_img_pos_idx[0] = 0;
        for (size_t col = 1; col < kernel_width; col++){
            kernel_reflect_img_pos_idx[col] = kernel_reflect_img_pos_idx[0] + col * channel_size;
        }

        size_t source_img_width = img.shape[2];
        for (size_t row = 1; row < kernel_high; row++){  // 最左排完成
            kernel_reflect_img_pos_idx[row * kernel_width + 0] = kernel_reflect_img_pos_idx[(row-1) * kernel_width] + channel_size * source_img_width;
            for (size_t col = 1; col < kernel_width; col++){
                kernel_reflect_img_pos_idx[row * kernel_width + col] = kernel_reflect_img_pos_idx[row * kernel_width] + col * channel_size;
            }
        }


        // 捲積
//        size_t col_jump = kernel_width * channel_size;
        for (size_t k = 0; k < img.shape[0]; k++){  // 對一張圖片

            for (size_t j = 0; j < filter_size; j++){  // 一張圖片對一個filter

                for (size_t result_row = 0; result_row < result->shape[1]; result_row++){
                    double *img_target_ptr = &(img.get(k, result_row, 0, 0));  // 現在的目標像素點是哪個

                    for (size_t result_col = 0; result_col < result->shape[2]; result_col++) {  // 一個filter對一張圖片中的一個像素點做捲機計算

                        // 對原始照片的一個像素點做卷積運算後存入result
                        double *filter_ptr = &(filter.get(j, 0, 0, 0));  // 現在要做卷積的filter是哪個
                        double temp = 0;
                        for (size_t i = 0; i < kernel_area_size; i++){

                            double *pixel_a = img_target_ptr + kernel_reflect_img_pos_idx[i];  // 目標像素點對映到捲積計算時要與kernel相乘的像素點位置
                            double *kernel_a =  filter_ptr + i * channel_size;
                            for (size_t z = 0; z < channel_size; z++){
                                temp += pixel_a[z] * kernel_a[z];
                            }

                        }
                        temp += bias.matrix[j];
                        result->get(k, result_row, result_col, j) = temp;  // 把「一個捲積核」的計算結果存進result matrix
                        img_target_ptr += channel_size;
                    }
                }
            }
        }

        delete[] kernel_reflect_img_pos_idx;

        return *result;
    }

    Matrix &convolution_back_get_per_picture_per_channel_d_w(size_t which_picture, size_t which_channel, Matrix &delta_u){
        Matrix *result = new Matrix(w.shape, 0, true);
        Matrix target_x = Matrix::getPictures(x, which_picture, which_picture+1);
        target_x = Matrix::get_per_channel(target_x, which_channel);

        Matrix target_u = Matrix::getPictures(delta_u, which_picture, which_picture+1);
        Matrix u_change;
        u_change = target_u.per_picture_change_shape_3_to_0_for_conv();

        *result = ConvLayer::convolution(target_x, u_change);

        return *result;
    }

    size_t kernel_size, filter_size;

    ConvLayer(size_t _filter_size, size_t _kernel_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        kernel_size = _kernel_size;
        filter_size = _filter_size;
        active_func = _activeFunc;
        optimizer = _optimizer;
//        w = *(new Matrix(_filter_size, _kernel_size, _kernel_size, chanel_size, 0));
        b = *(new Matrix(1, _filter_size, 0, true));
//        w.random_matrix();
//        b.random_matrix();
//        grad_w = *(new Matrix(w.shape[2], w.shape[3], 0));
        grad_b = *(new Matrix(b.shape[2], b.shape[3], 0, true));
    }


    Matrix &forward(Matrix &_x, bool is_train) override {
        if (w.size_1d == 0){
            w = *(new Matrix(filter_size, kernel_size, kernel_size, _x.shape[3], 0, true));
            w.random_matrix();
            grad_w = *(new Matrix(filter_size, kernel_size, kernel_size, _x.shape[3], 0, true));
        }
//        _x.print_shape();
//        w.print_shape();

        x = _x;
        u = ConvLayer::convolution(x, w, b);
        y = active_func->func_forward(u);

        return y;
    }

    Matrix &backward(Matrix &_delta, bool is_train) override {
//        Matrix *result = new Matrix(true);

        Matrix my_delta = active_func->func_backward(u) * _delta;
        Matrix u_delta = my_delta * u;

        Matrix temp(grad_w.shape[3], grad_w.shape[1], grad_w.shape[2], grad_w.shape[0], 0);

        for(size_t i = 0; i < x.shape[0]; i++){
            Matrix temp2(temp.shape, 0);
            for (size_t _channel = 0; _channel < x.shape[3]; _channel++){
                Matrix w_change_i = convolution_back_get_per_picture_per_channel_d_w(i, _channel, u_delta);
                Matrix w_change_i_3_to_0 = w_change_i.shape_3_to_0();
                temp2.set_per_channel(w_change_i_3_to_0, _channel);
            }
            temp = temp + temp2;
        }

        grad_w = temp.shape_3_to_0() / double(x.shape[0]);

        Matrix w_0_to_3_rotate = w.shape_3_to_0();
        w_0_to_3_rotate = w_0_to_3_rotate.rotate_180();

        Matrix padding_delta = Matrix::padding(u_delta, kernel_size-1);
        delta = ConvLayer::convolution(padding_delta, w_0_to_3_rotate);

        return delta;
    }

    void update() override {
        optimizer->gradient_descent(w, b, grad_w, grad_b);
    }


};

class FlattenLayer: public Layer{
public:
    size_t input_shape[4] = {0, 0, 0, 0};
//    size_t output_shape[4] = {0, 0, 0, 0};
    Matrix &forward(Matrix &_x, bool is_train) override {
        Matrix *result = new Matrix(_x, true);
        input_shape[0] = _x.shape[0];
        input_shape[1] = _x.shape[1];
        input_shape[2] = _x.shape[2];
        input_shape[3] = _x.shape[3];
//        output_shape[0] = 1;
//        output_shape[1] = 1;
//        output_shape[2] = _x.shape[0];
//        output_shape[3] = _x.size_1d * _x.shape[0];
        result->reshape(1, 1, _x.shape[0], _x.shape[1] * _x.shape[2] * _x.shape[3]);
        return *result;
    }

    Matrix &backward(Matrix &_delta, bool is_train) override {
        Matrix *result = new Matrix(_delta, true);
        result->reshape(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        return *result;
    }

    void update() override {
    }
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
    size_t output_size;

    DenseLayer(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        init(output_size, _activeFunc, _optimizer);
    }

    DenseLayer(size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        init(output_size, _activeFunc, _optimizer);
    }

    void init(size_t input_size, size_t _output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        w = *(new Matrix(input_size, output_size, 0, true));
        w.random_matrix();
        b = *(new Matrix(1, output_size, 0, true));
        b.random_matrix();
        grad_w = *(new Matrix(w.shape[2], w.shape[3], 0, true));
        grad_b = *(new Matrix(b.shape[2], b.shape[3], 0, true));
        active_func = _activeFunc;
        optimizer = _optimizer;
        output_size = _output_size;
    }

    void init(size_t _output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        active_func = _activeFunc;
        optimizer = _optimizer;
        output_size = _output_size;
    }

    Matrix& forward(Matrix& _x, bool is_train) override{
        if (w.size_1d == 0){
            w = *(new Matrix(_x.shape[3], output_size, 0, true));
            w.random_matrix();
            b = *(new Matrix(1, output_size, 0, true));
            b.random_matrix();
            grad_w = *(new Matrix(w.shape[2], w.shape[3], 0, true));
            grad_b = *(new Matrix(b.shape[2], b.shape[3], 0, true));
        }
        assert(_x.shape[0] == 1 || _x.shape[1] == 1);
        x = _x;
        u = Matrix::dot(x, w);
        u = u + b;
        y = active_func->func_forward(u);
        return y;
    }

    Matrix& backward(Matrix& _delta, bool is_train) override{

        Matrix active_func_back = active_func->func_backward(u);
        Matrix my_delta = _delta * active_func_back;

        Matrix x_t = Matrix::transpose(x);
        grad_w = Matrix::dot(x_t, my_delta);
        grad_w = grad_w / double(x.shape[2]);
        grad_b = _delta.row_sum();
        grad_b = grad_b / double(x.shape[2]);

        Matrix w_t = Matrix::transpose(w);
        delta = Matrix::dot(my_delta, w_t);
        return delta;
    }

    void update() override{
        optimizer->gradient_descent(w, b, grad_w, grad_b);
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

    void train_img_input(size_t epoch, Matrix &x, Matrix &target){
        assert(target.shape[0] == 1);
        assert(target.shape[1] == 1);
        assert(target.shape[2] == x.shape[0]);
        assert(target.shape[3] == layers[layers.size()-1]->y.shape[3]);  // 確認神經網路最終輸出與target的數量是一樣的
        size_t _batch = batch == -1 ? x.shape[0] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[0];  // 存著還有幾筆資料需要訓練

            for (size_t j = 0; data_left_size != 0 ; j++){
                if (data_left_size < _batch){
                    // 如果資料量"不足"填滿一個batch
                    Matrix _x = Matrix::getPictures(x, j * _batch, j * _batch + data_left_size);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + data_left_size);

                    train_one_time(_x, _target);

                    data_left_size = 0;
                }else{
                    // 如果資料量"足夠"填滿一個batch
                    Matrix _x = Matrix::getPictures(x, j * _batch, j * _batch + _batch);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + _batch);

                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }


            }
        }
    }

    void train(size_t epoch, Matrix &x, Matrix &target, bool img_input = false){  // 這裡擴充batch size
        if (!img_input){
            train_img_input(epoch, x, target);
            return;
        }

        assert(x.shape[0] == 1 && x.shape[1] == 1);  // 不可輸入圖片
        assert(target.shape[0] == 1);
        assert(target.shape[1] == 1);
        assert(target.shape[2] == x.shape[2]);
        assert(target.shape[3] == layers[layers.size()-1]->y.shape[3]);  // 確認神經網路最終輸出與target的數量是一樣的

        size_t _batch = batch == -1 ? x.shape[2] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[2];  // 存著還有幾筆資料需要訓練
//            train_one_time(x, target);
//            continue;
            for (size_t j = 0; data_left_size != 0 ; j++){
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

//                    cout << i << endl;
                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }

            }
        }
    }

    inline void train_one_time(Matrix &x, Matrix &target){  // 這裡客製化網路輸入
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y, true);
        }

        Matrix delta = lossFunc->backward(y, target);
        for (size_t i = 0; i < layers.size(); i++){  // 這裡會減到零以下，因此不能使用無號數
            delta = layers[layers.size() - 1 - i]->backward(delta, true);
        }

        for (size_t i = 0; i < layers.size(); ++i){
            layers[i]->update();
        }
    }

    void show(Matrix &x, Matrix &target){
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y, false);
        }

        cout << "\nlabel: " << endl;
        target.print_matrix();

        cout << "\nresult: " << endl;
        y.print_matrix();

        cout << "\nloss: " << lossFunc->forward(y, target) << endl;

    }

};

void img_train(){
    double temp_x[][25] =
            {
                    {0, 1, 1, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 1, 1, 1, 0},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 0,
                            1, 1, 1, 1, 1},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            0, 0, 1, 0, 1,
                            1, 1, 1, 1, 0},
                    {0, 0, 0, 1, 0,
                            0, 0, 1, 1, 0,
                            0, 1, 0, 1, 0,
                            1, 1, 1, 1, 1,
                            0, 0, 0, 1, 0},
                    {1, 1, 1, 1, 1,
                            1, 0, 0, 0, 0,
                            1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            1, 1, 1, 1, 0}
            };
    double temp_target[][5] =
            {{1, 0, 0, 0, 0},
             {0, 1, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 1, 0},
             {0, 0, 0, 0, 1}};

    double temp_validation[][25] =
            {
                    {0, 0, 1, 1, 0,
                            0, 0, 1, 1, 0,
                            0, 1, 0, 1, 0,
                            0, 0, 0, 1, 0,
                            0, 1, 1, 1, 0},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 1,
                            1, 1, 1, 1, 1},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 1,
                            1, 1, 1, 1, 0},
                    {0, 1, 1, 1, 0,
                            0, 1, 0, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 0, 1, 1,
                            0, 1, 1, 1, 0},
                    {0, 1, 1, 1, 1,
                            0, 1, 0, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 0, 1, 0,
                            1, 1, 1, 1, 0}
            };

    double temp_validation_target[][5] =
            {{1, 0, 0, 0, 0},
             {0, 1, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 1, 0},
             {0, 0, 0, 1, 0}};



//    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
//    vector<vector<double>> temp_target = {{0}, {1}, {1}, {0}};
    // data init
    Matrix x = *(new Matrix(temp_x[0], 5, 5, 5, 1, true));
    Matrix x_target = *(new Matrix(temp_target[0], 1, 1, 5, 5, true));
    Matrix validation = *(new Matrix(temp_validation[0], 5, 5, 5, 1, true));
    Matrix validation_target = *(new Matrix(temp_validation_target[0], 1, 1, 5, 5, true));

    x.reshape(1, 1, 5, 25);
    x_target.reshape(1, 1, 5, 5);
    validation.reshape(1, 1, 5, 25);
    validation_target.reshape(1, 1, 5, 5);

    // active func
    Sigmoid sigmoid = Sigmoid();
    SoftMax_CrossEntropy softmax = SoftMax_CrossEntropy();
    Tanh tanh = Tanh();
    Relu relu = Relu();

    /**
     * loss function: cross entropy with softmax
     * */
    MyFrame frame = MyFrame(new CrossEntropy_SoftMax, -1);


    frame.add(new DenseLayer(25, 32, &relu, new MMT(0.1)));
//    frame.add(new DenseLayer(64, 32, &softmax, new SGD(0.1)));
    frame.add(new DropoutLayer(32, 32, &softmax, new MMT(0.001), 0.65));
    frame.add(new DenseLayer(32, 5, &softmax, new MMT(0.1)));

    frame.train(6000, x, x_target);


    frame.show(validation, validation_target);
    frame.show(x, x_target);
}

void type_a_train(){

    double temp_x[][3] = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    double temp_target[][2] = {{0, 1}, {0, 1}, {1, 0}, {1, 0}};

    Matrix x(temp_x[0], 4, 3);
    Matrix target(temp_target[0], 4, 2);

    MyFrame myFrame(new CrossEntropy_SoftMax(), -1);

    myFrame.add(new DenseLayer(3, 32, new Relu(), new SGD(0.3)));
//    myFrame.add(new DenseLayer(32, 32, new Relu(), new SGD(0.3)));
    myFrame.add(new DenseLayer(32, 2, new SoftMax_CrossEntropy(), new SGD(0.3)));

    myFrame.train(3000, x, target);

    myFrame.show(x, target);


}

void test_conv(){
//    double _img[25] = {
//            2, 1, 0, 3, 1, 2, 1, 0, 0, 1,
//            0, 1, 0, 3, 1, 2, 1, 0, 2, 1,
//            0, 1, 1, 3, 0
//    };
//    double _filter[18] = {
//            0.1, 0.2, 0.2, 0.1, 0.3, 0.2,
//            0.3, 0.2, 0.5, 0.3, 0.7, 0.1,
//            0.7, 0.2, 0.8, 0.3, 0.9, 0.2
//    };
//    double _target[1] = {1};

    Matrix img(3, 10, 10, 2, 0);
    img.set_matrix_1_to_x();
    img = img / double(img.size_1d);


    Matrix target(3, 3, 0.4);

    MyFrame frame(new MSE, -1);
    frame.add(new ConvLayer(3, 3, new Sigmoid(), new SGD(0.3)));
    frame.add(new ConvLayer(5, 3, new Sigmoid(), new SGD(0.3)));
    frame.add(new FlattenLayer());
    frame.add(new DenseLayer(3, new Sigmoid(), new SGD(0.3)));

    frame.show(img, target);

    frame.train_img_input(500, img, target);

    frame.show(img, target);


}

int main() {
//    type_a_train();
//    img_train();
    test_conv();


    return 0;
}
