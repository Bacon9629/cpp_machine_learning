//
// Created by Bacon on 2022/12/14.
//

#ifndef AI_CLASS_MNIST_DATASET_H
#define AI_CLASS_MNIST_DATASET_H

Matrix *load_images(string file_name, int length) {
    char pix[784]; // 用來暫存二進制資料
    Matrix *img_matrix = new Matrix(length, 1, 28, 28); // 存放影像
    std::ifstream file;
    file.open(file_name, ios::binary); // 用二進制方式讀取檔案
    isFileExists(file, file_name); // 確認檔案是否存在

// 先拿出前面16bytes不必要的資訊
    char p[16];
    file.read(p, 16);

// 讀取影像
    for (int b = 0; b < length; b++) {
        file.read(pix, 784);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                img_matrix->matrix_4d[b][0][r][c] = (unsigned char) pix[r * 28 + c] / 255.;
            }
        }
    }

// 關閉檔案
    file.close();

    return img_matrix;
}

Matrix *load_label(string file_name, int length) {
    char label[1];// 用來暫存二進制資料
    Matrix *label_matrix = new Matrix(length, 1);  // 存放label
    std::ifstream file;
    file.open(file_name, ios::binary); // 用二進制方式讀取檔案
    isFileExists(file, file_name); // 確認檔案是否存在

    // 先拿出前面8bytes不必要的資訊
    char p[8];
    file.read(p, 8);

    // 讀取label
    for (int i = 0; i < length; i++) {
        file.read(label, 1);
        label_matrix->matrix[i][0] = (unsigned char) label[0];
    }

    // 關閉檔案
    file.close();

    return label_matrix;
}

#endif //AI_CLASS_MNIST_DATASET_H
