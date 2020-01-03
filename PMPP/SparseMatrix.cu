// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/1/2

#include <cuda.h>
#include <vector>
#include <random>

// 此章节主要是关于稀疏矩阵计算，对应不同类型的稀疏矩阵有不同的存储格式。
// 主要是介绍为主，没什么代码。此处就是Dense-Matrix转CSR，ELL，COO格式

class Matrix{
public:
    int row;
    int column;
    int num;
    std::vector<std::vector<float>> data;

    Matrix(const std::vector<std::vector<float>> &data){
        this->row = data.size();
        this->column = data[0].size();
        for (int r = 0; r < data.size(); ++r){
            std::vector<float> tmp;
            for (int c = 0; c < data[0].size(); ++c){
                tmp.push_back(data[r][c]);
            }
            this->data.push_back(tmp);
        }
    }

    void show(){
        printf(" ===================   Origin Matrix   ===================>\n");
        for (int r = 0; r < this->row; ++r){
            for(int c = 0; c < this->column; ++c){
                printf("%.3f ", data[r][c]);
            }
            printf("\n");
        }
        printf("\n");
    }
};

class CSR{
public:
    int column;
    int row;
    std::vector<int> col_idx;
    std::vector<int> row_ptr;
    std::vector<float> data;

    CSR(const Matrix &matrix){
        this->column =  matrix.data[0].size();
        this->row = matrix.data.size();

        int count = 0;
        row_ptr.push_back(0);
        for (int r = 0; r < this->row; ++r){
            for (int c = 0; c < this->column; ++c){
                float tmp = matrix.data[r][c];
                if (tmp != 0){
                    ++count;
                    data.push_back(tmp);
                    col_idx.push_back(c);
                }
            }
            row_ptr.push_back(count);
        }
    }

    void show(){
        printf(" ===================   CSR   ===================>\n");
        printf("CSR data ===> ");
        for (int i = 0; i < data.size(); ++i){
            printf("%.3f ", data[i]);
        }
        printf("\nCSR col_idx ===> ");
        for (int i = 0; i < col_idx.size(); ++i){
            printf("%d ", col_idx[i]);
        }
        printf("\nCSR row_ptr ===> ");
        for (int i = 0; i < row_ptr.size(); ++i){
            printf("%d ", row_ptr[i]);
        }
        printf("\n\n");
    }
};

class COO{
public:
    int column;
    int row;
    std::vector<int> col_idx;
    std::vector<int> row_idx;
    std::vector<float> data;

    COO(const Matrix &matrix){
        this->column = matrix.column;
        this->row = matrix.row;

        for (int r = 0; r < this->row; ++r){
            for (int c = 0; c < this->column; ++c){
                float tmp = matrix.data[r][c];
                if (tmp != 0){
                    data.push_back(tmp);
                    col_idx.push_back(c);
                    row_idx.push_back(r);
                }
            }
        }
    }

    void show(){
        printf(" ===================   COO   ===================>\n");
        printf("COO data ===> ");
        for (int i = 0; i < data.size(); ++i){
            printf("%.3f ", data[i]);
        }
        printf("\nCOO col_idx ===> ");
        for (int i = 0; i < col_idx.size(); ++i){
            printf("%d ", col_idx[i]);
        }
        printf("\nCOO row_ptr ===> ");
        for (int i = 0; i < row_idx.size(); ++i){
            printf("%d ", row_idx[i]);
        }
        printf("\n\n");
    }
};

class ELL{
public:
    std::vector<std::vector<float>> data;
    std::vector<std::vector<int>> col_idx;

    ELL(const Matrix &matrix){
        int max_len = 0;
        for (int r = 0; r < matrix.row; ++r){
            std::vector<int> tmp_col;
            std::vector<float> tmp_data;
            for (int c = 0; c < matrix.column; ++c){
                float tmp = matrix.data[r][c];
                if (tmp != 0){
                    printf("%d ", c);
                    tmp_col.push_back(c);
                    tmp_data.push_back(tmp);
                }
            }
            if(max_len < tmp_data.size()){
                max_len = tmp_data.size();
            }
            data.push_back(tmp_data);
            col_idx.push_back(tmp_col);
        }
        for (int r = 0; r <  data.size(); ++r){
            for (int c = data[r].size(); c < max_len; ++c){
                data[r].push_back(0);
                col_idx[r].push_back(0);
            }
        }

    }

    void show(){
        printf(" ===================   ELL   ===================>\n");
        for (int r = 0; r < data.size(); ++r){
            for (int c = 0; c < data[0].size(); ++c){
                printf("%.3f ", data[r][c]);
            }
            printf("       ");
            for (int c = 0; c < col_idx[0].size(); ++c){
                printf("%d ", col_idx[r][c]);
//                printf("%d ", c);
            }
            printf("\n");
        }
        printf("\n");
    }
};

const int ROW = 10;
const int COL = 10;

int main(int args, char **argv){
    // 构建稀疏矩阵
    std::default_random_engine e;
    std::uniform_real_distribution<float > probability(0, 1);
    std::uniform_real_distribution<float > number(0, 10);
    std::vector<std::vector<float>> data;
    for (int i=0; i<ROW; ++i){
        std::vector<float> tmp;
        for (int j = 0; j < COL; ++j){
            if(probability(e) < 0.1){
                tmp.push_back(number(e));
            }else{
                tmp.push_back(0);
            }
        }
        data.push_back(tmp);
    }
    Matrix matrix{data};
    matrix.show();
    CSR csr{matrix};
    csr.show();
    COO coo{matrix};
    coo.show();
    ELL ell(matrix);
    ell.show();

}