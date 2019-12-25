// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2019/12/25

#include <cuda.h>
#include <random>
#include <iostream>


// CPU进行计算的函数以及验证结果函数
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template <class T>
void prefixsum_cpu( T *array,  T *output, int length){
    T sum = 0;
    for (int i =0; i < length; ++i){
        sum += array[i];
        output[i] = sum;
    }
}

template <class T>
bool is_equale(const T *array1, const T *array2, int length){
    for(int i = 0; i < length; ++i){
        if (array1[i] - array2[2] > 1e-3){
            return false;
        }
    }
    return true;
}

// 一些参数设置
const int length = 100;
const int thread_num = length;


//length不大 可以用一个block处理
__global__ void  kogge_stone(float *array, float *output){
//    assert(length <= blockDim.x);
    __shared__ float array_ds[length];
    // 读入shared memory
    int tx = threadIdx.x;
    if (tx < length){
        array_ds[tx] = array[tx];
    }

    // 计算
    // 浮点数累加精度损失
//    for (int stride = 1; stride < length; stride *= 2){
//        __syncthreads();
//        if (tx >= stride && tx < length){
//            array_ds[tx] += array_ds[tx - stride];
//        }
//    }
    // Kahan浮点数加法
    float c = 0.0;
    for (int stride = 1; stride < length; stride *= 2){
        __syncthreads();
        if (tx >= stride && tx < length){
            array_ds[tx - stride] = array_ds[tx - stride] - c;
            c = array_ds[tx] + array_ds[tx - stride] - array_ds[tx] - array_ds[tx - stride];
            array_ds[tx] += array[tx - stride];
        }
    }

    if (tx < length){
        output[tx] = array_ds[tx];
    }

}


int main(int args, char **argv){
    // 声明
    float *array_host = new float[length];
    float *output_host = new float[length];
    float *output_cpu = new float[length];

    float *array_device, *output_device ;

    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    // 初始化array以及分配空间
    std::default_random_engine e;
    std::uniform_real_distribution<float> distribution(-10, 10);
    std::cout<< "Random" << std::endl;
    for (int i = 0; i < length; ++i){
       array_host[i] = distribution(e);
    }

    HANDLE_ERROR(cudaMalloc((void**)&array_device, sizeof(float) * length));
    HANDLE_ERROR(cudaMalloc((void**)&output_device, sizeof(float) * length));
    HANDLE_ERROR(cudaMemcpy(array_device, array_host, sizeof(float) * length, cudaMemcpyHostToDevice));

    // 记录时间并启动kernel，同时记录结束时间
    HANDLE_ERROR(cudaEventRecord(start, 0));
    kogge_stone<<<1, thread_num>>>(array_device, output_device);
    std:: cout << "kogge_stone"<< std::endl;
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "Elapsed Time is "<< elapsed_time << std::endl;

    // 将数据拷贝出来
    HANDLE_ERROR(cudaMemcpy(output_host, output_device, sizeof(float) * length, cudaMemcpyDeviceToHost));

    // 验证结果
    prefixsum_cpu(array_host, output_cpu, length);
    if (is_equale(output_cpu, output_host, length)){
        std::cout << "Answer is Correct"<< std::endl;
    }else{
        std::cout << "Answer is Wrong"<< std::endl;
        for (int i = 0; i < length; ++i){
            std::cout<< array_host[i] << " ";
        }
        std::cout<<std::endl;
        for (int i = 0; i < length; ++i){
            std::cout<< output_cpu[i] << " ";
        }
        std::cout<<std::endl;
        for (int i = 0; i < length; ++i){
            std::cout<< output_host[i] << " ";
        }
    }

    // Destroy
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(array_device));
    HANDLE_ERROR(cudaFree(output_device));
    delete[] array_host;
    delete[] output_cpu;
    delete[] output_host;

}