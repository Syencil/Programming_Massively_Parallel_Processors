// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2019/12/25

#include <cuda.h>
#include <random>
#include <iostream>

// 此项目分为两个部分，kogge_stone进行少量数据的prefix sum，brent_kung进行任意规模数据量的prefix sum
// kogge_stone和cpu计算部分使用kahan浮点数累加算法提升精度
// brent_kung被用在任意规模数量的single-pass prefix sum算法中。并利用global memory和atomic operation以及__threadfence()实现block之间的通信。

// CPU进行计算的函数以及验证结果函数
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

template <typename T>
void prefixsum_cpu( T *array,  T *output, int length){
    T sum = 0;
    for (int i =0; i < length; ++i){
        sum += array[i];
        output[i] = sum;
    }
}

template <typename T>
void prefixsum_cpu_kahan( T *array,  T *output, int length){
    T sum = 0;
    float c = 0;
    float tmp ;
    for (int i =0; i < length; ++i){
        tmp = array[i] - c;
        c = sum + tmp - sum - tmp;
        sum += tmp;
        output[i] = sum;
    }
}

template <class T>
bool is_equale(const T *array1, const T *array2, int length){
    for (int i = 0; i< length; ++i){
        if(abs(array1[i] - array2[i])> 0.0001){
            return false;
        }
    }
    return true;
}

// 一些参数设置
const unsigned int length = 2048;
const unsigned int section = 32;
const int thread_num = section;
const int block_num = (length + thread_num - 1) / thread_num;


__device__ float aux[block_num] = {0};
__device__ int flags[block_num + 1] = {1};

//length不大 可以用一个block处理
__global__ void  kogge_stone(float *array, float *output, const unsigned int max_len){
    extern __shared__ float array_ds[];
    // 读入shared memory
    int tx = threadIdx.x;
    if (tx < max_len){
        array_ds[tx] = array[tx];
    }

    // Kahan浮点数加法
    float c = 0.0;
    for (int stride = 1; stride < max_len; stride *= 2){
        __syncthreads();
        if (tx >= stride && tx < max_len){
            float tmp = array_ds[tx] - c;
            c = array_ds[tx- stride] + tmp - array_ds[tx- stride] - tmp;
            array_ds[tx] = tmp + array_ds[tx - stride];
        }
    }
    if (tx < max_len){
        output[tx] = array_ds[tx];
    }
}

// 假设length 非常大 shared memory都存不下， 此时只处理block里面的数据，使用aux数组
__global__ void brent_kung(float *array, float *output, const unsigned int max_len){
    extern __shared__ float array_ds[];
    // 读入shared memory
    int bidx = blockIdx.x;
    int bdx = blockDim.x;
    int tx = threadIdx.x;
    int idx = 2 * bidx * bdx + tx;
    int pidx;

    if (idx < max_len){
        array_ds[tx] = array[idx];
    }
    if (idx + bdx < max_len){
        array_ds[tx + bdx] = array[idx + bdx];
    }

    // stage 1
    for (unsigned int stride = 1; stride < max_len ; stride*=2){
        __syncthreads();
        pidx = (tx + 1) * stride * 2 - 1;
        if ( pidx < section ){
            array_ds[pidx] += array_ds[pidx - stride];
        }
    }

    // stage 2 reversed tree
    for (unsigned int stride = section / 2; stride > 0; stride/=2){
        __syncthreads();
        pidx = (tx + 1) * stride * 2 - 1;
        if ( pidx + stride < section){
            array_ds[pidx + stride] += array_ds[pidx];
        }
    }
    __syncthreads();

    // 进行block间通信
    __shared__ float val;
    if (tx == bdx - 1){
        while (atomicAdd(&flags[bidx], 0) == 0){

        }
        val = aux[bidx];
        aux[bidx + 1] = val + array_ds[section - 1];
        // 保证在执行atomicAdd之前aux数组更新完成了
        __threadfence();
        atomicAdd(&flags[bidx + 1], 1);
    }
    __syncthreads();
    idx = 2 * bidx * bdx + tx;
    if (idx < max_len){
        output[idx] = array_ds[tx] + val ;
    }
    if (idx + bdx < max_len){
        output[idx+ bdx] = array_ds[tx + bdx] + val;
    }

}


int main(int args, char **argv){

    printf("Block num is %d\nThread num is %d\n",block_num, thread_num);

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
    for (int i = 0; i < length; ++i){
       array_host[i] = distribution(e);
    }

    HANDLE_ERROR(cudaMalloc((void**)&array_device, sizeof(float) * length));
    HANDLE_ERROR(cudaMalloc((void**)&output_device, sizeof(float) * length));
    HANDLE_ERROR(cudaMemcpy(array_device, array_host, sizeof(float) * length, cudaMemcpyHostToDevice));

    // 记录时间并启动kernel，同时记录结束时间
    HANDLE_ERROR(cudaEventRecord(start, 0));

    //    kogge_stone<<<1, thread_num, section * sizeof(float)>>>(array_device, output_device, length);
    //    std:: cout << "kogge_stone"<< std::endl;

    brent_kung<<<block_num, (thread_num+1) / 2, section * sizeof(float)>>>(array_device, output_device, length);
    std:: cout << "brent_kung"<< std::endl;


    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsed_time;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    std::cout << "Elapsed Time is "<< elapsed_time << std::endl;

    // 将数据拷贝出来
    HANDLE_ERROR(cudaMemcpy(output_host, output_device, sizeof(float) * length, cudaMemcpyDeviceToHost));

    // 验证结果
    prefixsum_cpu_kahan(array_host, output_cpu, length);

    if (is_equale(output_cpu, output_host, length)){
        std::cout << "Answer is Correct"<< std::endl;
    }else{
        std::cout << "Answer is Wrong"<< std::endl;
        for (int i = 0; i < length; ++i){
            printf("%d  %f  %f  %f \n",i, array_host[i], output_cpu[i], output_host[i]);
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