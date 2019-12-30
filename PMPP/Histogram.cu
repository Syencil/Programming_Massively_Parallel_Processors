// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2019/12/27

#include <cuda.h>
#include <random>

static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void histogram_cpu(const unsigned char *array, unsigned int *hist, unsigned int max_len){
    int idx;
    for (int i = 0; i < max_len; ++i){
        idx = array[i] - 'a';
        if (0 <= idx && idx < 26){
            hist[idx] += 1;
        }
    }
}

bool is_equal(const unsigned int *hist1, const unsigned int *hist2, const int &max_len){
    for (unsigned int i = 0; i < max_len; ++i){
        if (hist1[i] != hist2[i]){
            return false;
        }
    }
    return true;
}

// 一些参数
const int length = 80;
const int thread_num = 32;
const int per_thread = 2;
const int hist_num = 26;
const int block_num = (length + thread_num * per_thread - 1) / (thread_num * per_thread);

__device__ unsigned int hist_global[hist_num] = {0};

__global__ void histogram(const unsigned char *array, unsigned int max_len){
    __shared__ float hist_ds[hist_num];
    int pos;
    int pre = -1;
    int acc = 0;
    int tx = threadIdx.x;
    int bidx = blockIdx.x;
    int bdx = blockDim.x;
    int idx = tx + bidx * bdx * per_thread;
    for (unsigned int i = tx; i < hist_num; i += bdx){
        hist_ds[i] = 0u;
    }
    __syncthreads();
    for (unsigned int i = idx; i < (bidx+1) * bdx * per_thread && i < max_len; i += bdx){
        pos = array[i] - 'a';
        if (pre != pos){
            if (0 <= pre && pre < hist_num){
                atomicAdd(&hist_ds[pre], acc);
            }
            acc = 1;
            pre = pos;
        }else{
            acc +=1;
        }
    }
    if (0 <= pre && pre < hist_num){
        atomicAdd(&hist_ds[pre], acc);
    }
    __syncthreads();
    for (unsigned int i = tx; i < hist_num; i += bdx){
        atomicAdd(&hist_global[i], hist_ds[i]);
    }
}

int main(int args, char **argv){
    printf("Block num is %d\nThread num is %d\n",block_num, thread_num);
    // Definition
    float elapsed_time;
    char tmp[26];
    unsigned char *array_host = new unsigned char[length];
    unsigned int *hist_host = new unsigned int [hist_num];
    unsigned int *hist_cpu = new unsigned int [hist_num];
    unsigned char *array_dev;
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaMalloc((void**)&array_dev, sizeof(char) * length));
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    // Init Host ====> Dev
    for (int i = 0; i < 26; ++i){
        tmp[i] = 'a' + i;
    }
    std::default_random_engine e;
    std::uniform_int_distribution<int> distribution(0, 26);
    for (int i = 0; i < length; ++i){
        array_host[i] = tmp[distribution(e)];
    }
    HANDLE_ERROR(cudaMemcpy(array_dev, array_host, sizeof(char) * length, cudaMemcpyHostToDevice));
    // launch kernel
    HANDLE_ERROR(cudaEventRecord(start, 0));
    histogram<<<block_num, thread_num>>>(array_dev,  length);
    printf("Histogram \n");
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    // elapsed time
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time ,start, stop));
    printf("Elapsed Time is %f \n",elapsed_time);
    // Dev ====> Host
    HANDLE_ERROR(cudaMemcpyFromSymbol(hist_host, hist_global, sizeof(int) * hist_num));
    // verify the output
    histogram_cpu(array_host, hist_cpu, length);
    if (is_equal(hist_host, hist_cpu, hist_num)){
        printf("Answer is Correct\n");
    }else{
        printf("Answer is Wrong\n");
        for (int i = 0; i < hist_num; ++i){
            printf("%d  %d  %d  \n", i, hist_host[i], hist_cpu[i]);
        }
    }
    // Destroy
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(array_dev));
    free(array_host);
    free(hist_host);
    free(hist_cpu);
}