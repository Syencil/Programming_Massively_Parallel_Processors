// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2020/1/3

#include <cuda.h>
#include <random>

const int m = 1000;
const int n = 1048;

static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

bool verify_output(int *array1, int *array2, int len){
    for (int i = 0; i < len; ++i){
        if (array1[i] != array2[i]){
            return false;
        }
    }
    return true;
}

__device__ __host__
void merge(int *array1, int len1, int *array2, int len2, int *output){
    int i{0};
    int j{0};
    int k{0};
    while(i < len1 && j < len2){
        if (array1[i] <= array2[j]){
            output[k++] = array1[i++];
        }else{
            output[k++] = array2[j++];
        }
    }
    if (i == len1){
        while (j < len2){
            output[k++] = array2[j++];
        }
    }else{
        while (i < len1){
            output[k++] = array1[i++];
        }
    }
}

// 书中所使用的算法，每次都移动下限
__device__ __host__
int co_rank_aux(int k, int *A, int m, int *B, int n){
    int i = k < m ? k : m ;
    int j = k - i;
    int i_low = (k - n) < 0 ? 0 : k - n;
    int j_low = (k - m) < 0 ? 0 : k - m;
    int delta;
    while (true){
        if(i > 0 && j < n && A[i-1]>B[j]){
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            i -= delta;
            j += delta;
        }else if(j > 0 && i < m && A[i] <= B[j-1]){
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            i += delta;
            j -= delta;
        }else{
            break;
        }
    }
    return i;
}

// 书中每次用了一个delta来控制，每次移动都是选择移动i或者j的下限，感觉没必要，因为k=i+j，我就直接使用标准的二分查找了
__device__ __host__
int co_rank(int k, int *A, int m, int *B, int n){
    int i_max = k < m - 1? k : m - 1;
    int i_min = k < n ? 0 : k - n ;
    while (i_min < i_max){
        int i = (i_max + i_min + 1) / 2;
        int j = k - i;
        if (i > 0 && j < n && A[i - 1] > B[j]){
            i_max = i - 1;
        }else if (j > 0 && i < m && A[i] <= B[j - 1]){
            i_min = i + 1;
        }else{
            break;
        }
    }
    return (i_max + i_min + 1) / 2;
}

__global__ void merge_co_rank(int *array1, int m, int *array2, int n, int *output){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int section_size = (m + n - 1) / (blockDim.x * gridDim.x) + 1;
    int start_k = tid * section_size;
    int end_k = min((tid + 1) * section_size, m + n);
    int start_i = co_rank(start_k, array1, m, array2, n);
    int end_i = co_rank(end_k, array1, m, array2, n);
    int start_j = start_k - start_i;
    int end_j = end_k - end_i;
    merge(&array1[start_i], end_i - start_i, &array2[start_j], end_j - start_j, &output[start_k]);
}

void show(int *array, int num, std::string str=""){
    printf("%s\n", str.c_str());
    for(int i = 0; i < num; ++i){
        printf("%d ", array[i]);
    }
    printf("\n");
}

void init_order(int *array, int num, int seed = 1){
    std::default_random_engine e;
    e.seed(seed);
    std::uniform_real_distribution<float> prob(0, 1);
    int i = 0;
    int count = 0;
    while (i < num){
        if (prob(e) < 0.5){
            array[i++] = count;
        }
        ++count;
    }
}

int main(int args, char **argv){
    int *array1 = new int [m];
    int *array2 = new int [n];
    int *merge_cpu = new int [m + n];
    int *output_cpu = new int [m + n];

    init_order(array1, m, 1);
    init_order(array2, n,2);

    int *array1_dev, *array2_dev, *output_dev;

    cudaEvent_t start, end;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    HANDLE_ERROR(cudaMalloc((void**)&array1_dev, sizeof(int) * m));
    HANDLE_ERROR(cudaMalloc((void**)&array2_dev, sizeof(int) * n));
    HANDLE_ERROR(cudaMalloc((void**)&output_dev, sizeof(int) * (m + n)));
    HANDLE_ERROR(cudaMemcpy(array1_dev, array1,  sizeof(int) * m, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(array2_dev, array2,  sizeof(int) * n, cudaMemcpyHostToDevice));

    dim3 grid(2);
    dim3 block(16);
    merge(array1, m, array2, n, merge_cpu);

    HANDLE_ERROR(cudaEventRecord(start, 0));
    merge_co_rank<<<grid, block>>>(array1_dev, m, array2_dev, n, output_dev);
//    merge_co_rank<<<grid, block>>>(array2_dev, n, array1_dev, m, output_dev);
    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    float elapsed_time;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));
    printf("Elapsed Time is %f \n",elapsed_time);

    show(array1, m,"array1 ===>" );
    show(array2, n,"array2 ===>");

    HANDLE_ERROR(cudaMemcpy(output_cpu, output_dev, sizeof(int) * (m+n), cudaMemcpyDeviceToHost));
    if (verify_output(output_cpu, merge_cpu, m + n)){
        printf("Answer is Correct\n");
    } else{
        printf("Answer is Wrong\n");
        show(merge_cpu, m+n, "output_cpu ===>");
        show(output_cpu, m+n, "output_device ===>");

    }

    delete []array1;
    delete []array2;
    delete []output_cpu;
    delete []merge_cpu;
    HANDLE_ERROR(cudaFree(array1_dev));
    HANDLE_ERROR(cudaFree(array2_dev));
    HANDLE_ERROR(cudaFree(output_dev));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));


}