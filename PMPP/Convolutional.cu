// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2019/12/23

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

// 此项目实现多通道卷积，主要涉及const memory，shared memory，cache，corner tuning策略，coalescing读取
// 但是由于水平有限，CHW的一维模式coalescing读取到共享内存暂时不知道怎么处理

// 需要的一些函数
static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

cv::Mat ImageRead(const std::string &image_path, const int &height, const int &width){
    cv::Mat original_image = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat resized_image(height, width, CV_8UC3);
    cv::resize(original_image, resized_image, cv::Size(width, height));
    resized_image.convertTo(resized_image, CV_32FC3);
    return resized_image;
}

// 定义常量内存 以及一些常量参数
const int mask_width = 3;
const int image_channel = 3;
const int image_height = 448;
const int image_width = 448;
const int thread_num = 32;
const int tiled_width = thread_num;

__constant__ float mask_device[image_channel * mask_width *mask_width];

// 定义kernel
__global__ void convolutional_2D(float *image, float *image_conv, const int image_h, const int image_w){

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int timg = tidx + tidy * image_w;
    // 3通道图像 以blockDim为tile大小 CHW 读入shared memory
    __shared__ float image_ds[image_channel * tiled_width * tiled_width];
    // not coalescing =====> total 0.046s
    for (int c = 0; c < image_channel; ++c){
        if (0 <= tidx && tidx < image_w && 0 <= tidy && tidy <=image_h){
            image_ds[(tx + ty * tiled_width) * 3 + c] = image[timg * 3 + c ];
        }else{
            image_ds[(c * tiled_width * tiled_width) + tx + ty * tiled_width ] = 0.0f;
        }
    }
    __syncthreads();
    // 计算
    float output = 0.0f;
    int tile_x_start = blockIdx.x * blockDim.x;
    int tile_x_end = (blockIdx.x + 1) * blockDim.x;
    int tile_y_start = blockIdx.y * blockDim.y;
    int tile_y_end = (blockIdx.y + 1) * blockDim.y;
    int halo = mask_width / 2;

    for (int i = 0; i < mask_width; ++i){
        int x_idx = tidx - halo + i;
        for (int j = 0; j < mask_width; ++j){
            int y_idx = tidy - halo + j;
            for (int c=0; c<image_channel; ++c){
                if (0 <= x_idx && x_idx < image_w && 0 <= y_idx && y_idx <=image_h){
                    if (tile_x_start <= x_idx && x_idx < tile_x_end && tile_y_start <= y_idx && y_idx < tile_y_end){
                        output += mask_device[c * mask_width * mask_width + j * mask_width + i] * image_ds[(((ty - halo + j) * tiled_width + tx - halo + i)) * 3 + c] ;
                    }else{
                        output += mask_device[c * mask_width * mask_width + j * mask_width + i] * image[(x_idx + y_idx * image_w) * image_channel + c] ;
                    }
                }
            }
        }
    }
    if (0 <= tidx && tidx < image_w && 0 <= tidy && tidy <=image_h){
        image_conv[timg] = output;
    }
}


int main(int args, char **argv){
    // 声明变量
    float *mask_host = new float[mask_width*mask_width*image_channel];
    float *image_device, *image_output_device;
    cudaEvent_t start;
    cudaEvent_t end;

    // 初始化事件
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    // 初始化const memory
    for (int i = 0; i < mask_width*mask_width*image_channel; ++i){
        mask_host[i] = 1.0/(mask_width*mask_width*image_channel);
    }

    HANDLE_ERROR(cudaMemcpyToSymbol(mask_device, mask_host, image_channel * mask_width * mask_width * sizeof(float)));

    // 读取图像数据，载入Device中
    std::string image_path = "/work/tensorRT/PMPP/data/164.jpg";
    cv::Mat image_host = ImageRead(image_path, image_height, image_width);


    HANDLE_ERROR(cudaMalloc((void**)&image_device, sizeof(float)*image_host.channels() * image_host.rows * image_host.cols));
    HANDLE_ERROR(cudaMalloc((void**)&image_output_device, sizeof(float) * image_host.rows * image_host.cols));
    HANDLE_ERROR(cudaMemcpy(image_device, image_host.data, sizeof(float)*image_host.channels() * image_host.rows * image_host.cols, cudaMemcpyHostToDevice));

    // 开始记录时间
    HANDLE_ERROR(cudaEventRecord(start, 0));

    // 启动核函数，计算block所需要数量
    int blockDim_x = (image_width+thread_num-1) / thread_num;
    int blockDim_y = (image_height+thread_num-1) / thread_num;
    dim3 block(thread_num, thread_num);
    dim3 grid(blockDim_x, blockDim_y);
    convolutional_2D<<<grid,block>>>(image_device, image_output_device, image_height, image_width);
    std::cout<<"convolution"<<std::endl;

    // 读取计算时间
    HANDLE_ERROR(cudaEventRecord(end, 0));
    float elapsed_time;
    HANDLE_ERROR(cudaEventSynchronize(end));

    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));
    std::cout<< "Execution time is "<<elapsed_time << std::endl;

    // 将数据读取出来
    // 验证计算结果
    cv::Mat render_image = cv::Mat::ones(image_height, image_width, CV_32FC1);
    HANDLE_ERROR(cudaMemcpy(render_image.data, image_output_device, sizeof(float)*image_host.rows * image_host.cols, cudaMemcpyDeviceToHost));
    cv::imwrite("/work/tensorRT/PMPP/data/render.jpg", render_image);

    // destory
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaFree(image_device));
    HANDLE_ERROR(cudaFree(image_output_device));
    return 0;
}