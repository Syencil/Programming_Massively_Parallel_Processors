// Created by luozhiwang (luozw1994@outlook.com)
// Date: 2019/12/27

#include <cuda.h>
#include <random>

// 稀疏矩阵那一章里面是用的vector来实现的，这一次试一下堆上的动态数组
// 配置一些参数 做BFS求节点路径
const int SOURCE_VERTEX = 0;
const int MAX_VERTEX = 15;
const int BLOCK_QUEUE_NUM = 4;
const int BLOCK_QUEUE_SIZE = 16;
const int BLOCK_SIZE = BLOCK_QUEUE_SIZE * BLOCK_QUEUE_NUM;
texture<int, 1> row_ptr_dev;


static void HandleError(cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

class Graph{
private:
    const  int vertex_num;
     int edge_num = 0;
     int **joint_matrix;
     int *dest;
     int *row_ptr;
private:
    void show_dense_matrix() const{
        printf(" ===================   Origin Matrix   ===================>\n");
        for (int r = 0; r < this->vertex_num; ++r){
            for(int c = 0; c < this->vertex_num; ++c){
                printf("%.1d ", this -> joint_matrix[r][c]);
            }
            printf("\n");
        }
        printf("\n");
    }
    void show_csr_matrix() const{
        printf(" ===================   CSR   ===================>\n");
        printf("\nCSR Dest ===> ");
        for (int i = 0; i < this->edge_num; ++i){
            printf("%d ", this-> dest[i]);
        }
        printf("\nCSR Row_ptr ===> ");
        for (int i = 0; i < this -> vertex_num + 1; ++i){
            printf("%d ", this -> row_ptr[i]);
        }
        printf("\n\n");
    }

public:
    Graph(const  int &vertex_num, const float &sparse_ratio): vertex_num(vertex_num){
        std::default_random_engine e;
        std::uniform_real_distribution<float> prob(0, 1);

        this -> joint_matrix = new  int*[this->vertex_num];
        for(int i = 0; i < this->vertex_num; ++i){
            this -> joint_matrix[i] = new  int[this -> vertex_num];
        }

        // Dense joint matrix
        for (int i = 0; i < vertex_num; ++i){
            for (int j = 0; j < vertex_num; ++j){
                // 自己和自己没有路径
                if (prob(e) <= sparse_ratio && i != j){
                    this -> joint_matrix[i][j] = 1;
                    ++edge_num;
                } else{
                    this -> joint_matrix[i][j] = 0;
                }
            }
        }

        // CSR
        dest = new  int[this -> edge_num];
        row_ptr = new  int[this -> vertex_num + 1];
        int count = 0;
        row_ptr[0] = 0;
        for (int i = 0; i < vertex_num; ++i){
            for (int j = 0; j < vertex_num; ++j){
                if (this -> joint_matrix[i][j] != 0){
                    dest[count] = j;
                    ++count;
                }
            }
            row_ptr[i + 1] = count;
        }
    }

    ~Graph(){
        delete []row_ptr;
        delete []dest;
        for (int i = 0; i < this-> vertex_num; ++i){
            delete []joint_matrix[i];
        }
    }

    void show(int type = 0) const{
        switch (type){
            case 0: this -> show_dense_matrix();
                break;
            case 1: this -> show_csr_matrix();
                break;
            default:
                break;
        }
    }

     int get_edge_num() const{
        return edge_num;
    }
     int** get_joint_matrix() const{
        return joint_matrix;
    }
     int* get_dest() const{
        return dest;
    }
     int* get_row_ptr() const{
        return row_ptr;
    }
};

bool verify_output(int *array1, int *array2, int len){
    bool is_right = true;
    for (int i = 0; i < len; ++i){
        if (array1[i] != array2[i]){
            is_right = false;
            printf("wrong %d, %d\n", array1[i], array2[2]);
            break;
        }
    }
    if (is_right){
        printf("Answer is Correct\n");
    }else{
        printf("Answer is Wrong\n");
        for (int i = 0; i< len; ++i){
            printf("%d ", array1[i]);
        }
        printf("\n");
        for (int i = 0; i< len; ++i){
            printf("%d ", array2[i]);
        }
        printf("\n");
    }
}

void insert_into_dist(int source, int *frontier, int *frontier_size){
    frontier[(*frontier_size)++] = source;
}

void BFS_sequential(const int &source, const int *row_ptr, const int *dest, int *dist){
    int frontier[2][MAX_VERTEX];
    int *pre_froniter = &frontier[0][0];
    int *cur_frontier = &frontier[1][0];
    int pre_size = 0;
    int cur_size = 0;
    // 初始化配置
    insert_into_dist(source, pre_froniter, &pre_size);
    dist[source] = 0;
    while (pre_size > 0){
        // 遍历所有存储的节点
        for (int i = 0; i < pre_size; ++i){
            int cur_vertex = pre_froniter[i];
            // 遍历当前节点中的所有分支
           for (int j = row_ptr[cur_vertex]; j < row_ptr[cur_vertex+1]; ++j){
                if (dist[dest[j]] == -1){
                    insert_into_dist(dest[j], cur_frontier, &cur_size);
                    dist[dest[j]] = dist[cur_vertex] + 1;
                }
            }
        }
        // cur赋值给pre，重置cur
        std::swap(pre_froniter, cur_frontier);
        pre_size = cur_size;
        cur_size = 0;
    }
}

__global__ void BFS_Bqueue_kernel( int *pre_frontier,  int *pre_size,  int *cur_frontier,
                                   int *cur_size,  int *dest,  int *dist, int *visited){
    // 3级队列缓存优化
    // shared memory 分别存level 3的cur_frontier，对应的大小，level 2的cur_frontier，合并时对应的idx
    __shared__ int sub_queue_sd[BLOCK_QUEUE_NUM][BLOCK_QUEUE_SIZE];
    __shared__  int sub_queue_size[BLOCK_QUEUE_NUM];
    __shared__  int block_queue[BLOCK_QUEUE_NUM * BLOCK_QUEUE_SIZE];
    __shared__  int block_queue_insert_idx;
    __shared__  int sub_queue_total_size;
    const  int tx = threadIdx.x;
    const  int tid = tx + blockDim.x * blockIdx.x;
    const  int queue_idx = tx % BLOCK_QUEUE_NUM;
    if (tx < BLOCK_QUEUE_NUM){
        sub_queue_size[tx] = 0;
        if (tx == 0){
            sub_queue_total_size = 0;
        }
    }
    __syncthreads();
    // 开始遍历
    if (tid < *pre_size){
        const  int cur_vertex = pre_frontier[tid];
        for( int i = tex1D(row_ptr_dev, cur_vertex); i < tex1D(row_ptr_dev, cur_vertex + 1); ++i){
            const  int was_visited = atomicExch(&visited[dest[i]], 1);
            if (!was_visited){
                dist[dest[i]] = dist[cur_vertex] + 1;
                const  int cur_sub_size = atomicAdd(&sub_queue_size[queue_idx], 1);
                if (cur_sub_size < BLOCK_QUEUE_SIZE){
                    sub_queue_sd[queue_idx][cur_sub_size] = dest[i];
                }else{
                    // overflow 直接放入global memory中
                    sub_queue_size[queue_idx] = BLOCK_QUEUE_SIZE;
                    const  int global_idx = atomicAdd(cur_size, 1);
                    cur_frontier[global_idx] = dest[i];
                }
            }
        }
    }
    __syncthreads();
    // 开始执行合并操作
    // level 3 ===> level 2 cur_frontier
    for ( int i = 0; i < BLOCK_QUEUE_NUM; ++i){
        for ( int idx = tx; idx < sub_queue_size[i]; idx += blockDim.x){
            block_queue[idx + i * sub_queue_size[i]] = sub_queue_sd[i][idx];
        }
    }
    // level 3 ===> level 2 cur_size
    for ( int i = tx; i < BLOCK_QUEUE_NUM; i += blockDim.x){
        atomicAdd(&sub_queue_total_size, sub_queue_size[i]);
    }
    __syncthreads();

    // level 2 ===> level 1 cur_frontier，cur_size
    if (tx ==0){
        block_queue_insert_idx = atomicAdd(cur_size, sub_queue_total_size);
    }
    __syncthreads();
    for ( int i = tx; i < sub_queue_total_size; i += blockDim.x){
        cur_frontier[block_queue_insert_idx + i] = block_queue[i];
    }
}

void BFS_Bqueue(const int &source,  int *dest,  int *row_ptr,  int *dist,  int edge_num){
    // 初始化host
    int frontier[2][MAX_VERTEX];
    int *pre_froniter = &frontier[0][0];
    int *cur_frontier = &frontier[1][0];
    int visited[MAX_VERTEX] = {0};
    int pre_size = 1;
    int cur_size = 0;

    int *dist_output = new  int[MAX_VERTEX];
    for (int i = 0; i < MAX_VERTEX; ++i){
        dist[i] = -1;
        dist_output[i] = -1;
    }

    pre_froniter[0] = source;
    visited[source] = 1;
    dist[source] = 0;

    // 初始化 dev
    int *cur_size_dev, *pre_size_dev;
    int *dest_dev, *dist_dev, *visited_dev;
    int *cur_frontier_dev, *pre_frontier_dev;

    cudaEvent_t start, end;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&end));

    HANDLE_ERROR(cudaMalloc((void**)&cur_size_dev, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&pre_size_dev, sizeof(int)));
    HANDLE_ERROR(cudaMalloc((void**)&dest_dev, sizeof(int) * edge_num));
    HANDLE_ERROR(cudaMalloc((void**)&dist_dev, sizeof(int) * MAX_VERTEX));
    HANDLE_ERROR(cudaMalloc((void**)&cur_frontier_dev, sizeof(int) * (MAX_VERTEX)));
    HANDLE_ERROR(cudaMalloc((void**)&pre_frontier_dev, sizeof(int) * (MAX_VERTEX)));
    HANDLE_ERROR(cudaMalloc((void**)&visited_dev, sizeof(int) * MAX_VERTEX));
    HANDLE_ERROR(cudaMemcpy(cur_size_dev, &cur_size, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pre_size_dev, &pre_size, sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dest_dev, dest, sizeof(int) * edge_num, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dist_dev, dist, sizeof(int) * MAX_VERTEX, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(cur_frontier_dev, cur_frontier, sizeof(int) * (MAX_VERTEX), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(pre_frontier_dev, pre_froniter, sizeof(int) * (MAX_VERTEX), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(visited_dev, visited, sizeof(int) * (MAX_VERTEX), cudaMemcpyHostToDevice));

    cudaArray *t_array = 0;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
    HANDLE_ERROR(cudaMallocArray(&t_array, &desc, (MAX_VERTEX + 1)));
    HANDLE_ERROR(cudaMemcpyToArray(t_array, 0, 0, row_ptr, sizeof(int) * (MAX_VERTEX + 1), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaBindTextureToArray(row_ptr_dev, t_array));

    HANDLE_ERROR(cudaEventRecord(start, 0));
    while (pre_size > 0){
        // 求解出cur_frontier
        int BLOCK_NUM = (pre_size - 1) / BLOCK_SIZE + 1;
        BFS_Bqueue_kernel<<<BLOCK_NUM, BLOCK_SIZE>>>(pre_frontier_dev, pre_size_dev, cur_frontier_dev, cur_size_dev, dest_dev, dist_dev, visited_dev);
        // dev ===> host
        std::swap(pre_frontier_dev, cur_frontier_dev);
        HANDLE_ERROR(cudaMemcpy(pre_size_dev, cur_size_dev, sizeof(int), cudaMemcpyDeviceToDevice));
        HANDLE_ERROR(cudaMemset(cur_size_dev, 0, sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(&pre_size, pre_size_dev, sizeof(int), cudaMemcpyDeviceToHost));
    }
    // Kernel launch
    HANDLE_ERROR(cudaEventRecord(end, 0));
    HANDLE_ERROR(cudaEventSynchronize(end));
    float elapsed_time;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start, end));
    printf("Elapsed Time is %f \n",elapsed_time);

    // 验证结果
    HANDLE_ERROR(cudaMemcpy(dist, dist_dev, sizeof(int) * MAX_VERTEX, cudaMemcpyDeviceToHost));
    BFS_sequential(SOURCE_VERTEX, row_ptr, dest, dist_output);
    verify_output(dist_output, dist, MAX_VERTEX);

    // destroy
    delete []dist_output;
    HANDLE_ERROR(cudaUnbindTexture(row_ptr_dev));
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaFree(dest_dev));
    HANDLE_ERROR(cudaFree(dist_dev));
}

int main(int args, char** argv){

    // 先打印一下graph 验证一下是否正确
    Graph graph = Graph(MAX_VERTEX, 0.2);
    graph.show(0);
    graph.show(1);

    // 初始化 host
     int *dest = graph.get_dest();
     int *row_ptr = graph.get_row_ptr();
     int *dist = new  int[MAX_VERTEX];

    BFS_Bqueue(SOURCE_VERTEX, dest, row_ptr, dist, graph.get_edge_num());

    delete []dist;
    return 0;

}