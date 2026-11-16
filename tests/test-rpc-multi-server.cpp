#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

// copies src to dst n_steps times with new values each step, and checks dst after each copy
static void copy_steps(ggml_backend_t backend_src, ggml_backend_t backend_dst, ggml_tensor * src, ggml_tensor * dst,
                       float sign, int n_steps) {
    for (int step = 0; step < n_steps; step++) {
        float data[4];
        for (int i = 0; i < 4; i++) {
            data[i] = sign * (float) (4*step + i + 1);
        }
        ggml_backend_tensor_set(src, data, 0, sizeof(data));
        ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst);
        float got[4];
        ggml_backend_tensor_get(dst, got, 0, sizeof(got));
        for (int i = 0; i < 4; i++) {
            GGML_ASSERT(got[i] == data[i]);
        }
    }
}

static bool file_exists(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    fclose(f);
    return true;
}

static void touch(const std::string & path) {
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f != nullptr);
    fclose(f);
}

// the test script creates path within 60 s
static void wait_for_file(const std::string & path) {
    for (int i = 0; i < 6000 && !file_exists(path); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    GGML_ASSERT(file_exists(path));
}

// Opens the link from server A to server B, then posts a copy from A to B larger than the default socket buffers while the script has A stopped.
// The script kills B before it resumes A, so A sends the copy to a closed link and the send fails however late B's reset comes.
static void copy_after_peer_exit(const char * endpoint_a, const char * endpoint_b, const std::string & dir) {
    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);
    const int64_t n_copy = 4*1024*1024;
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx_a = ggml_init(params);
    ggml_context * ctx_b = ggml_init(params);
    GGML_ASSERT(ctx_a != nullptr);
    GGML_ASSERT(ctx_b != nullptr);
    ggml_tensor * src = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, n_copy);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx_b, GGML_TYPE_F32, n_copy);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors(ctx_a, backend_a) != nullptr);
    GGML_ASSERT(ggml_backend_alloc_ctx_tensors(ctx_b, backend_b) != nullptr);
    // the first copy opens the link
    copy_steps(backend_a, backend_b, src, dst, 1.0f, 1);
    touch(dir + "/linked");

    wait_for_file(dir + "/stopped");
    ggml_backend_tensor_copy_async(backend_a, backend_b, src, dst);
    // these wait only until the commands are written, which the socket of a stopped server still takes
    ggml_backend_synchronize(backend_a);
    ggml_backend_synchronize(backend_b);
    touch(dir + "/posted");
    wait_for_file(dir + "/go");
    // nothing is freed: with B gone, a command to B aborts the client, and exiting closes both connections
}

int main(int argc, char ** argv) {
    const bool peer_exit = argc == 5 && strcmp(argv[3], "--peer-exit") == 0;
    GGML_ASSERT(argc == 3 || peer_exit);
    ggml_backend_load_all();

    const char * endpoint_a = argv[1];
    const char * endpoint_b = argv[2];
    if (peer_exit) {
        copy_after_peer_exit(endpoint_a, endpoint_b, argv[4]);
        return 0;
    }

    ggml_backend_t backend_a = ggml_backend_rpc_init(endpoint_a, 0);
    ggml_backend_t backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_a != nullptr);
    GGML_ASSERT(backend_b != nullptr);

    ggml_init_params params = {
        /* .mem_size   = */ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(1, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_a);
    GGML_ASSERT(buffer != nullptr);

    // A remote pointer allocated by server A is not meaningful to server B.
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1, false);
    graph->nodes[0] = tensor;
    graph->n_nodes = 1;

    GGML_ASSERT(ggml_backend_graph_compute(backend_b, graph) == GGML_STATUS_SUCCESS);
    // Wait for server B to finish the graph before the script checks its log.
    size_t free_mem;
    size_t total_mem;
    ggml_backend_rpc_get_device_memory(endpoint_b, 0, &free_mem, &total_mem);
    GGML_ASSERT(total_mem > 0);
    ggml_backend_buffer_free(buffer);

    // Two tensors with the same ne[] but different nb[] must not share a cached alloc size.
    // ref: https://github.com/ggml-org/llama.cpp/issues/28360
    ggml_backend_buffer_type_t buft = ggml_backend_rpc_buffer_type(endpoint_a, 0);
    GGML_ASSERT(buft != nullptr);

    // MUL_MAT may need extra memory, so the size is read from the server [TAG_ALLOC_SIZE_EXPAND]
    ggml_tensor * packed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    packed->op = GGML_OP_MUL_MAT;

    // same ne[], twice the row stride
    ggml_tensor * strided = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    strided->op = GGML_OP_MUL_MAT;
    strided->nb[1] = 2*strided->nb[1];
    strided->nb[2] = strided->ne[1]*strided->nb[1];
    strided->nb[3] = strided->nb[2];
    GGML_ASSERT(ggml_nbytes(strided) > ggml_nbytes(packed));

    // ask for the packed tensor first, so a cache keyed without nb[] holds the smaller size
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, packed)  >= ggml_nbytes(packed));
    GGML_ASSERT(ggml_backend_buft_get_alloc_size(buft, strided) >= ggml_nbytes(strided));

    // Copies from server A to server B and back are posted between the graphs without waiting for any of them, as the scheduler does for consecutive pipeline stages.
    {
        ggml_init_params params_link = {
            /* .mem_size   = */ 3*ggml_tensor_overhead() + ggml_graph_overhead_custom(2, false),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx_a = ggml_init(params_link);
        ggml_context * ctx_b = ggml_init(params_link);
        GGML_ASSERT(ctx_a != nullptr);
        GGML_ASSERT(ctx_b != nullptr);

        ggml_tensor * x   = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, 4);
        ggml_tensor * y_a = ggml_scale(ctx_a, x, 2.0f);
        ggml_tensor * w   = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, 4);
        ggml_tensor * y_b = ggml_new_tensor_1d(ctx_b, GGML_TYPE_F32, 4);
        ggml_tensor * z   = ggml_scale(ctx_b, y_b, 3.0f);
        ggml_backend_buffer_t buffer_a = ggml_backend_alloc_ctx_tensors(ctx_a, backend_a);
        ggml_backend_buffer_t buffer_b = ggml_backend_alloc_ctx_tensors(ctx_b, backend_b);
        GGML_ASSERT(buffer_a != nullptr);
        GGML_ASSERT(buffer_b != nullptr);

        ggml_cgraph * graph_a = ggml_new_graph_custom(ctx_a, 2, false);
        ggml_build_forward_expand(graph_a, y_a);
        ggml_cgraph * graph_b = ggml_new_graph_custom(ctx_b, 2, false);
        ggml_build_forward_expand(graph_b, z);

        for (int step = 0; step < 2; step++) {
            // new values each step, so a copy that did not arrive leaves a wrong result
            float x_data[4];
            for (int i = 0; i < 4; i++) {
                x_data[i] = (float) (4*step + i + 1);
            }
            ggml_backend_tensor_set(x, x_data, 0, sizeof(x_data));
            GGML_ASSERT(ggml_backend_graph_compute_async(backend_a, graph_a) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_copy_async(backend_a, backend_b, y_a, y_b);
            GGML_ASSERT(ggml_backend_graph_compute_async(backend_b, graph_b) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_copy_async(backend_b, backend_a, z, w);

            float w_data[4];
            ggml_backend_tensor_get(w, w_data, 0, sizeof(w_data));
            for (int i = 0; i < 4; i++) {
                GGML_ASSERT(w_data[i] == 6.0f*x_data[i]);
            }
        }

        ggml_backend_buffer_free(buffer_b);
        ggml_backend_buffer_free(buffer_a);
        ggml_free(ctx_b);
        ggml_free(ctx_a);
    }

    // Two threads post copies at once, A to B and B to A, with backends of their own that share each server's queue.
    // The two halves of each copy must reach both queues in the same order, or the servers wait for each other and this step hangs.
    {
        const char * endpoints[2][2] = { { endpoint_a, endpoint_b }, { endpoint_b, endpoint_a } };
        ggml_backend_t        backends[2][2];
        ggml_context *        ctxs[2][2];
        ggml_tensor *         tensors[2][2];
        ggml_backend_buffer_t buffers[2][2];
        for (int t = 0; t < 2; t++) {
            for (int s = 0; s < 2; s++) {
                ggml_init_params params_copy = {
                    /* .mem_size   = */ ggml_tensor_overhead(),
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true,
                };
                backends[t][s] = ggml_backend_rpc_init(endpoints[t][s], 0);
                ctxs[t][s] = ggml_init(params_copy);
                GGML_ASSERT(backends[t][s] != nullptr);
                GGML_ASSERT(ctxs[t][s] != nullptr);
                tensors[t][s] = ggml_new_tensor_1d(ctxs[t][s], GGML_TYPE_F32, 4);
                buffers[t][s] = ggml_backend_alloc_ctx_tensors(ctxs[t][s], backends[t][s]);
                GGML_ASSERT(buffers[t][s] != nullptr);
            }
        }
        std::thread threads[2];
        for (int t = 0; t < 2; t++) {
            threads[t] = std::thread(copy_steps, backends[t][0], backends[t][1], tensors[t][0], tensors[t][1],
                                     t == 0 ? 1.0f : -1.0f, 1000);
        }
        for (int t = 0; t < 2; t++) {
            threads[t].join();
        }
        for (int t = 0; t < 2; t++) {
            for (int s = 0; s < 2; s++) {
                ggml_backend_buffer_free(buffers[t][s]);
                ggml_free(ctxs[t][s]);
                ggml_backend_free(backends[t][s]);
            }
        }
    }

    // One thread all-reduces a tensor across A and B while another copies from A to B, and each makes one server wait for the other.
    // Their commands must reach both queues in the same order, or the servers wait for each other and this step hangs.
    {
        ggml_backend_reg_t reg = ggml_backend_rpc_reg();
        auto comm_init      = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
        auto comm_free      = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
        auto comm_allreduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
        GGML_ASSERT(comm_init != nullptr);
        GGML_ASSERT(comm_free != nullptr);
        GGML_ASSERT(comm_allreduce != nullptr);

        // [0] all-reduces, [1] copies
        const char * endpoints[2] = { endpoint_a, endpoint_b };
        ggml_backend_t        backends[2][2];
        ggml_context *        ctxs[2][2];
        ggml_tensor *         tensors[2][2];
        ggml_backend_buffer_t buffers[2][2];
        for (int t = 0; t < 2; t++) {
            for (int s = 0; s < 2; s++) {
                ggml_init_params params_comm = {
                    /* .mem_size   = */ ggml_tensor_overhead(),
                    /* .mem_buffer = */ nullptr,
                    /* .no_alloc   = */ true,
                };
                backends[t][s] = ggml_backend_rpc_init(endpoints[s], 0);
                ctxs[t][s] = ggml_init(params_comm);
                GGML_ASSERT(backends[t][s] != nullptr);
                GGML_ASSERT(ctxs[t][s] != nullptr);
                tensors[t][s] = ggml_new_tensor_1d(ctxs[t][s], GGML_TYPE_F32, 4);
                // the all-reduce takes only tensors marked as computed
                if (t == 0) {
                    tensors[t][s]->flags |= GGML_TENSOR_FLAG_COMPUTE;
                }
                buffers[t][s] = ggml_backend_alloc_ctx_tensors(ctxs[t][s], backends[t][s]);
                GGML_ASSERT(buffers[t][s] != nullptr);
            }
        }
        void * comm = comm_init(backends[0], 2);
        GGML_ASSERT(comm != nullptr);
        std::thread copier(copy_steps, backends[1][0], backends[1][1], tensors[1][0], tensors[1][1], 1.0f, 1000);
        for (int step = 0; step < 1000; step++) {
            const float a[4] = { (float) step, 1.0f, 2.0f, 3.0f };
            const float b[4] = { 1.0f, (float) step, 4.0f, 5.0f };
            ggml_backend_tensor_set(tensors[0][0], a, 0, sizeof(a));
            ggml_backend_tensor_set(tensors[0][1], b, 0, sizeof(b));
            GGML_ASSERT(comm_allreduce(comm, tensors[0]));
            for (int s = 0; s < 2; s++) {
                float got[4];
                ggml_backend_tensor_get(tensors[0][s], got, 0, sizeof(got));
                for (int i = 0; i < 4; i++) {
                    GGML_ASSERT(got[i] == a[i] + b[i]);
                }
            }
        }
        copier.join();
        comm_free(comm);
        for (int t = 0; t < 2; t++) {
            for (int s = 0; s < 2; s++) {
                ggml_backend_buffer_free(buffers[t][s]);
                ggml_free(ctxs[t][s]);
                ggml_backend_free(backends[t][s]);
            }
        }
    }

    // B's session ends while A's goes on, and the next copy from A to B opens the link again under its old id, so A replaces the dead connection
    ggml_backend_free(backend_b);
    backend_b = ggml_backend_rpc_init(endpoint_b, 0);
    GGML_ASSERT(backend_b != nullptr);
    {
        ggml_init_params params_relink = {
            /* .mem_size   = */ ggml_tensor_overhead(),
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx_a = ggml_init(params_relink);
        ggml_context * ctx_b = ggml_init(params_relink);
        GGML_ASSERT(ctx_a != nullptr);
        GGML_ASSERT(ctx_b != nullptr);
        ggml_tensor * src = ggml_new_tensor_1d(ctx_a, GGML_TYPE_F32, 4);
        ggml_tensor * dst = ggml_new_tensor_1d(ctx_b, GGML_TYPE_F32, 4);
        ggml_backend_buffer_t buffer_a = ggml_backend_alloc_ctx_tensors(ctx_a, backend_a);
        ggml_backend_buffer_t buffer_b = ggml_backend_alloc_ctx_tensors(ctx_b, backend_b);
        GGML_ASSERT(buffer_a != nullptr);
        GGML_ASSERT(buffer_b != nullptr);
        copy_steps(backend_a, backend_b, src, dst, 1.0f, 3);
        ggml_backend_buffer_free(buffer_b);
        ggml_backend_buffer_free(buffer_a);
        ggml_free(ctx_b);
        ggml_free(ctx_a);
    }

    ggml_free(ctx);
    ggml_backend_free(backend_b);
    ggml_backend_free(backend_a);
    return 0;
}
