/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/kernels/ban_bad_words.h"
#include "src/fastertransformer/utils/cuda_utils.h"

namespace fastertransformer {

// CPU_C
#include <limits.h>
#include <stdbool.h>


// output_ids_buf 通常是一个三维数组，用于表示多个批次、束和时间步。
// 形状可以是 [batch_size, beam_width，max_sequence_length]
// ==> eg.
/*
    [
        [1, 2, 3],  // batch 0, beam 0
        [1, 2, 4],  // batch 0, beam 1
        [1, 2, 5],  // batch 0, beam 2
        [0, 1, 2],  // batch 1, beam 0
        [0, 1, 3],  // batch 1, beam 1
        [0, 1, 4]   // batch 1, beam 2
    ]
*/
// 词的观念
// eg. [1, 2, 3],  // batch 0, beam 0
// 表示 束搜索中表示一个特定束（beam）在某个时间步生成的标记序列
/*
    1、2 和 3 是词汇表中某些单词或子词的索引，表示生成的序列的内容
        
    假设词汇表如下：
    Index	Word
    0	"我"
    1	"爱"
    2	"编程"
    3	"学习"
    
    [1, 2, 3] 表示生成的序列为 "爱 编程 学习"
*/

// cpu 的实现
void ban_bad_words_cpu(float* logits,
                       const int* output_ids_buf,
                       const int* parent_ids_buf,
                       int batch_size,
                       int beam_width,
                       const int* bad_words,
                       size_t bad_words_len,
                       bool share_words,
                       int id_offset,
                       int vocab_size_padded,
                       size_t step) {
    // 遍历每个批次和束
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (int beam_idx = 0; beam_idx < beam_width; ++beam_idx) {

            // 计算当前的不良单词基准
            const int* base_bad_words = share_words ? bad_words : bad_words + batch_idx * 2 * bad_words_len;
            const int* base_bad_words_offsets = base_bad_words + bad_words_len;

            for (size_t id = 0; id < bad_words_len; ++id) {
                if (base_bad_words_offsets[id] < 0) {     // JUMP -1 
                    continue;  // 跳过无效的偏移
                }

                const int item_end = base_bad_words_offsets[id];
                const int item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
                const int item_size = item_end - item_start;

                // 单一标记的情况，直接禁止
                bool should_ban = (item_size == 1);
                
                // 多标记： 指的是由两个或多个单词组成的短语或序列
                // 多标记情况，检查是否足够生成的标记
                if (item_size > 1 && step >= item_size - 1) {   // step 检查当前生成的序列长度是否足够
                    should_ban = true;
                    int parent_id = beam_idx;
                    
                    // Bad_word 的长度 来限制了向前回溯的长度 
                    for (int token_idx = item_size - 2; token_idx >= 0; --token_idx) {
                        // 从 output_ids_buf 中提取先前生成的标记（token）
                        // because beam_search 是 树形结构
                        // 树形结构 向前找 父亲节点

                        // 计算当前要检查的标记的索引
                        const int previous_token = output_ids_buf[(step - (item_size - 1) + token_idx) * batch_size * beam_width
                                                                  + id_offset + batch_idx * beam_width + parent_id];
                        /*
                            计算步骤
                                step - (item_size - 1) + token_idx：
                                    step：当前生成的步骤
                                    item_size：当前不良标记的长度
                                    token_idx：当前要检查的标记索引
                                    这部分计算出当前要访问的时间步

                                * (batch_size * beam_width)：
                                    这部分用于计算在 output_ids_buf 中的行偏移量。每一步有 batch_size * beam_width 个元素
                                
                                id_offset：
                                    用于调整当前的索引，在不同的上下文或模型中可能会有所不同
                                
                                batch_idx * beam_width：
                                    这部分用于定位到具体的批次（batch）在当前时间步的起始位置
                                
                                + parent_id：
                                    指定当前束的索引，用于访问该束的具体生成标记
                        */


                        // 检查 previous_token 是否与不良标记匹配
                        if (previous_token != base_bad_words[item_start + token_idx]) {
                            should_ban = false;
                            break;
                        }  
                        // 更新 parent_id，向上查找 父节点
                        // !! 有两个 _buf 来实现不同功能
                        // parent_ids_buf：存储每个标记的来源信息，用于追踪生成过程中的路径。
                        parent_id = parent_ids_buf[(step - (item_size - 1) + token_idx) * beam_width * batch_size + id_offset
                                                    + batch_idx * beam_width + parent_id];

                        if (parent_id < 0 || parent_id >= beam_width) {
                            should_ban = false;
                            break;
                        }
                    }

                    // 使用 output_ids_buf 访问生成的标记
                    // 使用 parent_ids_buf 更新父节点

                // 如果应该禁止，则设置 logits
                if (should_ban) {
                    int banned_token = base_bad_words[item_end - 1];
                    if (0 < banned_token && banned_token < vocab_size_padded) {
                        logits[batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token] =
                            -INFINITY;  // 使用负无穷表示禁止
                    }
                }
            }
        }
    }
}


template<typename T>
__global__ void ban_bad_words(T*         logits,
                              const int* output_ids_buf,    // 已生成的输出序列的 ID 缓冲区
                              const int* parent_ids_buf,    // 束搜索中每个输出 ID 的父 ID 缓冲区
                              int        batch_size,
                              int        beam_width,
                              const int* bad_words,
                              size_t     bad_words_len,
                              bool       share_words,        // 指示是否共享不良单词的标志
                              int        id_offset,          // 定位当前批次的偏移量
                              int        vocab_size_padded,  // 填充后的词汇表大小
                              size_t     step)
{
    const int id        = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y / beam_width;
    const int beam_idx  = blockIdx.y % beam_width;

    const int* base_bad_words         = share_words ? bad_words : bad_words + batch_idx * 2 * bad_words_len;
    const int* base_bad_words_offsets = base_bad_words + bad_words_len;

    if (id >= bad_words_len || base_bad_words_offsets[id] < 0) {
        return;
    }

    const int item_end   = base_bad_words_offsets[id];
    const int item_start = (id > 0) ? base_bad_words_offsets[id - 1] : 0;
    const int item_size  = item_end - item_start;
    // bad_words_len 表示有多少个 bad_words
    // base_bad_words_offsets 用于标记 不同 bad_words 的 结束 位置
    // 每个 bad_words 词长度为 base_bad_words_offsets[id] - base_bad_words_offsets[id-1]
    //     第一个位置特殊计算：   base_bad_words_offsets[0] - 0

    /* The single-token case unconditionally bans the token */
    // 由于这个词为一个单词，并被明确标记为不良词，直接禁止它
    bool should_ban = item_size == 1;

    /* Multi-token case and enough previously generated tokens to look for a match */
    // 在 beam_width 为 1 的情况下，将 bad_words_len （bad_words 的个数）进行切分，并行
    if (item_size > 1 && step >= item_size - 1) {
        should_ban             = true;
        int        parent_id   = beam_idx;
        const bool gather_beam = beam_width > 1;

        for (int token_idx = item_size - 2; token_idx >= 0; token_idx--) {
            const int previous_token = output_ids_buf[(step - (item_size - 1) + token_idx) * batch_size * beam_width
                                                      + id_offset + batch_idx * beam_width + parent_id];

            if (previous_token != base_bad_words[item_start + token_idx]) {
                should_ban = false;
                break;
            }
            if (gather_beam) {
                parent_id = parent_ids_buf[(step - (item_size - 1) + token_idx) * beam_width * batch_size + id_offset
                                           + batch_idx * beam_width + parent_id];

                if (parent_id < 0 || parent_id >= beam_width) {
                    should_ban = false;
                    break;
                }
            }
        }
    }

    // 疑惑： 不会有不同线程的都匹配 bad_words 吗 ???
    // 应该不会 bad_words 应该没有重叠（本来就只能匹配到一个 
    if (should_ban) {
        int banned_token = base_bad_words[item_end - 1];
        if (0 < banned_token && banned_token < vocab_size_padded) {
            logits[batch_idx * beam_width * vocab_size_padded + beam_idx * vocab_size_padded + banned_token] =
                static_cast<T>(-INFINITY);
        }
    }
}

template<typename T>
void invokeBanBadWords(T*           logits,
                       const int*   output_ids_buf,
                       const int*   parent_ids_buf,
                       int          batch_size,
                       int          local_batch_size,
                       int          beam_width,
                       const int*   bad_words,
                       bool         share_words,
                       size_t       bad_words_len,
                       int          id_offset,
                       int          vocab_size_padded,
                       size_t       step,
                       cudaStream_t stream)
{
    dim3 block, grid;
    block.x = min(((bad_words_len + 32 - 1) / 32) * 32, 256UL);
    grid.x  = (bad_words_len + block.x - 1) / block.x;
    grid.y  = local_batch_size * beam_width;

    ban_bad_words<<<grid, block, 0, stream>>>(logits,
                                              output_ids_buf,
                                              parent_ids_buf,
                                              batch_size,
                                              beam_width,
                                              bad_words,
                                              bad_words_len,
                                              share_words,
                                              id_offset,
                                              vocab_size_padded,
                                              step);
    sync_check_cuda_error();
}

template void invokeBanBadWords(half*        logits,
                                const int*   output_ids_buf,
                                const int*   parent_ids_buf,
                                int          batch_size,
                                int          local_batch_size,
                                int          beam_width,
                                const int*   bad_words,
                                bool         share_words,
                                size_t       bad_words_len,
                                int          id_offset,
                                int          vocab_size_padded,
                                size_t       step,
                                cudaStream_t stream);
#ifdef ENABLE_BF16
template void invokeBanBadWords(__nv_bfloat16* logits,
                                const int*     output_ids_buf,
                                const int*     parent_ids_buf,
                                int            batch_size,
                                int            local_batch_size,
                                int            beam_width,
                                const int*     bad_words,
                                bool           share_words,
                                size_t         bad_words_len,
                                int            id_offset,
                                int            vocab_size_padded,
                                size_t         step,
                                cudaStream_t   stream);
#endif
template void invokeBanBadWords(float*       logits,
                                const int*   output_ids_buf,
                                const int*   parent_ids_buf,
                                int          batch_size,
                                int          local_batch_size,
                                int          beam_width,
                                const int*   bad_words,
                                bool         share_words,
                                size_t       bad_words_len,
                                int          id_offset,
                                int          vocab_size_padded,
                                size_t       step,
                                cudaStream_t stream);

}  // namespace fastertransformer
