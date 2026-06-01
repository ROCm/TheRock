# Fused-MoE with CK Tile

This example implements a highly optimized fused Mixture-of-Experts (MoE) block using the CK Tile programming model. The design fuses MoE sorting, group-GEMM, activation, and top-k weighting into a single kernel, minimizing memory traffic and maximizing throughput for large language models.

---

## Algorithm and Math

### MoE Block Structure

Given:
- **Input**: $X$ of shape $[\text{tokens}, \text{hidden}]$
- **TopK indices/weights**: $I, W$ from gating (shape $[\text{tokens}, \text{topk}]$)
- **Expert weights**: $[\text{experts}, \text{hidden}, \text{hidden}]$

**Steps:**
1. **MoE Sorting**: Rearrange tokens so each expert receives its assigned tokens in contiguous blocks (see [13_moe_sorting](../13_moe_sorting/README.md)).
2. **Group-GEMM**: For each expert, perform GEMM on its assigned tokens:
   $$
   Y^{(e)} = X^{(e)} W^{(e)}
   $$
3. **Activation + TopK Weighting**: Apply activation (e.g., GELU) and multiply by top-k weights.
4. **Scatter/Gather**: Write results back to the original token order.

### Technical Details

- **Scatter/Gather Group-GEMM**: Uses indirect indexing to map tokens to experts and back.
- **Block Partitioning**: Tokens are partitioned into slices per expert, with padding for alignment.
- **Atomic Accumulation**: Second GEMM uses atomics for accumulation to support overlapping tokens.
- **Buffer Zeroing**: Output buffer is zeroed in the sorting step, eliminating extra kernels.
- **Pre-shuffled Weights**: Expert weights are pre-shuffled for coalesced memory access.
- **Micro-kernel Pipeline**: Uses block-inline-asm micro-kernels for peak performance, while retaining composability.


## Build & Run

```bash
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>
make tile_example_fused_moe -j
./bin/tile_example_fused_moe -?
```

---

## Source Structure

- **Kernel**: [`fused_moe.hpp`](fused_moe.hpp), [`fused_moegemm.hpp`](fused_moegemm.hpp), [`fused_moesorting.hpp`](fused_moesorting.hpp)
- **Executable**: [`main.cpp`](main.cpp)
- **Build**: `CMakeLists.txt`, `instances/`, `misc/`

---

## Technical Notes

 This is a scatter/gather-group-gemm based solution, similiar to that of [vllm moe](https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py), but we introduce more kernel fusion to boost performance
![](misc/moe-0.png)

The benifit of this fused-moe:
* 1.5~2x perf boost compared with current vllm solution
* zero workspace to reduce memory footprint
* much less kernel instance, easy to maintain

# Implementation and feature support
## NOTES:
currently gate+up in fp16 case will very easily cause accumulator overflow the fp16 max(65504), hence result in INF. Please use BF16 for gate+up case, API side will have no check for this.

## moe-sorting
this is a common pre-process step before the actual moe-gemm. The purpose is to transform the moe loop over from token-by-token to expert-by-expert, make sure very workgroup is working for a single expert (B matrix). Besides, we extend this op to do the zeroing of the output buffer(to be used for reduce buffer with atomic)

## moe-gemm
`moe-gemm` is a group-gemm based back-to-back gemm, where the row-id of input token comes from another buffer. Naive understanding of fused-moe is from token-by-token view as below picture:
![](misc/moe-1.png)
After `moe-sorting`, we can view this algorithm as expert-by-expert, as below:
![](misc/moe-2.png)

## optimization
summary of the key design of this fused-moe operator:
* fuse 2 group-gemm + activation + `topk-weight` multiply into single kernel, using atomic for 2nd gemm accumualation
* fuse buffer-zeroing in `moe-sorgin`, user no longer need call extra torch.zero() for the out buffer
* fused scatter-gather for row index(same as vllm)
* pre-shuffle B matric(weight) to maximize memory throughput. input(activation) keep original layout `[batch, hidden]`.
* extrem optimized pipeline using block-inline-asm(we call it `micro-kernel` or `uk`), while not breaking the *composable* design of ck

## 
```
// [indexing implementation-1]
// using M_a as constexpr block_size to partition all tokens into different slices
// each slice map to one expert, and one expert can have multiple slices
// e.g. num_experts = 6, topk=3, M_a = 4, input_tokens = 5
// before sort, topk_ids is : [[0, 3, 5], [2, 3, 5], [1, 3, 5], [1, 2, 3], [1, 3, 5]]
//                            tok-0      tok-1      tok-2      tok-3      tok-4
//           topk_weight is : [[a, b, c], [d, e, f], [g, h, i], [j, k, l], [m, n, o]] (some float number)
//
// token_id_per_expert is : [[0], [2, 3, 4], [1, 3], [0, 1, 2, 3, 4], [], [0, 1, 2, 5]]
//  (only for reference)    exp-0  exp-1     exp-2   exp-3          exp-4  exp-5
// weight_id_per_expert is: [[a], [g, j, m], [d, k], [b, e, h, l, n], [], [c, f, i, o]]
//
// max_num_tokens_padded : topk * input_tokens + num_experts * M_a - topk (updated)
// * this could be larger than actual, since actual tokens are on GPU
//
// sorted_token_ids_ptr   : [0, 6, 6, 6, 2, 3, 4, 6, 1, 3, 6, 6, 0, 1, 2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 5]
//                          |-  exp-0  -|-  exp-1  -|-  exp-2  -|-      exp-3          -|-  exp-4 -|-  exp-5  -|
// sorted_weight_ptr      : [a, *, *, *, g, j, m, *, d, k, *, *, b, e, h, l, n, *, *, *, *, *, *, *, c, f, i, o]
//
// * length is max_num_tokens_padded, actual size is num_tokens_post_padded_ptr
//
// sorted_expert_ids_ptr  : [0, 1, 2, 3, 3, 4, 5]
// * length is (max_num_tokens_padded + block_size - 1) / block_size
//
// num_tokens_post_padded_ptr : [28]
// num_sorted_tiles_ptr : [7]
//
// * different from vLLM
//   1) token_id stored in sorted_token_ids_ptr is actual token_id, not token_id*top_K expanded id
//   2）need sorted_weight_ptr
//   3) use num_sorted_tiles_ptr, already divided by M_a
//
// * below used for indexing
//  1) sorted_token_ids_ptr [max_num_tokens_padded]
//  2) sorted_weight_ptr
//  3) sorted_expert_ids_ptr
//  4）num_tokens_post_padded_ptr/num_sorted_tiles_ptr (select one)
//
//   max_num_tokens_padded: opk_ids.numel() + num_experts * (block_size - 1)
```

## example
```
args:
          -t    number of input tokens. (default:128)
                If "local_t" presents, this value indicates global concurrency of all ranks.
    -local_t    Number of local input tokens for curent rank. (default:-1)
                This value must be within range "[0, t)", or "-1"(no such feature)
                This feature is to simulate EP case where where each rank has different tokens.
                Besides, this value will be stored in a GPU buffer, which is friendly for CUDA graph.
          -e    num of experts (default:32)
          -k    topk (default:5)
          -h    hidden_size of this model (default:8192)
          -i    intermediate_size between 2 gemms of FFN (default:8192)
     -stride    stride per row, if -1 then equal to hidden_size (default:-1)
         -bm    blocking factor for sorted tokens (default:32)
         -tp    tensor parallel size (default:8)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
     -prec_i    input precision (default:bf16)
     -prec_w    weight precision (default:bf16)
     -prec_o    output precision (default:bf16)
    -prec_st    token scale data type. auto will set to fp32 (default:auto)
    -prec_sw    weight scale data type. auto will set to fp32 (default:auto)
    -prec_sq    (dynamic) smooth quant data type. auto will set to fp32 (default:auto)
    -prec_kw    topk-weight data type. auto will set to fp32 (default:auto)
     -fquant    fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant (default:0)
  -gate_only    w0(gate/up) style, 0:gate+up will double interm size, 1:only gate (default:1)
        -api    benchmark api set: 0:fused-moe(moe-gemm+moe-sorting), 1:moe-gemm (default:0)
        -act    activation after first gemm. 0:gelu, 1:silu (default:0)
    -balance    if set to 1, will try balance the expert in topk-ids(convenient for testing) (default:0)
       -init    init method. 0:random stepped float(fast). 1: random uniform[-0.5, 0.5], 2:rand normalized[0, 1]normalized(slow) (default:1)
       -seed    seed used to do random (default:11939)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:fused_moe.json)
```
## Related CK Tile Examples

- [13_moe_sorting](../13_moe_sorting/README.md): MoE sorting for expert dispatch
- [09_topk_softmax](../09_topk_softmax/README.md): TopK-Softmax for MoE gating
- [03_gemm](../03_gemm/README.md): GEMM with tiles

For distribution, see [`include/ck_tile/tile_program/tile_distribution/`](../../../include/ck_tile/tile_program/tile_distribution/).

---
[Back to CK Tile Examples](../README.md)
