import torch
import triton
import triton.language as tl




## The sequence to write is this script: 
## class FlashAttention --> _attn_fwd --> _attn_fwd_inner
## --> _attn_bwd_preprocess --> _attn_bwd

## for float8, the V matrix is transposed, it is so unconvenient.


#(batch,seq_len,head,head_dim)

@triton.jit
def _attn_fwd_inner(O_i,l_i,m_i,q,
                    desc_k,desc_v,
                    qk_scale,
                    offset_y,
                    start_block,
                    dtype: tl.constexpr,
                    STAGE: tl.constexpr, 
                    HEAD_DIM: tl.constexpr,
                    BlOCK_R: tl.constexpr,
                    BlOCK_C: tl.constexpr,
                    SEQ_LEN: tl.constexpr,
                    warp_specialize: tl.constexpr,
                    offs_m: tl.constexpr, offs_n: tl.constexpr
    ):
    
    if STAGE== 1:
    # if causal attention, from 0 to the left of the diagonal     
           lo,hi=0,start_block*BlOCK_R
    elif STAGE== 2:
    #diagonal
        lo,hi=start_block*BlOCK_R,(start_block+1)*BlOCK_R
        lo=tl.multiple_of(lo,BlOCK_R)  ## faster memory load
    else:
        lo,hi=0,SEQ_LEN

    offsetk_y=offset_y+lo
    if dtype == tl.float8e5:
        # if the mateix is transposed, the next token will be calculated this way
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo


    #loop over k,v and update O_i(why called accumulator, acc)
    for start_kv in range(lo,hi,BlOCK_C,warp_specialize=warp_specialize):
        start_kv = tl.multiple_of(start_kv, BlOCK_C)  ## faster memory load

        k=desc_k.load([offsetk_y,0]).T
        qk=tl.dot(q,k)
        # update the maximum value for numerical stability
        if STAGE==2:
            mask=offs_m[:,None] >= (offs_n[None,:] + start_kv)
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)  # 
            m_ij=tl.maximum(m_i,tl.max(qk,1))
            qk-= m_ij[:,None]
        else:
            qk = qk * qk_scale
            m_ij=tl.maximum(m_i,tl.max(qk,1))
            qk-= m_ij[:,None]
        p_ij=tl.math.exp2(qk)  
        l_ij=tl.sum(p_ij,1)
        alpha=tl.math.exp2(m_i-m_ij)
        l_i=l_i*alpha+l_ij          #algorithm 1 line 9
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p_ij=p_ij.to(dtype)
        O_i= Q_i * alpha[:,None]
        O_i=tl.dot(p_ij,v,O_i)       #algorithm 1  line 10
        m_i=m_ij
        offsetk_y+=BlOCK_C
        offsetv_y+=BlOCK_C
    return O_i,l_i,m_i





@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    """
    for the newer GPU like Hopper,Blackwell, return the same tenosr
    for the older GPU like Ampere, return a tensor descriptor made by triton.
    """
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

#######
#  |
#  | 
#  |
#  |
#  |
#  _____|_____|_____|
#sequence length/block_size
@triton.jit
def _attn_fwd(sm_scale,
    BATCH_SIZE,NUM_HEADS,SEQ_LEN,HEAD_DIM,
    Q,K,V,
    M,
    O,
    FP8_OUTPUT: tl.constexpr,
    BLOCK_R: tl.constexpr,  ## change BLOCK_M to BLOCK_R, maintain its consistency with paper. There
    BLOCK_C: tl.constexpr,  ## column
    STAGE: tl.constexpr,  ## 1 for non-causal attention, 3 for causal attention, may i change the code?
    warp_specialize: tl.constexpr

):
    start_block = tl.program_id(0)
    index_batch_head = tl.program_id(1)  # in original code, it use off_hz.
    index_batch=index_batch_head // NUM_HEADS
    index_head=index_batch_head % NUM_HEADS 

    

    y_dim=BATCH_SIZE*NUM_HEADS*SEQ_LEN
    desc_q=_maybe_make_tensor_desc(Q,shape=[y_dim,HEAD_DIM],strides=[HEAD_DIM,1],
                                   block_shape=[BLOCK_R,HEAD_DIM] )
    if FP8_OUTPUT: 
        #if the output is in FP8, the second matrix is transposed.
        desc_v = _maybe_make_tensor_desc(V, shape=[HEAD_DIM, y_dim], strides=[SEQ_LEN, 1],
                                         block_shape=[HEAD_DIM, BLOCK_C])
    else:
        desc_v = _maybe_make_tensor_desc(V, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_C, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(K, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_C, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(O, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_R, HEAD_DIM])

    offset_y=index_batch*NUM_HEADS*SEQ_LEN + index_head*SEQ_LEN
    offset_block=offset_y + start_block*BLOCK_R

    offs_m=start_block*BLOCK_R +tl.arange(0,BLOCK_R)
    offs_n=tl.arange(0,BLOCK_C)

    ## initialize O_i,l_i,m_i, about the dtype, original code use float32, this may cause difference.
    O_i=tl.zeros((BLOCK_R,HEAD_DIM),dtype=tl.float16)
    l_i=tl.zeros((BLOCK_R,),dtype=tl.float16)
    m_i=tl.full((BLOCK_R,),float("-inf"),dtype=tl.float16)

    qk_scale=sm_scale 
    qk_scale *= 1.442   #1/log(2)
    q=desc_q.load([offset_block,0])

    if STAGE & 1: #1 or 3
        O_i,l_i,m_i=_attn_fwd_inner(O_i,l_i,m_i,q,
                    desc_k,desc_v,
                    qk_scale,
                    offset_y,
                    start_block,
                    dtype,
                    STAGE=4-STAGE,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_R=BLOCK_R,
                    BLOCK_C=BLOCK_C,
                    SEQ_LEN=SEQ_LEN,
                    warp_specialize=warp_specialize,
                    offs_m=offs_m,offs_n=offs_n
                    )
        
    if STAGE & 2: # 3
        O_i,l_i,m_i=_attn_fwd_inner(O_i,l_i,m_i,q,
                    desc_k,desc_v,
                    qk_scale,
                    offset_y,
                    start_block,
                    dtype,
                    STAGE=2,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_R=BLOCK_R,
                    BLOCK_C=BLOCK_C,
                    SEQ_LEN=SEQ_LEN,
                    warp_specialize=warp_specialize,
                    offs_m=offs_m,offs_n=offs_n)
    
    #epolpgue
    O_i=O_i/l_i[:,None]    #algorithm 1 ,line 12 , why it can directly use 1/l_i ?
    m_i+=tl.math.log2(l_i)
    l_ptr=M +offs_m + index_batch_head*SEQ_LEN
    tl.store(l_ptr,m_i)
    tl.store([offset_block,0],O_i.to(dtype))


@triton.jit
def _attn_bwd_preprocess(O,DO,
                         Delta,
                         SEQ_LEN: tl.constexpr,
                         BLOCK_R: tl.constexpr,
                         HEAD_DIM: tl.constexpr,):
    # algorithm 2 lines 1-2 & 4
    off_m=tl.program_id(0)*BLOCK_R + tl.arange(0,BLOCK_R)
    off_hz=tl.program_id(1)
    off_n=tl.arange(0,HEAD_DIM)
    
    o=tl.load(O+off_hz*SEQ_LEN*HEAD_DIM + off_m[:,None]*HEAD_DIM + off_n[None,:])
    do=tl.load(DO+off_hz*SEQ_LEN*HEAD_DIM + off_m[:,None]*HEAD_DIM + off_n[None,:])
    delta=tl.sum(o*do,1)
        
    tl.store(Delta+off_hz*SEQ_LEN + off_m,delta)


@triton.jit
def _attn_bwd_dkdv(dk,dv,M,D,
                    Q,k,v,DO,
                    stride_s,stride_d,
                    start_m,start_n,
                    HEAD_DIM: tl.constexpr,
                    BLOCK_M1 : tl.constexpr,## adjusted by BLK_SLICE_FACTOR
                    BLOCK_N1: tl.constexpr,
                    num_steps: tl.constexpr,
                    MASK: tl.constexpr

                   ):
    offs_m=start_m + tl.arange(0,BLOCK_M1)
    offs_n=start_n + tl.arange(0,BLOCK_N1)
    offs_k=tl.arange(0,HEAD_DIM)
    QT_ptrs=Q+offs_m[None,:]*stride_s + offs_k[:,None]*stride_d   ## Q^T
    do_ptrs=DO+offs_m[:,None]*stride_s + offs_k[None,:]*stride_d
    curr_m =start_m
    step_m =BLOCK_M1
    for blk_idx in range(num_steps):
        qT=tl.load(QT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m=curr_m + tl.arange(0, BLOCK_M1)
        m=tl.load(M + offs_m) ## m_i
        qkT=tl.dot(k,qT)  ## k Q^t = S^T
        PT=tl.math.exp2(qkT - m[None,:])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        ppT=pT
        ppT=ppT.to(tl.float16)

        ##algorithm 2 line 12
        dv+=tl.dot(ppT,do)
        ##algorithm 2 line 13
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)

        ##load Di for algorithm 2 line 14
        ##D(delta) is calculated in the preprocess kernel, algorithm 2 line 3
        Di=tl.load(D+offs_m)
        ##algorithm 2 line 14
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))

        curr_m += step_m
        QT_ptrs += step_m * stride_s
        do_ptrs += step_m * stride_s
    return dk,dv


@triton.jit
def _attn_bwd_dq(dq,q,do,m,V,K,D,start_m,start_n,num_steps,
                 stride_s,stride_d,
                 HEAD_DIM: tl.constexpr,
                 BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
                 MASK: tl.constexpr,
                 ):
    offs_m=start_m + tl.arange(0,BLOCK_M2)
    offs_n=start_n + tl.arange(0,BLOCK_N2)
    offs_k=tl.arange(0,HEAD_DIM)
    k_ptrs=K+offs_n[:,None]*stride_s + offs_k[None,:]*stride_d   ## K   different from its original code
    kT_ptrs=K+offs_m[None,:]*stride_s + offs_k[:,None]*stride_d   ## K^T
    vT_ptrs=V+offs_n[None,:]*stride_s + offs_k[:,None]*stride_d   ## V^T
    
    D_i=tl.load(D+offs_m) ## D_i

    curr_n= start_n
    step_n=BLOCK_N2


    ## whether it is faster in triton to save the k and k^t or transpose it ?
    ## if i save both k and k^t, it will consume more memory, but if i transpose it, it will cause more computation. I need to test it in the future.


    for blk_idx in range(num_steps):
        k=tl.load(k_ptrs)
        kT=tl.load(kT_ptrs)
        vT=tl.load(vT_ptrs)
        qk=tl.dot(q,kT)
        p=tl.math.exp2(qk-m)

        if MASK:
            offs_n= curr_n + tl.arange(0,BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p=tl.where(mask, p, 0.0)

        dp=tl.dot(do,vT).to(tl.float16)
        ds=p*(dp-D_i[:,None])
        ds=ds.to(tl.float16)
        dq+= tl.dot(ds,k)
        curr_n+=step_n
        k_ptrs+=step_n*stride_s
        kT_ptrs+=step_n*stride_s
        vT_ptrs+=step_n*stride_s
    return dq









@triton.jit
def _attn_bwd(Q,K,V,sm_scale,
            dO,# gradient of output
            DQ,DK,DV,
            M, # M :maximum
            D,# delta
            stride_b,stride_h,stride_s,stride_d,
            NUM_HEADS: tl.constexpr,
            SEQ_LEN: tl.constexpr,
            BATCH_SIZE: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr, BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
            NUM_WARPS: tl.constexpr, NUM_STAGES: tl.constexpr,
            CAUSAL: tl.constexpr,
            BLK_SLICE_FACTOR : tl.constexpr, ## 2 
            dtype,
            ):
    #grid= (SEQ_LEN // pre_block, 1 ,BATCH_SIZE*NUM_HEADS)
    LN2: tl.constexpr = 0.6931471824645996 

    start_block=tl.program_id(0)

    index_batch_head=tl.program_id(2)

    index_batch=index_batch_head // NUM_HEADS
    index_head=index_batch_head % NUM_HEADS
    adj= (index_batch*stride_b + index_head*stride_h).to(tl.int64)
    off_chz=(SEQ_LEN*index_batch_head).to(tl.int64)  # for M and D, descriptor
    off_n=tl.arange(0,HEAD_DIM).to(tl.int64)

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    #batch/head/seq
    M += off_chz
    D += off_chz

    offs_k=tl.arange(0,HEAD_DIM)
    ##N_1,M2 for K,V,dK,dV
    start_n=start_block*BLOCK_N1
    start_m=0

    MUSK_BLOCK_M1 :tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR   ##64 in orignal code.
    offs_n=start_n + tl.arange(0,BLOCK_N1)

    # algorithm 2 line 6
    k=tl.load(K+offs_n[:,None]*stride_s + offs_k[None,:]*stride_d)
    v=tl.load(V+offs_n[:,None]*stride_s + offs_k[None,:]*stride_d)
    # algorithm 2 line 7
    dv=tl.zeros_like([BLOCK_N1,HEAD_DIM],dtype=dtype) 
    dk=tl.zeros_like([BLOCK_N1,HEAD_DIM],dtype=dtype)

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR  #2
    ## outer loop for blocks of K,V
    ## quries_idx >=keys_idx. 

    ## why seperate the causal and non-causal :
    ## the causal is the first black of block_m = block_n.
    ## if add another loop in the non-causal part, it will decrease the performance.
    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk,dv=_attn_bwd_dkdv(dk=dk,dv=dv,M=M,D=D,
                    Q=Q,k=k,v=v,DO=DO,
                    stride_s=stride_s,stride_d=stride_d,
                    start_m=start_m,start_n=start_n,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_M1= MASK_BLOCK_M1,  #2 ,## adjusted by BLK_SLICE_FACTOR
                    BLOCK_N1=BLOCK_N1,
                    num_steps=num_steps,
                    MASK=True,)
        start_m += num_steps * MASK_BLOCK_M1   ## block_n1

    num_steps = (HEAD_DIM - start_m) // BLOCK_M1
    dk,dv=_attn_bwd_dkdv(dk=dk,dv=dv,M=M,D=D,
                    Q=Q,k=k,v=v,DO=DO,
                    stride_s=stride_s,stride_d=stride_d,
                    start_m=start_m,start_n=start_n,
                    HEAD_DIM=HEAD_DIM,
                    BLOCK_M1= BLOCK_M1,  #2 ,## adjusted by BLK_SLICE_FACTOR
                    BLOCK_N1=BLOCK_N1,
                    num_steps=num_steps,
                    MASK=False,)

    ## algorithm 2 line 18, store dk,dv to global memory

    dv_ptr=DV+offs_n[:,None]*stride_s + offs_k[None,:]*stride_d
    tl.store(dv_ptr,dv)

    dk*=sm_scale
    dk_ptr=DK+offs_n[:,None]*stride_s + offs_k[None,:]*stride_d
    tl.store(dk_ptr,dk)


    # preprocess for dQ.

    start_m=pid*BLOCK_M2
    start_n=0
    num_steps=SEQ_LEN // BLOCK_N2

    MASK_BLOCK_N2 : tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR  #for causal mask step.
    offs_m=start_m + tl.arange(0,BLOCK_M2)

    q=tl.load(Q + offs_m[:,None]*stride_s + offs_k[None,:]*stride_d)
    dq=tl.zeros_like([BLOCK_M2,HEAD_DIM],dtype=tl.float16)
    do=tl.load(DO + offs_m[:,None]*stride_s + offs_k[None,:]*stride_d)

    m=tl.load(M+offs_m)  #m do not have the head dimension level.
    m=m[:,None]

    if CAUSAL:
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.

        ## my explanation for the original author's explanation : that due to end_n -num_steps*MASK_BLOCK_N2
        dq=_attn_bwd_dq(dq,q,do,m,V,K,D,start_m,end_n -num_steps*MASK_BLOCK_N2,num_steps,
                stride_s=stride_s,stride_d=stride_d,
                 HEAD_DIM=HEAD_DIM,
                 BLOCK_M2=BLOCK_M2, BLOCK_N2=MASK_BLOCK_N2,
                 MASK=True,
                )
        end_n -=num_steps*MASK_BLOCK_N2

        num_steps= end_n  // BLOCK_N2
        start_n=end_n - num_steps*BLOCK_N2

    dq=_attn_bwd_dq(dq,q,do,m,V,K,D,start_m,start_n,num_steps,
            stride_s=stride_s,stride_d=stride_d,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            MASK=False,
            )
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)



class FlashAttention(torch.nn.Module):
    @staticmethod
    def forward(ctx,Q,K,V,sm_scale,causal=False,warp_specialize=True):
        #ctx is a context object that can be used to save intermediate results for backward pass
        #in order to calcalate gradients in backward pass
        #Q : (batch,num_heads,seq_len,head_dim)
        #when implementing the multi head attention, we first get (batch,seq_len,embed_dim) and then we reshape it to (batch,num_heads,seq_len,head_dim )
        HEAD_DIM_Q = Q.shape[-1]
        HEAD_DIM_K = K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        BATCH_SIZE,NUM_HEADS,SEQ_LEN,HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V, "Head dimensions of Q, K, V must be the same"
        #preliminary verification

        O=torch.empty_like(Q) #output tensor.

        stage= 3 if causal else 1
        #i am confused about the stage, i may use something easier to change it
        M=torch.empty((Q.shape[0],Q.shape[1],Q.shape[2]),dtype=torch.float16)
        # M is the space for maximum values
        grid= lambda meta : (
            triton.cdiv(SEQ_LEN,meta['BLOCK_SIZE']), #number of blocks in the sequence dimension
            BATCH_SIZE * NUM_HEADS,
            1
        )

        ctx.grid = grid
        M=torch.empty((BATCH_SIZE,NUM_HEADS,SEQ_LEN),dtype=torch.float16)
        #M serves as a temporary storage for the maximum values in the softmax computation, which is used for numerical stability.
        _attn_fwd[grid](sm_scale,
            BATCH_SIZE,NUM_HEADS,SEQ_LEN,HEAD_DIM,
            Q,K,V,
            M,
            O,
            FP8_OUTPUT=Q.dtype == torch.float8_e5m2,    
            STAGE=stage,
            waep_specialize=warp_specialize,
        )

        ## save
        ctx.save_for_backward(Q,K,V,O,M)
        ctx.sm_scale=sm_scale
        ctx.causal=causal
        ctx.HEAD_DIM=HEAD_DIM
        return O
    
    @staticmethod
    def backward(ctx, dO):
        q,k,v,o,m=ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH_SIZE,NUM_HEADS,SEQ_LEN,HEAD_DIM= q.shape

        NUM_WARPS, NUM_STAGES=4,5
        BLOCK_M1,BLOCK_N1,BLOCK_M2,BLOCK_N2=32,128,128,32 
        ##  M1 --> N1 --> M2 --> N2  (M1, N2 for Q, dQ, M2, N1 for K,V, dK,dV) ask ai if i forget
        ## why M1 not equal N1 : for M axis, it need to hold three matrix, Q,DO, DQ. But for N axis, it only need to hold K and V.
        ## Bascially, the block size for M axis is smaller than N axis, this may be related to the memory limitation.,
        ## Also, N1 % M1 == 0. That is because for causal mask, the Key can only access the past Queries.
        ## if N_1=64 and M1=128, in order to implement the causal mask, the GPU has to slice a tile in half and load a misaligned block from 64-191.
        BLK_SLICE_FACTOR = 2
        delta=torch.empty_like(m)

        pre_block=128

        pre_grid= [SEQ_LEN // pre_block, BATCH_SIZE*NUM_HEADS] 
        # two dimensions ? 

        _attn_bwd_preprocess[pre_grid](o,dO,
                delta,SEQ_LEN=SEQ_LEN,
                BLOCK_R=pre_block,HEAD_DIM=HEAD_DIM
            )

        grid= (SEQ_LEN // pre_block, 1 ,BATCH_SIZE*NUM_HEADS)

        _attn_bwd[grid](
            q,k,v,ctx.sm_scale,
            dO,
            dq,dk,dv,
            m,delta,
            stride_b=q.stride(0),
            stride_h=q.stride(1),
            stride_s=q.stride(2),
            stride_d=q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BATCH_SIZE=BATCH_SIZE,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1, BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
            NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
            CAUSAL=ctx.causal,
            BLK_SLICE_FACTOR = BLK_SLICE_FACTOR,
            dtype=q.dtype
        )

        return dq,dk,dv,None,None,None,None
    

attention=FlashAttention.apply

