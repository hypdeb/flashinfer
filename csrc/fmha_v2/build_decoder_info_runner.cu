namespace flashinfer
{

void build_decoder_info()
{
    BuildDecoderInfoParams<T> decoder_params{};
    decoder_params.seqQOffsets          = cu_q_seqlens;
    decoder_params.seqKVOffsets         = cu_kv_seqlens;
    decoder_params.seqCpPartialOffsets  = cu_cp_partial_seqlens;
    decoder_params.cpSize               = mCpSize;
    decoder_params.packedMaskRowOffsets = cu_mask_rows;
    decoder_params.paddingOffsets       = padding_offset;
    decoder_params.tokensInfo           = tokens_info;
    decoder_params.encoderPaddingOffsets =
        isCrossAttention() ? encoder_padding_offset : nullptr;  // cross attention takes offsets from encoder inputs
    decoder_params.attentionMask =
        isCrossAttention() ? nullptr : attention_mask;  // manually set for unfused cross attn
    // Fixed sequence length offset if not removing the padding (cu_q_seqlens[i] = i * seq_length).
    decoder_params.seqQLengths   = params.context_lengths;
    decoder_params.seqKVLengths  = isCrossAttention() ? params.encoder_input_lengths : params.sequence_lengths;
    decoder_params.batchSize     = params.batch_size;
    decoder_params.maxQSeqLength = params.input_seq_length;
    decoder_params.maxEncoderQSeqLength =
        isCrossAttention() ? params.cross_kv_length : 0;  // cross attention uses encoder seq length
    decoder_params.attentionWindowSize = params.cyclic_attention_window_size;
    decoder_params.sinkTokenLength     = params.sink_token_length;
    decoder_params.numTokens           = params.num_tokens;
    decoder_params.removePadding       = mRemovePadding;
    decoder_params.attentionMaskType   = mMaskType;
    decoder_params.blockSparseParams   = mBlockSparseParams;
    decoder_params.fmhaTileCounter     = fmha_tile_counter_ptr;
    decoder_params.quantScaleO         = params.attention_output_orig_quant;
    decoder_params.dequantScaleQkv     = params.kv_scale_quant_orig;
    decoder_params.separateQkvScales   = mKVCacheQuantMode.hasFp4KvCache();
    decoder_params.fmhaHostBmm1Scale   = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    decoder_params.fmhaBmm1Scale       = fmha_bmm1_scale_ptr;
    decoder_params.fmhaBmm2Scale       = fmha_bmm2_scale_ptr;
    // Rotary embedding inv_freq buffer.
    decoder_params.rotaryEmbeddingScale = mRotaryEmbeddingScale;
    decoder_params.rotaryEmbeddingBase  = mRotaryEmbeddingBase;
    decoder_params.rotaryEmbeddingDim   = mRotaryEmbeddingDim;
    decoder_params.rotaryScalingType    = mRotaryEmbeddingScaleType;
    // The inv freq might be updated during runtime with dynamic scaling type.
    decoder_params.rotaryEmbeddingInvFreq = rotary_inv_freq_buf;
    // This is pre-computed when building the engines.
    decoder_params.rotaryEmbeddingInvFreqCache = params.rotary_inv_freq;
    decoder_params.rotaryEmbeddingMaxPositions = mRotaryEmbeddingMaxPositions;

    invokeBuildDecoderInfo(decoder_params, stream);
}

}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_decoder_info, flashinfer::build_decoder_info);
