//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

private enum RopeParametersCodingKey: String, CodingKey {
    case ropeParameters = "rope_parameters"
}

public struct Qwen35TextConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 14336
    var attentionHeads: Int = 32
    var kvHeads: Int = 8
    var linearNumValueHeads: Int = 64
    var linearNumKeyHeads: Int = 16
    var linearKeyHeadDim: Int = 192
    var linearValueHeadDim: Int = 128
    var linearConvKernelDim: Int = 4
    var rmsNormEps: Float = 1e-6
    var vocabularySize: Int = 151_936
    var ropeTheta: Float = 100000.0
    var partialRotaryFactor: Float = 0.25
    var maxPositionEmbeddings: Int = 131072
    var tieWordEmbeddings: Bool = false
    var attentionBias: Bool = false
    var headDim: Int?
    var ropeScaling: [String: StringOrNumber]?
    var fullAttentionInterval: Int = 4

    // MTP (multi-token-prediction) fields. Default 0 ⇒ no head, checkpoints
    // without MTP decode unchanged (Part-C §1.3).
    var mtpNumHiddenLayers: Int = 0
    // Present in some checkpoints' config; not yet supported (see
    // Qwen35TextModel.speculationCapability) — decoded so it round-trips
    // rather than silently ignored, but a `true` value here disables
    // speculation rather than guessing at unimplemented behaviour.
    var mtpUseDedicatedEmbeddings: Bool = false

    // MoE fields
    var numExperts: Int = 0
    var numExpertsPerTok: Int = 0
    var decoderSparseStep: Int = 1
    var sharedExpertIntermediateSize: Int = 0
    var moeIntermediateSize: Int = 0
    var normTopkProb: Bool = true

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case fullAttentionInterval = "full_attention_interval"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaultRopeParameters: [String: StringOrNumber] = [
            "type": .string("default"),
            "mrope_section": .ints([11, 11, 10]),
            "rope_theta": .float(100000.0),
            "partial_rotary_factor": .float(0.25),
        ]

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        self.attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        self.linearNumValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
        self.linearNumKeyHeads =
            try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        self.linearKeyHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
        self.linearValueHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        self.linearConvKernelDim =
            try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        self.fullAttentionInterval =
            try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.mtpNumHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
        self.mtpUseDedicatedEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .mtpUseDedicatedEmbeddings) ?? false

        // MoE fields
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.decoderSparseStep =
            try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        self.sharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

        let ropeContainer = try decoder.container(keyedBy: RopeParametersCodingKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        if var ropeParameters {
            if ropeParameters["type"] == nil, let ropeType = ropeParameters["rope_type"] {
                ropeParameters["type"] = ropeType
            }
            self.ropeTheta = ropeParameters["rope_theta"]?.asFloat() ?? 100000.0
            self.partialRotaryFactor =
                ropeParameters["partial_rotary_factor"]?.asFloat() ?? 0.25
            self.ropeScaling = ropeParameters
        } else {
            self.ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100000.0
            self.partialRotaryFactor =
                try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? defaultRopeParameters
        }

        if self.headDim == nil {
            self.headDim = self.hiddenSize / self.attentionHeads
        }
    }
}

// MARK: - GatedDeltaNet

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    @ModuleInfo(key: "norm") var norm: Qwen3NextRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.hiddenSize = args.hiddenSize
        self.numVHeads = args.linearNumValueHeads
        self.numKHeads = args.linearNumKeyHeads
        self.headKDim = args.linearKeyHeadDim
        self.headVDim = args.linearValueHeadDim
        self.keyDim = headKDim * numKHeads
        self.valueDim = headVDim * numVHeads
        self.convKernelSize = args.linearConvKernelDim
        self.convDim = keyDim * 2 + valueDim

        precondition(
            numVHeads % numKHeads == 0,
            "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
        )

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: convDim,
            bias: false
        )

        _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
        _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
        _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

        _dtBias.wrappedValue = MLXArray.ones([numVHeads])
        let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
        _aLog.wrappedValue = log(a)

        _norm.wrappedValue = Qwen3NextRMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
        _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)

        super.init()
    }

    /// Process one contiguous chunk of the sequence through conv1d + the gated
    /// delta-net recurrence, given an explicit starting (conv, ssm) state.
    /// Factored out of `callAsFunction` so speculative verification can call it
    /// twice — once for the confirmed tokens, once for the draft tokens — with
    /// the recurrent state snapshotted at the boundary between the two calls.
    /// For the ordinary (non-speculative) path this is called once over the
    /// whole sequence and is numerically identical to the pre-MTP code.
    private func processChunk(
        qkvChunk: MLXArray, aChunk: MLXArray, bChunk: MLXArray,
        convState: MLXArray, ssmState: MLXArray?, mask: MLXArray?
    ) -> (out: MLXArray, convState: MLXArray, ssmState: MLXArray?) {
        let B = qkvChunk.dim(0)
        let Sc = qkvChunk.dim(1)

        var qkv = qkvChunk
        if let mask {
            qkv = MLX.where(mask[.ellipsis, .newAxis], qkv, 0)
        }

        let convInput = concatenated([convState, qkv], axis: 1)
        let newConvState = convInput[0..., (-(convKernelSize - 1))...]
        let convOut = silu(conv1d(convInput))

        let convSplit = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = convSplit[0].reshaped(B, Sc, numKHeads, headKDim)
        let k = convSplit[1].reshaped(B, Sc, numKHeads, headKDim)
        let v = convSplit[2].reshaped(B, Sc, numVHeads, headVDim)

        let dtype = q.dtype
        let invScale = pow(Float(headKDim), -0.5)
        let qNormed =
            MLXArray(pow(invScale, 2)).asType(dtype)
            * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kNormed =
            MLXArray(invScale).asType(dtype)
            * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let (out, newSsmState) = gatedDeltaUpdate(
            q: qNormed,
            k: kNormed,
            v: v,
            a: aChunk,
            b: bChunk,
            aLog: aLog,
            dtBias: dtBias,
            state: ssmState,
            mask: mask
        )

        return (out, newConvState, newSsmState)
    }

    /// - Parameter confirmedPrefix: how many of the leading `S` positions are
    ///   already-confirmed tokens (speculative verification); `0` outside
    ///   speculative decoding. When `0 < confirmedPrefix < S`, the confirmed
    ///   and draft sub-ranges are processed as two chunks so the recurrent
    ///   state can be snapshotted at the boundary between them
    ///   (`cache.markSpeculationBoundary()`) — the snapshot the engine
    ///   restores via `cache.rollbackToBoundary` on rejection (Part-B §2).
    func callAsFunction(
        _ inputs: MLXArray,
        mask: MLXArray? = nil,
        cache: MambaCache? = nil,
        confirmedPrefix: Int = 0
    ) -> MLXArray {
        let B = inputs.dim(0)
        let S = inputs.dim(1)

        let qkv = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        let initialConvState: MLXArray
        if let cacheState = cache?[0] {
            initialConvState = cacheState
        } else {
            initialConvState = MLXArray.zeros(
                [B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        }
        let initialSsmState = cache?[1]

        let out: MLXArray

        if confirmedPrefix > 0 && confirmedPrefix < S {
            let (outC, convC, ssmC) = processChunk(
                qkvChunk: qkv[0..., ..<confirmedPrefix],
                aChunk: a[0..., ..<confirmedPrefix],
                bChunk: b[0..., ..<confirmedPrefix],
                convState: initialConvState, ssmState: initialSsmState,
                mask: mask?[0..., ..<confirmedPrefix]
            )
            if let cache {
                // State as of exactly the confirmed tokens — the boundary the
                // recurrent cache must be able to return to on rejection.
                cache[0] = convC
                cache[1] = ssmC
                cache.markSpeculationBoundary()
            }
            let (outD, convF, ssmF) = processChunk(
                qkvChunk: qkv[0..., confirmedPrefix...],
                aChunk: a[0..., confirmedPrefix...],
                bChunk: b[0..., confirmedPrefix...],
                convState: convC, ssmState: ssmC,
                mask: mask?[0..., confirmedPrefix...]
            )
            if let cache {
                // Optimistic advance through the draft tokens too — rolled
                // back by the engine if verification rejects them.
                cache[0] = convF
                cache[1] = ssmF
            }
            out = concatenated([outC, outD], axis: 1)
        } else {
            let (outFull, convF, ssmF) = processChunk(
                qkvChunk: qkv, aChunk: a, bChunk: b,
                convState: initialConvState, ssmState: initialSsmState, mask: mask
            )
            if let cache {
                cache[0] = convF
                cache[1] = ssmF
            }
            out = outFull
        }

        let gated = norm(out, gate: z)
        return outProj(gated.reshaped(B, S, -1))
    }
}

// MARK: - Attention

final class Qwen35Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        let headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * 2, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qSplit[0]
        let gate = qSplit[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(sigmoidMultiply(output, gate))
    }
}

// MARK: - SparseMoeBlock

final class Qwen35SparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextConfiguration) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = weightedExpertSum(y, scores)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration, layerIdx: Int) {
        self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

        if isLinear {
            _linearAttn.wrappedValue = Qwen35GatedDeltaNet(args)
        } else {
            _selfAttn.wrappedValue = Qwen35Attention(args)
        }

        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize,
            eps: args.rmsNormEps
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?,
        confirmedPrefix: Int = 0
    ) -> MLXArray {
        let r: MLXArray
        if isLinear {
            r = linearAttn!(
                inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                confirmedPrefix: confirmedPrefix)
        } else {
            r = selfAttn!(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

// MARK: - Text Model

public class Qwen35TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen35DecoderLayer]
    let norm: RMSNorm

    let ssmIdx: Int
    let faIdx: Int

    init(_ args: Qwen35TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize,
            dimensions: args.hiddenSize
        )

        self.layers = (0 ..< args.hiddenLayers).map { layerIdx in
            Qwen35DecoderLayer(args, layerIdx: layerIdx)
        }

        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.ssmIdx = 0
        self.faIdx = args.fullAttentionInterval - 1

        super.init()
    }

    /// Backbone forward, returning the **pre-final-norm** hidden state — what
    /// the MTP head fuses with the next token's embedding (Part-A §4). The
    /// public `callAsFunction` below is a thin wrapper applying `norm` on top;
    /// this split is a pure refactor (identical output for `confirmedPrefix: 0`)
    /// so both entry points share one implementation.
    ///
    /// - Parameter confirmedPrefix: forwarded to each linear-attention layer as
    ///   its speculative-verification snapshot boundary (`0` = no speculation).
    func hiddenStates(
        _ inputs: MLXArray, cache: [KVCache?]? = nil, confirmedPrefix: Int = 0
    ) -> MLXArray {
        var hiddenStates = embedTokens(inputs)

        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let faMask = createAttentionMask(h: hiddenStates, cache: cacheArray?[faIdx])
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray?[i],
                confirmedPrefix: confirmedPrefix)
        }

        return hiddenStates
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(hiddenStates(inputs, cache: cache))
    }
}

// MARK: - MTP (multi-token-prediction) head
//
// ml-explore/mlx-lm#990 (reference to port; Part-A §4). Predicts token `t+2`
// from the backbone's pre-final-norm hidden state at `t` and the (sampled)
// token `t+1`'s embedding. No second resident model — the fused head lives
// inside Qwen3.6's own checkpoint.
//
// Weight key names verified against an actual checkpoint (not assumed): the
// standalone-drafter release `mlx-community/Qwen3.6-35B-A3B-MTP-bf16` (its own
// `model.safetensors` header, fetched as a metadata-only HTTP range read) and
// a merged, MTP-preserving quantized checkpoint
// (`stamsam/...-MLX-oQ4-MTP/model.safetensors.index.json`), whose keys are
// `language_model.mtp.{fc,norm,pre_fc_norm_hidden,pre_fc_norm_embedding}.weight`
// and `language_model.mtp.layers.<i>.{input_layernorm,post_attention_layernorm,
// self_attn.*,mlp.*}` — i.e. every MTP key is `mtp.`-prefixed once merged into
// a full checkpoint, and the `mlp.*` / `self_attn.*` sub-keys are byte-for-byte
// the same names `Qwen35SparseMoeBlock` / `Qwen35Attention` already declare.
// No `eh_proj` / `shared_head` / `nextn`-style bare keys were found for this
// architecture — that caution in Part-A §3.6 does not materialize for Qwen3.6.

/// Full-attention-only transformer layer for the MTP head — reuses
/// `Qwen35Attention` and the backbone's dense/MoE MLP block directly (verified
/// same weight-key shape as the backbone's non-linear decoder layer); the MTP
/// head never has a linear-attention (`GatedDeltaNet`) variant.
final class Qwen35MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        if args.numExperts > 0 {
            _mlp.wrappedValue = Qwen35SparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = Qwen3NextMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}

/// Multi-token-prediction head. Logits are **not** produced here — the caller
/// (`Qwen35TextModel.mtpForward`) applies the shared `lm_head` /
/// embedding-as-linear, exactly like the backbone does, so the head never
/// duplicates the output projection.
final class Qwen35MTPHead: Module {
    @ModuleInfo(key: "pre_fc_norm_hidden") var preFcNormHidden: RMSNorm
    @ModuleInfo(key: "pre_fc_norm_embedding") var preFcNormEmbedding: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    // Plain (non-@ModuleInfo) array, matching the backbone's own `layers`
    // convention (Qwen35TextModelInner.layers) — MLX Swift's reflection uses
    // the Swift property name as the key path segment, which is already
    // "layers", matching the checkpoint's `mtp.layers.<i>.*` verbatim.
    let layers: [Qwen35MTPDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _preFcNormHidden.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _preFcNormEmbedding.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        self.layers = (0 ..< max(args.mtpNumHiddenLayers, 0)).map { _ in
            Qwen35MTPDecoderLayer(args)
        }
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    /// - Parameters:
    ///   - hiddenState: backbone pre-final-norm hidden state, `[B, N, H]`.
    ///   - nextTokenIds: the token(s) positionally following `hiddenState`, `[B, N]`.
    ///   - embedTokens: the backbone's embedding table (shared — not duplicated
    ///     here; `mtp_use_dedicated_embeddings` is not supported, see
    ///     `Qwen35TextModel.speculationCapability`).
    ///   - cache: the MTP head's own KV cache — independent of the backbone's,
    ///     one entry per MTP layer (`Qwen35TextModel.makeMTPCache()`).
    /// - Returns: fused, normed hidden state `[B, N, H]` — not logits.
    func callAsFunction(
        _ hiddenState: MLXArray, nextTokenIds: MLXArray, embedTokens: Embedding,
        cache: [KVCache]
    ) -> MLXArray {
        let e = preFcNormEmbedding(embedTokens(nextTokenIds))
        let h = preFcNormHidden(hiddenState)
        var fused = fc(concatenated([e, h], axis: -1))

        let mask = createAttentionMask(h: fused, cache: cache.first)
        for (layer, c) in zip(layers, cache) {
            fused = layer(fused, mask: mask, cache: c)
        }

        return norm(fused)
    }
}

public class Qwen35TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    /// Present iff the loaded checkpoint's weights actually contain `mtp.*`
    /// keys — allocated in `sanitize(weights:)`, NOT here at construction.
    ///
    /// **This was originally config-driven** (`mtpNumHiddenLayers > 0`) and
    /// that was wrong: verified directly against the checkpoint we actually
    /// serve (`lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit`), its config
    /// reports `mtp_num_hidden_layers: 1` (inherited from the full
    /// architecture spec) while its weights carry **zero** `mtp.*` keys (the
    /// MLX conversion strips the head). Config-driven allocation would
    /// declare an `mtp` submodule with no backing weights, and strict
    /// loading (`Load.swift`, `verify: [.all]` ⊇ `.allModelKeysSet`) would
    /// throw on every load of the model we actually serve — the exact "a
    /// model that serves today must still serve" failure this whole
    /// mechanism exists to prevent. Weight presence is checked directly in
    /// `sanitize`, which sees the real file; config is no longer trusted for
    /// this decision at all.
    @ModuleInfo(key: "mtp") var mtp: Qwen35MTPHead?

    public init(_ args: Qwen35TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen35TextModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        // `mtp` is intentionally NOT allocated here — see its doc comment.
        // `sanitize(weights:)` allocates it iff the weight file actually has
        // `mtp.*` keys, and `attachSeparateMTPHead(from:)` (Part-C's separate-
        // checkpoint path) can allocate it later still, post-load.
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Speculative-decoding-aware forward (Part-C §1.2). Additive: the plain
    /// `callAsFunction(_:cache:)` above is untouched. `confirmedPrefix` reaches
    /// the linear-attention layers, which use it as their snapshot boundary;
    /// `hidden` is the pre-final-norm state the MTP head needs.
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, confirmedPrefix: Int
    ) -> (logits: MLXArray, hidden: MLXArray?) {
        let preNorm = model.hiddenStates(inputs, cache: cache, confirmedPrefix: confirmedPrefix)
        let normed = model.norm(preNorm)
        let logits = lmHead.map { $0(normed) } ?? model.embedTokens.asLinear(normed)
        return (logits, preNorm)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isLinear {
                return MambaCache()
            }
            return KVCacheSimple()
        }
    }

    /// Whether this model can speculate — computed once at load (Part-C §1.4).
    /// `mtp_use_dedicated_embeddings` is not supported by this port (the head
    /// always shares the backbone's embedding table); a checkpoint requesting
    /// it fails closed to non-speculative service rather than silently
    /// producing wrong fused-embedding input.
    public var speculationCapability: SpeculationCapability? {
        guard let mtp, !configuration.mtpUseDedicatedEmbeddings else {
            let reason =
                configuration.mtpUseDedicatedEmbeddings
                ? "mtp_use_dedicated_embeddings=true is not supported by this port "
                    + "(shared-embedding MTP head only) — serving non-speculatively"
                : "checkpoint has no MTP head (mtp_num_hidden_layers=0 or missing mtp.* weights)"
            return SpeculationCapability(hasHeads: false, allCachesRestorable: false, reason: reason)
        }
        let scratchCache = newCache(parameters: nil)
        let restorable = canRestoreCache(scratchCache)
        // Temporary diagnostic (2026-07-20): the reason string used to be a
        // flat, unconditional sentence with no per-layer detail. The real
        // served checkpoint reported `not restorable` despite every layer
        // `newCache()` can produce (MambaCache / KVCacheSimple) being
        // unconditionally `isRestorable == true` by static reading — this
        // enriches the message with the offending indices/types so the next
        // load tells us directly rather than requiring another synthetic
        // repro. Remove once the discrepancy is understood.
        let reason: String?
        if restorable {
            reason = nil
        } else {
            let bad = scratchCache.enumerated().compactMap { i, c -> String? in
                c.isRestorable ? nil : "\(i):\(type(of: c))"
            }
            reason =
                "one or more cache types in this model cannot be restored "
                + "(non-restorable: [\(bad.joined(separator: ", "))] of \(scratchCache.count) total)"
        }
        return SpeculationCapability(
            hasHeads: true, allCachesRestorable: restorable, proposalDepth: 1, reason: reason)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // --- The trap (Part-A §3.6 / Part-C §1.3) ---
        // The presence of `mtp.` keys used to be OR'd into the norm-shift
        // signal. That conflated two unrelated things: "is this a raw
        // (unsanitized) HF checkpoint" vs "does this checkpoint carry MTP
        // weights". An already-converted MLX checkpoint that happens to carry
        // `mtp.*` weights must NOT be shifted again. Detection now fires on
        // exactly the condition it always should have: unsanitized
        // (PyTorch-layout) conv1d weights — mirroring the fix upstream landed
        // in ml-explore/mlx-lm#990 itself (its own commit message: "norm +1
        // shift now triggered only on raw HF checkpoints ... not on presence
        // of MTP weights").
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        // Allocate `mtp` from ACTUAL WEIGHT PRESENCE, not config (see `mtp`'s
        // doc comment for why config alone is not trustworthy here — the
        // checkpoint we serve has `mtp_num_hidden_layers: 1` in its config
        // and zero `mtp.*` weights). `sanitize` is the first point that sees
        // the real file, so it's the right place for this decision, even
        // though it's a mutating side effect in what's nominally a pure
        // weight transform — the established precedent for post-construction
        // module attachment in this fork (LoRA's runtime adapter loader).
        let hasMTPWeights = weights.keys.contains { $0.contains("mtp.") }
        if hasMTPWeights, mtp == nil {
            let layerCount = configuration.mtpNumHiddenLayers > 0 ? configuration.mtpNumHiddenLayers : 1
            var headConfig = configuration
            headConfig.mtpNumHiddenLayers = layerCount
            _mtp.wrappedValue = Qwen35MTPHead(headConfig)
        }

        // Keep `mtp.*` weights only when this model actually has the head
        // allocated (now driven by weight presence, immediately above);
        // otherwise drop them exactly as before, so non-MTP checkpoints are
        // unaffected.
        var weights = weights
        if mtp == nil {
            weights = weights.filter { !$0.key.contains("mtp.") }
        }

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            // MTP-specific norms (not covered by the patterns above).
            ".pre_fc_norm_hidden.weight",
            ".pre_fc_norm_embedding.weight",
            "mtp.norm.weight",
        ]

        for k in Array(weights.keys) {
            guard let v = weights[k] else { continue }
            if k.contains("conv1d.weight") && v.dim(-1) != 1 {
                weights[k] = v.movedAxis(source: 2, destination: 1)
                continue
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                weights[k] = v + MLXArray(1, dtype: v.dtype)
            }
        }

        return weights
    }
}

extension Qwen35TextModel: MTPSpeculativeModel {
    public func makeMTPCache() -> [KVCache] {
        guard let mtp else { return [] }
        return mtp.layers.map { _ in KVCacheSimple() }
    }

    public func mtpForward(hiddenState: MLXArray, nextTokenIds: MLXArray, cache: [KVCache])
        -> MLXArray
    {
        guard let mtp else {
            fatalError(
                "mtpForward called on a model with no MTP head — check speculationCapability first"
            )
        }
        let fused = mtp(
            hiddenState, nextTokenIds: nextTokenIds, embedTokens: model.embedTokens, cache: cache)
        return lmHead.map { $0(fused) } ?? model.embedTokens.asLinear(fused)
    }
}

extension Qwen35TextModel: MTPHeadAttachable {
    /// Loads `directory` as a standalone MTP-head checkpoint (config.json +
    /// one or more `*.safetensors`, unprefixed keys — see
    /// `MTPHeadAttachable`'s doc comment) and attaches it. Fails closed: any
    /// problem logs once to stderr and returns `false`; never throws, never
    /// crashes a model that otherwise loads and serves fine.
    ///
    /// Key mapping: **none needed.** The standalone file's keys (`fc.weight`,
    /// `layers.0.input_layernorm.weight`, `norm.weight`, …) are already
    /// rooted at the head itself — exactly what `Qwen35MTPHead.update(
    /// parameters:)` expects when called directly on a freestanding head
    /// instance (as opposed to the fused path, where those same module names
    /// are reached via the main model's `mtp.` prefix). Loading the
    /// standalone file onto its own `Qwen35MTPHead` object needs no renaming
    /// at all.
    ///
    /// Quantization: the one real standalone release
    /// (`mlx-community/Qwen3.6-35B-A3B-MTP-bf16`) is bf16/unquantized, and is
    /// loaded as-is here — MLX's per-op dtype promotion handles the boundary
    /// against an 8-bit-quantized main model's (bf16-computed) hidden state
    /// without an explicit cast. A **quantized** standalone head is not
    /// specially handled (no group-size/bits re-derivation from `.scales`
    /// keys) — flagged as a known gap, not silently "handled": such a file
    /// would fail to load here (a plain Linear can't absorb `.scales`/
    /// `.biases` tensors), and this function's `catch` reports that plainly
    /// rather than crashing the process.
    public func attachSeparateMTPHead(from directory: URL) -> Bool {
        if mtp != nil {
            FileHandle.standardError.write(
                Data(
                    "mlx: fused MTP head already present — ignoring separate head directory \(directory.path)\n"
                        .utf8))
            return true
        }
        do {
            let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let headTop = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
            let headConfig = headTop.textConfig

            guard headConfig.mtpNumHiddenLayers > 0 else {
                FileHandle.standardError.write(
                    Data(
                        "mlx: MTP head directory \(directory.path) has mtp_num_hidden_layers=0 — not a valid head, ignoring\n"
                            .utf8))
                return false
            }
            guard headConfig.hiddenSize == configuration.hiddenSize else {
                let msg: String =
                    "mlx: MTP head directory \(directory.path) hiddenSize=\(headConfig.hiddenSize) "
                    + "!= main model hiddenSize=\(configuration.hiddenSize) — refusing to attach\n"
                FileHandle.standardError.write(Data(msg.utf8))
                return false
            }

            let head = Qwen35MTPHead(headConfig)

            var weights = [String: MLXArray]()
            let files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
            var sawSafetensors = false
            for file in files where file.pathExtension == "safetensors" {
                let (w, _) = try loadArraysAndMetadata(url: file)
                for (k, v) in w { weights[k] = v }
                sawSafetensors = true
            }
            guard sawSafetensors, !weights.isEmpty else {
                FileHandle.standardError.write(
                    Data(
                        "mlx: MTP head directory \(directory.path) has no *.safetensors — refusing to attach\n"
                            .utf8))
                return false
            }
            if weights.keys.contains(where: { $0.hasSuffix(".scales") }) {
                let msg: String =
                    "mlx: MTP head directory \(directory.path) appears to be quantized "
                    + "(.scales keys present) — not supported by this port, refusing to attach\n"
                FileHandle.standardError.write(Data(msg.utf8))
                return false
            }

            let params = ModuleParameters.unflattened(weights)
            try head.update(parameters: params, verify: [.all])
            eval(head)

            _mtp.wrappedValue = head
            let msg: String =
                "mlx: MTP head attached from separate checkpoint \(directory.path) "
                + "(\(headConfig.mtpNumHiddenLayers) layer(s), hiddenSize=\(headConfig.hiddenSize))\n"
            FileHandle.standardError.write(Data(msg.utf8))
            return true
        } catch {
            FileHandle.standardError.write(
                Data(
                    "mlx: failed to attach MTP head from \(directory.path): \(error) — serving non-speculatively\n"
                        .utf8))
            return false
        }
    }
}

extension Qwen35TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Top-level Model

public class Qwen35Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public init(_ args: Qwen35Configuration) {
        let textModel = Qwen35TextModel(args.textConfig)
        self.vocabularySize = textModel.vocabularySize
        self.kvHeads = textModel.kvHeads
        _languageModel.wrappedValue = textModel
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Passthrough — `LLMModelFactory` returns this wrapper as the concrete
    /// `any LanguageModel`, so the confirmed-prefix overload and speculation
    /// capability must be forwarded here, not just implemented on the inner
    /// `Qwen35TextModel` (which no caller outside this file ever sees).
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?, confirmedPrefix: Int
    ) -> (logits: MLXArray, hidden: MLXArray?) {
        languageModel(inputs, cache: cache, confirmedPrefix: confirmedPrefix)
    }

    public var speculationCapability: SpeculationCapability? {
        languageModel.speculationCapability
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }

            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        return languageModel.sanitize(weights: sanitized)
    }
}

extension Qwen35Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

extension Qwen35Model: MTPSpeculativeModel {
    public func makeMTPCache() -> [KVCache] {
        languageModel.makeMTPCache()
    }

    public func mtpForward(hiddenState: MLXArray, nextTokenIds: MLXArray, cache: [KVCache])
        -> MLXArray
    {
        languageModel.mtpForward(hiddenState: hiddenState, nextTokenIds: nextTokenIds, cache: cache)
    }
}

extension Qwen35Model: MTPHeadAttachable {
    @discardableResult
    public func attachSeparateMTPHead(from directory: URL) -> Bool {
        languageModel.attachSeparateMTPHead(from: directory)
    }
}
