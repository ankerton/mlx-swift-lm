// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/mistral4.py
// (Mistral Small 4: DeepSeek-V3-style multi-head latent attention, softmax-routed
// experts with a shared expert, YaRN rope, llama-4 attention scaling).
//
// The attention is the uncompressed form (kv_b_proj applied per step), as in
// DeepseekV3.swift: mlx-lm absorbs kv_b_proj into embed_q / unembed_out to keep a
// smaller cache; that is a speed/memory refinement over the same mathematics.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Mistral4Configuration: Codable, Sendable {
    var modelType: String = "mistral4"
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var normTopkProb: Bool
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var topkGroup: Int
    var numExpertsPerTok: Int
    var firstKDenseReplace: Int
    var nGroup: Int
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var ropeTheta: Float
    var tieWordEmbeddings: Bool
    var ropeInterleave: Bool?
    var attentionBias: Bool
    var ropeParameters: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case normTopkProb = "norm_topk_prob"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case nGroup = "n_group"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeInterleave = "rope_interleave"
        case attentionBias = "attention_bias"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let top = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try decoder.container(keyedBy: VLMCodingKeys.self)
        // Mistral Small 4 ships as a mistral3 VLM: the text model's settings sit under text_config.
        let c =
            nested.contains(.textConfig)
            ? try nested.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            : top

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral4"
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? (try top.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)) ?? false
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        // rope_parameters carries rope_theta and the YaRN scaling (as in mlx-lm's __post_init__).
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        let fromParameters = ropeParameters?["rope_theta"]?.asFloat()
        let direct = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = fromParameters ?? direct ?? 10_000
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
    }

    var qHeadDim: Int { qkNopeHeadDim + qkRopeHeadDim }
}

/// mlx-lm `apply_yarn_mscale`: the `mscale_all_dim` half of YaRN folded into the attention scale.
private func yarnAttentionScale(_ scale: Float, _ rope: [String: StringOrNumber]?) -> Float {
    guard let rope else { return scale }
    func text(_ v: StringOrNumber?) -> String? { if case .string(let s)? = v { return s }; return nil }
    let type = text(rope["type"]) ?? text(rope["rope_type"]) ?? "default"
    let mscaleAllDim = rope["mscale_all_dim"]?.asFloat() ?? 0
    let factor = rope["factor"]?.asFloat() ?? 1
    if type == "default" || mscaleAllDim == 0 || factor <= 1 { return scale }
    let s = 0.1 * mscaleAllDim * log(factor) + 1.0
    return scale * s * s
}

/// Llama-4 position scaling of the query: `1 + beta * log(1 + floor(pos / original_max))`.
private func llama4AttentionScale(start: Int, stop: Int, beta: Float, originalMax: Int) -> MLXArray {
    let positions = MLXArray(Int32(start) ..< Int32(stop)).asType(.float32)
    let scaling = 1 + beta * MLX.log(1 + MLX.floor(positions / Float(originalMax)))
    return scaling[0..., .newAxis]
}

class Mistral4Attention: Module {
    let config: Mistral4Configuration
    let numHeads: Int
    let qHeadDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: Mistral4Configuration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.qHeadDim = config.qHeadDim
        self.scale = yarnAttentionScale(pow(Float(config.qHeadDim), -0.5), config.ropeParameters)

        if let qLoraRank = config.qLoraRank {
            _qAProj.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: config.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: 1e-6)
            _qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * qHeadDim, bias: false)
        }
        _kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, config.kvLoraRank + config.qkRopeHeadDim, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: 1e-6)
        _kvBProj.wrappedValue = Linear(
            config.kvLoraRank, numHeads * (config.qkNopeHeadDim + config.vHeadDim), bias: false)
        _oProj.wrappedValue = Linear(numHeads * config.vHeadDim, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: config.qkRopeHeadDim, base: config.ropeTheta,
            traditional: config.ropeInterleave ?? true,
            scalingConfig: config.ropeParameters,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q: MLXArray
        if let qProj { q = qProj(x) } else { q = qBProj!(qALayerNorm!(qAProj!(x))) }
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qs = split(q, indices: [config.qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (qs[0], qs[1])

        let ckv = split(kvAProjWithMqa(x), indices: [config.kvLoraRank], axis: -1)
        var kPe = ckv[1].reshaped(B, L, 1, config.qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kv = kvBProj(kvALayerNorm(ckv[0]))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let kvs = split(kv, indices: [config.qkNopeHeadDim], axis: -1)
        let kNope = kvs[0]
        let values = kvs[1]

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        // The llama-4 scale applies to the whole query.
        qNope = qNope * attnScale
        qPe = qPe * attnScale

        // attentionWithCacheUpdate puts this step's keys and values in the cache (plain or quantised).
        let output = attentionWithCacheUpdate(
            queries: concatenated([qNope, qPe], axis: -1),
            keys: concatenated([kNope, kPe], axis: -1), values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(output)
    }
}

class Mistral4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }
}

class Mistral4MoE: Module, UnaryLayer {
    let config: Mistral4Configuration
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "shared_experts") var sharedExperts: Mistral4MLP?

    init(_ config: Mistral4Configuration) {
        self.config = config
        let experts = config.nRoutedExperts ?? 1
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize, numExperts: experts)
        _gate.wrappedValue = Linear(config.hiddenSize, experts, bias: false)
        if let shared = config.nSharedExperts {
            _sharedExperts.wrappedValue = Mistral4MLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.moeIntermediateSize * shared)
        }
    }

    /// mlx-lm `mistral4_expert_select`: softmax over the experts (no correction bias),
    /// optional group pruning, top-k, renormalised, scaled.
    func select(_ gates: MLXArray) -> (MLXArray, MLXArray) {
        let scores = softmax(gates.asType(.float32), axis: -1)
        var forChoice = scores
        if config.nGroup > 1 {
            var grouped = scores.reshaped(Array(scores.shape.dropLast()) + [config.nGroup, -1])
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = config.nGroup - config.topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            grouped = putAlong(grouped, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            forChoice = flattened(grouped, start: -2, end: -1)
        }
        let k = config.numExpertsPerTok
        let inds = stopGradient(argPartition(-forChoice, kth: k - 1, axis: -1)[.ellipsis, ..<k])
        var selected = takeAlong(scores, inds, axis: -1)
        if config.normTopkProb {
            selected = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (inds, selected * config.routedScalingFactor)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = select(gate(x))
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts { y = y + sharedExperts(x) }
        return y
    }
}

class Mistral4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Mistral4Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Mistral4Configuration, layerIdx: Int) {
        _selfAttn.wrappedValue = Mistral4Attention(config)
        if config.nRoutedExperts != nil, layerIdx >= config.firstKDenseReplace {
            mlp = Mistral4MoE(config)
        } else {
            mlp = Mistral4MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attnScale: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), attnScale: attnScale, mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Mistral4ModelInner: Module {
    let config: Mistral4Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Mistral4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Mistral4Configuration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = (0 ..< config.numHiddenLayers).map { Mistral4DecoderLayer(config, layerIdx: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil) -> MLXArray {
        var h = inputEmbeddings ?? embedTokens(inputs)
        let offset = cache?.first?.offset ?? 0
        let mask = createAttentionMask(h: h, cache: cache?.first)
        let attnScale: MLXArray
        if let rope = config.ropeParameters,
            let beta = rope["llama_4_scaling_beta"]?.asFloat(),
            let originalMax = rope["original_max_position_embeddings"]?.asInt()
        {
            attnScale = llama4AttentionScale(
                start: offset, stop: offset + h.dim(1), beta: beta, originalMax: originalMax
            ).asType(h.dtype)
        } else {
            attnScale = MLXArray.ones([h.dim(1), 1]).asType(h.dtype)
        }
        for (i, layer) in layers.enumerated() {
            h = layer(h, attnScale: attnScale, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class Mistral4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let config: Mistral4Configuration
    public let model: Mistral4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Mistral4Configuration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = Array(repeating: config.numAttentionHeads, count: config.numHiddenLayers)
        self.model = Mistral4ModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead { return lmHead(out) }
        return model.embedTokens.asLinear(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // A mistral3 VLM checkpoint: the text model sits under language_model; the
        // vision tower and projector are not part of this model.
        var w = weights
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            w = Dictionary(uniqueKeysWithValues: lm.flattened())
        }
        w = w.filter { !$0.key.contains("rotary_emb.inv_freq") && !$0.key.hasSuffix("activation_scale") }
        if config.tieWordEmbeddings { w["lm_head.weight"] = nil }

        // Per-expert weights (an unconverted checkpoint) are stacked for SwitchGLU.
        for l in 0 ..< config.numHiddenLayers {
            let prefix = "model.layers.\(l).mlp"
            for m in ["gate_proj", "down_proj", "up_proj"] {
                for k in ["weight", "scales", "biases"] where w["\(prefix).experts.0.\(m).\(k)"] != nil {
                    let parts = (0 ..< (config.nRoutedExperts ?? 0)).compactMap {
                        w.removeValue(forKey: "\(prefix).experts.\($0).\(m).\(k)")
                    }
                    w["\(prefix).switch_mlp.\(m).\(k)"] = stacked(parts)
                }
            }
        }
        return w
    }

    public var loraLayers: [Module] { model.layers }
}
