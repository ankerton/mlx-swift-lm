// Copyright © 2026 Apple Inc.
//
// MTP (multi-token-prediction) native speculative decoding — ch. 7B of the
// ai-server component doc; reference to port: ml-explore/mlx-lm#990.
//
// This file is the single-stream (batch-of-one) driver, `MTPSpeculativeTokenIterator`,
// plus the accept/reject rule shared with the batched driver in mlx-chatd's
// `BatchEngine`. It sits beside — and does not modify — the existing classic
// draft-model `SpeculativeTokenIterator` in Evaluate.swift (Part-D §3: that
// type is reference/precedent, not this task's target).

import Foundation
import MLX
import MLXNN

// MARK: - Operator toggle (Part-C §2.3)

/// Process-wide operator toggle for native MTP speculative decoding. Read
/// **once at startup** by `mlx-chatd` from an environment variable / CLI flag
/// and set here — never a per-request decision, never exposed on any
/// request/response surface (the hard limit in Part-C §2.3: no request
/// field, no per-model `inferd.toml` policy, no HTTP surface).
///
/// `ChatSession` (single-stream) and `BatchEngine` (batched, mlx-chatd) both
/// consult this before attempting native speculation. Default `true`
/// ("auto": speculate wherever the loaded checkpoint supports it) — with
/// `enabled = false`, behaviour is byte-identical to today on both engines.
///
/// `nonisolated(unsafe)`: this is intentionally process-global mutable state,
/// matching an environment variable's own nature — set once at startup,
/// before any request-handling `Task` reads it, and not mutated concurrently
/// with reads in normal operation.
public enum MTPSpeculationToggle {
    nonisolated(unsafe) public static var enabled: Bool = true
}

// MARK: - Accept/reject rule

/// Distribution-preserving accept/reject for MTP speculative decoding
/// (Part-B §1: "Acceptance semantics must preserve the output distribution").
///
/// At `temperature == 0` this is exact-match — the byte-parity gate (DoD 1)
/// and unambiguous.
///
/// At `temperature > 0`, naive equality silently biases output. The correct
/// rule, `accept with probability min(1, p_target / p_draft)`, and on
/// rejection resample from the residual `max(p_target - p_draft, 0) / Z`
/// (Leviathan et al. 2022 §2.3), is ported directly from the reference this
/// task ports, ml-explore/mlx-lm#990 (`mtp_generate_step`'s probabilistic
/// acceptance).
///
/// **Flagged discrepancy (Part-D §10 — routed back, not papered over):** Part-C
/// §1.5 directs this rule be taken from "the semantics already implemented in
/// the fork's `SpeculativeTokenIterator` accept/reject path
/// (`Evaluate.swift:896-904`)". On inspection at the pinned rev `4c5e335`,
/// those lines implement only greedy exact-match equality
/// (`mainTokensList[i] == draftTokensList[i]`) — there is no
/// temperature-aware/probabilistic logic there to mirror. Since Part-C's own
/// next sentence states "naive equality at temperature is wrong and silently
/// biases output", mirroring those lines literally would reproduce the exact
/// defect Part-C warns against. This implementation instead ports the
/// probabilistic rule from the actual reference (PR #990). Temp=0 (the gate)
/// is identical either way. Temp>0 is a smoke run only (Part-B §8), not
/// gated — see the accompanying report.
public enum SpeculativeAcceptance {

    /// - Parameters:
    ///   - draftToken: the token the MTP head proposed.
    ///   - targetLogProbs: backbone log-probabilities at this position, `[V]`.
    ///   - draftLogProbs: MTP head log-probabilities at this position, `[V]`,
    ///     under the same processor filtering as `targetLogProbs`.
    ///   - isGreedy: `temperature == 0`.
    ///   - uniformDraw: a fresh `Uniform(0,1)` scalar draw; ignored when `isGreedy`.
    public static func accept(
        draftToken: Int,
        targetLogProbs: MLXArray,
        draftLogProbs: MLXArray,
        isGreedy: Bool,
        uniformDraw: Float
    ) -> Bool {
        if isGreedy {
            let targetArgmax = argMax(targetLogProbs, axis: -1).item(Int.self)
            return targetArgmax == draftToken
        }
        let logAccept =
            targetLogProbs[draftToken].item(Float.self) - draftLogProbs[draftToken].item(Float.self)
        return logAccept >= 0 || uniformDraw < Foundation.exp(logAccept)
    }

    /// Residual-distribution correction token on rejection (temp > 0 path
    /// only — the greedy/temp=0 path uses the backbone's own argmax instead).
    /// Guarantees the marginal output distribution exactly equals the target.
    public static func residualToken(targetLogProbs: MLXArray, draftLogProbs: MLXArray) -> Int {
        let pTarget = exp(targetLogProbs)
        let pDraft = exp(draftLogProbs)
        let residual = maximum(pTarget - pDraft, MLXArray(0))
        let z = residual.sum()
        let dist = z.item(Float.self) > 0 ? residual : pTarget
        return categorical(log(dist).reshaped([1, -1]), axis: -1).item(Int.self)
    }
}

// MARK: - Single-stream (batch-of-one) driver

/// Generator of tokens using native MTP speculative decoding — the
/// batch-of-one case of the one shared restoration mechanism (Part-A §5,
/// Part-B §2). `BatchEngine` (mlx-chatd) drives the ragged N>1 case directly
/// against the same model/cache primitives; this iterator is what
/// `ChatSession` selects for the single-stream (`MLXEngine`) path.
///
/// Reference depth: proposes exactly **one** token ahead per round (Part-B
/// §6's default) — the MTP head predicts `t+2` from the hidden state at `t`
/// and the embedding of `t+1`, so the natural unit of work is one draft + one
/// bonus token per verify pass.
public struct MTPSpeculativeTokenIterator: TokenIteratorProtocol {
    let model: any LanguageModel & MTPSpeculativeModel
    var mainCache: [KVCache]
    var mtpCache: [KVCache]

    /// The last confirmed (already-emitted) token — the seed for the next round.
    var confirmed: MLXArray
    /// Backbone pre-final-norm hidden state at `confirmed`'s position — what
    /// the MTP head fuses with the next token's embedding to propose a draft.
    var lastHidden: MLXArray

    /// A still-unverified draft proposal, if one is pending.
    var draftToken: MLXArray?
    var draftLogProbs: MLXArray?

    var processor: LogitProcessor?
    let sampler: LogitSampler
    let isGreedy: Bool

    public var tokenCount = 0
    public let maxTokens: Int?
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    public var promptPrefillTime: TimeInterval = 0.0

    /// Acceptance telemetry (Part-B §7, Part-C §2.4): proposals accepted ÷
    /// proposals made, the self-diagnosing signal for a broken restoration.
    public private(set) var proposedCount = 0
    public private(set) var acceptedCount = 0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Set when a `rollbackToBoundary` mismatch makes the cache untrustworthy
    /// (Part-C §4: fatal for speculation). `next()` is not `throws`, so this
    /// iterator fails the request the only way it can from here: it stops
    /// emitting further tokens (ending the stream, as if EOS had been hit)
    /// rather than emitting from state it cannot trust. It never emits the
    /// unverified token itself.
    private var finished = false

    public init(
        input: LMInput,
        model: any LanguageModel & MTPSpeculativeModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.mainCache = cache ?? model.newCache(parameters: parameters)
        guard canRestoreCache(self.mainCache) else {
            throw KVCacheError(
                message:
                    "MTP speculative decoding requires every cache to be restorable (trimmable or snapshot-capable)."
            )
        }
        self.mtpCache = model.makeMTPCache()

        self.sampler = parameters.sampler()
        self.processor = parameters.processor()
        self.isGreedy = parameters.temperature == 0

        self.maxTokens = parameters.maxTokens
        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart

        // Placeholders — `prepare` sets both to real values below before this
        // initializer returns. (Required because `confirmed`/`lastHidden` are
        // non-optional `let`-like state that `prepare` must fill in, and Swift
        // needs every stored property initialized before `self` is fully
        // formed; `prepare` runs as part of `init`, immediately after.)
        self.confirmed = MLXArray(0)
        self.lastHidden = MLXArray.zeros([1, 1, 1])

        // `Evaluate.swift`'s `measure(_:)` helper is `private` to that file
        // (file-scoped, not module-scoped), so it isn't reachable here — this
        // inlines the identical measurement rather than widening that
        // helper's access (Part-D §3: additive, don't touch existing code
        // beyond what's needed).
        let start = Date()
        try prepare(input: input, windowSize: parameters.prefillStepSize)
        self.promptPrefillTime = Date().timeIntervalSince(start)
    }

    mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
        processor?.prompt(input.text.tokens)

        switch try model.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            // One more confirmed-only step (confirmedPrefix irrelevant — no
            // draft pending yet) to get both the first emitted token and the
            // hidden state needed to propose the first draft.
            let (token, hidden) = confirmedStep(previous: tokens)
            confirmed = token
            lastHidden = hidden
            pendingTokens = [token.item(Int.self)]
            asyncEval(confirmed)

        case .logits(let result):
            // `LLMModel.prepare` (what Qwen3.5/3.6 actually uses) always
            // returns `.tokens` — chunked prefill, never pre-computed logits.
            // This branch exists only for protocol completeness (e.g. a
            // future model overriding `prepare` differently); it re-derives
            // the hidden state with one extra confirmed-only step so this
            // iterator degrades to "one redundant forward", never to
            // incorrect output.
            let token = convertToToken(logits: result.logits)
            let (_, hidden) = confirmedStep(previous: .init(tokens: token))
            confirmed = token
            lastHidden = hidden
            pendingTokens = [token.item(Int.self)]
            asyncEval(confirmed)
        }
    }

    /// Run the backbone on exactly one already-confirmed token
    /// (`confirmedPrefix` doesn't matter — there is no draft in this call).
    mutating func confirmedStep(previous: LMInput.Text) -> (token: MLXArray, hidden: MLXArray) {
        let inputTokens = previous.tokens.reshaped([1, previous.tokens.dim(0)])
        let (logits, hidden) = model(
            inputTokens, cache: mainCache, confirmedPrefix: 0)
        maybeQuantizeKVCache(
            cache: &mainCache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart)
        let token = convertToToken(logits: logits[0..., -1, 0...])
        guard let hidden else {
            fatalError(
                "MTPSpeculativeTokenIterator requires a hidden state from the confirmedPrefix-aware call; got nil for \(type(of: model))."
            )
        }
        return (token, hidden[0..., (hidden.dim(1) - 1)..., 0...])
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        return token
    }

    /// Propose the next draft token from the MTP head, given the hidden state
    /// and the token that follows it positionally.
    mutating func proposeDraft(hidden: MLXArray, afterToken: MLXArray) -> (
        token: MLXArray, logProbs: MLXArray
    ) {
        let mtpLogits = model.mtpForward(
            hiddenState: hidden, nextTokenIds: afterToken.reshaped([1, 1]), cache: mtpCache)
        maybeQuantizeKVCache(
            cache: &mtpCache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart)
        var logits = mtpLogits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let logProbs = logSoftmax(logits, axis: -1)
        let token = sampler.sample(logits: logits)
        return (token, logProbs)
    }

    /// One speculative round: ensure a draft exists, verify it against the
    /// backbone in one pass, accept or reject, restore cache state exactly to
    /// the accepted boundary, and buffer the emitted token(s).
    mutating func speculateRound() {
        if draftToken == nil {
            let (tok, lp) = proposeDraft(hidden: lastHidden, afterToken: confirmed)
            draftToken = tok
            draftLogProbs = lp
        }
        guard let draftTok = draftToken, let draftLP = draftLogProbs else {
            finished = true
            return
        }

        let verifyInput = concatenated([confirmed.reshaped([1]), draftTok.reshaped([1])])
        let (rawLogits, hiddenOpt) = model(
            verifyInput.reshaped([1, 2]), cache: mainCache, confirmedPrefix: 1)
        maybeQuantizeKVCache(
            cache: &mainCache, kvBits: kvBits, kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart)
        guard let hidden = hiddenOpt else {
            finished = true
            return
        }

        proposedCount += 1

        var verifyLogits0 = rawLogits[0..., 0, 0...]
        verifyLogits0 = processor?.process(logits: verifyLogits0) ?? verifyLogits0
        let verifyLogProbs0 = logSoftmax(verifyLogits0, axis: -1)
        let draftTokenId = draftTok.item(Int.self)

        let accepted = SpeculativeAcceptance.accept(
            draftToken: draftTokenId,
            targetLogProbs: verifyLogProbs0,
            draftLogProbs: draftLP,
            isGreedy: isGreedy,
            uniformDraw: isGreedy ? 0 : uniform().item(Float.self)
        )

        if accepted {
            acceptedCount += 1
            commitBoundary(mainCache)
            processor?.didSample(token: draftTok)

            var bonusLogits = rawLogits[0..., 1, 0...]
            bonusLogits = processor?.process(logits: bonusLogits) ?? bonusLogits
            let bonusTok = sampler.sample(logits: bonusLogits)
            processor?.didSample(token: bonusTok)

            pendingTokens.append(draftTokenId)
            pendingTokens.append(bonusTok.item(Int.self))

            confirmed = bonusTok.reshaped([1])
            lastHidden = hidden[0..., 1..<2, 0...]
        } else {
            let discarded = rollbackToBoundary(mainCache, discarding: 1)
            guard discarded == 1 else {
                // Part-C §4: a rollback mismatch means the cache state is no
                // longer trustworthy. Fail closed: stop here rather than emit
                // from corrupt state. (See `finished`'s doc comment — this
                // `next()` cannot throw, so ending the stream is the fail
                // mechanism available at this layer.)
                FileHandle.standardError.write(
                    Data(
                        "mtp speculative decoding: rollbackToBoundary discarded \(discarded) of 1 requested — cache state untrustworthy, ending generation early.\n"
                            .utf8))
                finished = true
                return
            }

            let replacementTok: MLXArray
            if isGreedy {
                replacementTok = argMax(verifyLogits0, axis: -1)
            } else {
                let residual = SpeculativeAcceptance.residualToken(
                    targetLogProbs: verifyLogProbs0, draftLogProbs: draftLP)
                replacementTok = MLXArray(UInt32(residual)).reshaped([1])
            }
            processor?.didSample(token: replacementTok)
            pendingTokens.append(replacementTok.item(Int.self))

            confirmed = replacementTok.reshaped([1])
            // Position-0 hidden state is unaffected by the (now rolled back)
            // draft token — attention is causal, so it's still valid to seed
            // the next draft proposal.
            lastHidden = hidden[0..., 0..<1, 0...]
        }

        draftToken = nil
        draftLogProbs = nil

        // Acceptance rate — recorded, not just measurable (Part-B §7): a
        // collapsed rate is the self-diagnosing sign restoration is broken.
        // Logged to stderr periodically; `proposedCount`/`acceptedCount` are
        // also public for a caller to read directly (e.g. the DoD gate script).
        if proposedCount % 50 == 0 {
            let rate = proposedCount > 0 ? Double(acceptedCount) / Double(proposedCount) : 0
            FileHandle.standardError.write(
                Data(
                    "mlx-chatd: mtp acceptance \(acceptedCount)/\(proposedCount) (\(String(format: "%.1f", rate * 100))%, single-stream)\n"
                        .utf8))
        }
    }

    mutating public func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            tokenCount += 1
            return token
        }

        guard !finished else { return nil }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        if pendingTokens.isEmpty {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }
}
