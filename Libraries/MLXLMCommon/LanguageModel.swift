// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Abstract form of a model that processes language.
public protocol BaseLanguageModel: Module {
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?
    public let audio: ProcessedAudio?

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Time, height, and width of the images
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Representation of prepared input audio(s).
    public struct ProcessedAudio {

        public let samples: MLXArray

        public init(
            samples: MLXArray
        ) {
            self.samples = samples
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text,
        image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audio: LMInput.ProcessedAudio? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
        self.audio = audio
    }
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    /// typed key for use in ``State``
    public struct Key<T>: Identifiable, Sendable {
        public let id: String

        public init(_ id: String) {
            self.id = id
        }
    }

    /// Dictionary of typed ``Key`` to carry state between steps.
    public struct State {
        private var contents: [String: Any]

        public init() {
            self.contents = [:]
        }

        public subscript<T>(_ key: Key<T>) -> T? {
            get {
                contents[key.id] as? T
            }
            set {
                contents[key.id] = newValue
            }
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:windowSize:)``
public enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:windowSize:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol LanguageModel: BaseLanguageModel {

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// Speculative-decoding-aware forward. Additionally reports, via
    /// `confirmedPrefix`, how many of the leading tokens in `inputs` are
    /// already confirmed — the linear-attention (recurrent) layers use this as
    /// their snapshot boundary (see ``MambaCache/markSpeculationBoundary()``
    /// and MTP speculative decoding, ch. 7B).
    ///
    /// This is purely additive: the existing ``callAsFunction(_:cache:)``
    /// keeps its exact signature and behaviour, and this overload's default
    /// implementation just forwards to it with `hidden: nil` — so every model
    /// in the fork conforms unchanged. Only the Qwen3.5/3.6 text path honours
    /// `confirmedPrefix`.
    ///
    /// `hidden` is the backbone's pre-final-norm hidden state, needed by
    /// ``MTPSpeculativeModel/mtpForward(hiddenState:nextTokenIds:cache:)``.
    /// `nil` for models without an MTP head. Shapes are otherwise unchanged:
    /// input `[B, L]`, `logits` `[B, L, V]`.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, confirmedPrefix: Int) -> (
        logits: MLXArray, hidden: MLXArray?
    )

    /// Whether this model can speculate, and why not if it can't — computed
    /// once at load. `nil` ⇒ serve non-speculatively (checkpoint has no MTP
    /// head). Never a per-request decision; no caller may override this.
    var speculationCapability: SpeculationCapability? { get }

    /// create a new array of ``KVCache``: automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
    func newCache(parameters: GenerateParameters?) -> [KVCache]
}

extension LanguageModel {
    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?, confirmedPrefix: Int) -> (
        logits: MLXArray, hidden: MLXArray?
    ) {
        (callAsFunction(inputs, cache: cache), nil)
    }

    public var speculationCapability: SpeculationCapability? { nil }
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

// MARK: - MTP speculative decoding

/// Describes whether a loaded model can speculate, and if not, why — in a form
/// loggable verbatim. Computed **once at load** (never per-request): capability
/// is a property of the checkpoint, not a policy.
///
/// `nil` from ``LanguageModel/speculationCapability`` means "serve
/// non-speculatively", the same as an unavailable capability — callers should
/// not need to distinguish the two.
public struct SpeculationCapability: Sendable {
    /// The checkpoint ships a usable MTP head (declared module + weights loaded).
    public let hasHeads: Bool
    /// Every cache this model allocates (`newCache(parameters:)`) can be
    /// restored to an earlier position — trimmable or snapshot-restorable.
    /// A single non-restorable cache disqualifies the whole model (all-or-nothing).
    public let allCachesRestorable: Bool
    /// Default speculation depth (tokens proposed per round). The reference
    /// value is 1 (Part-B §6) — deeper speculation is a tunable, not a default.
    public let proposalDepth: Int
    /// Loggable, human-readable reason when `isAvailable` is `false`. `nil`
    /// when available.
    public let reason: String?

    public init(
        hasHeads: Bool, allCachesRestorable: Bool, proposalDepth: Int = 1,
        reason: String? = nil
    ) {
        self.hasHeads = hasHeads
        self.allCachesRestorable = allCachesRestorable
        self.proposalDepth = proposalDepth
        self.reason = reason
    }

    /// Whether this model can actually speculate right now. Fail-closed: both
    /// conditions must hold.
    public var isAvailable: Bool { hasHeads && allCachesRestorable }
}

/// Conformed by models with a native multi-token-prediction (MTP) head (e.g.
/// Qwen3.5/3.6). Separate from ``LanguageModel`` because it is not something
/// every model implements — callers downcast (`model as? MTPSpeculativeModel`)
/// after checking ``LanguageModel/speculationCapability``.
public protocol MTPSpeculativeModel {
    /// A fresh KV cache for the MTP head's own transformer layer(s). Independent
    /// of the backbone's cache array.
    func makeMTPCache() -> [KVCache]

    /// Run the MTP head and apply the shared `lm_head`/embedding-as-linear, the
    /// same way the backbone does. `hiddenState` is the backbone's **pre-final-
    /// norm** hidden state (see the `hidden` component of
    /// ``LanguageModel/callAsFunction(_:cache:confirmedPrefix:)``); `nextTokenIds`
    /// is the already-sampled token(s) that follow it. Returns logits,
    /// `[B, N, vocab]`.
    func mtpForward(hiddenState: MLXArray, nextTokenIds: MLXArray, cache: [KVCache]) -> MLXArray
}

/// Conformed by models that can load their MTP head from a **separate**
/// checkpoint directory, alongside (not instead of) a fused head already
/// present in the main checkpoint's own weights.
///
/// Exists because the checkpoint we actually serve
/// (`lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit`) ships **no** `mtp.*`
/// weights at all — every MLX conversion strips them — while
/// `mlx-community` publishes the head as its own standalone model directory
/// (e.g. `Qwen3.6-35B-A3B-MTP-bf16`: a `config.json` + one `model.safetensors`,
/// keys unprefixed relative to what a fused checkpoint's `mtp.*` keys would
/// be). `attachSeparateMTPHead` reads that directory directly and applies its
/// weights to the SAME module shape (`Qwen35MTPHead`/`Qwen35MTPDecoderLayer`)
/// the fused path already declares — this is an additional source for the
/// head, not a second mechanism.
public protocol MTPHeadAttachable {
    /// Attempts to attach an MTP head loaded from `directory`. A no-op
    /// (returns `true`) if a fused head is already present — the fused
    /// checkpoint takes precedence, never duplicated or replaced. Fails
    /// closed (returns `false`, logs the reason once, throws nothing) on any
    /// incompatibility: no `mtp_num_hidden_layers` in the directory's config,
    /// a `hiddenSize` mismatch with the main model, no `.safetensors` found,
    /// or a load/shape error.
    @discardableResult
    func attachSeparateMTPHead(from directory: URL) -> Bool
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count

        // Follow Python logic: use RotatingKVCache if maxKVSize is provided
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}
