// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

struct SpeculativeDecodingTests {

    let processor: any UserInputProcessor
    let mainContext: ModelContext
    let draftContext: ModelContext

    init() {
        let processor = TestInputProcessor()
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 8, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 8,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 2048
        )

        let mainModel = withRandomState(.init(seed: 0)) {
            Gemma3TextModel(modelConfig)
        }
        let mainContext = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let draftModel = withRandomState(.init(seed: 0)) {
            Gemma3TextModel(modelConfig)
        }
        let draftContext = ModelContext(
            configuration: processor.configuration,
            model: draftModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        eval(mainModel, draftModel)
        //        print(mainModel.lmHead.weight)
        //        print(sum(mainModel.lmHead.weight))
        //        print(draftModel.lmHead.weight)
        //        print(sum(draftModel.lmHead.weight))
        print("mm: \(mainModel.parameters().mapValues { sum($0).item(Float.self) }.reduce(0, +))")
        print("dm: \(draftModel.parameters().mapValues { sum($0).item(Float.self) }.reduce(0, +))")
        self.processor = processor
        self.mainContext = mainContext
        self.draftContext = draftContext
    }

    @Test(arguments: [48], [false])
    func `Speculative decoding matches default token generation`(
        numDraftTokens: Int,
        withLogitProcessor: Bool
    ) async throws {
        let input = UserInput(prompt: "Input text")
        // TODO dkoski -- outside for fixed tokens, inside for different tokens
        // let modelInput = try await processor.prepare(input: input)
        for i in 0 ..< 100 {
            print("iter \(i)")
            let modelInput = try await processor.prepare(input: input)
            let parameters = GenerateParameters(
                maxTokens: 32,
                temperature: 0.0,  // Use greedy decoding for deterministic output
                repetitionPenalty: withLogitProcessor ? 1.5 : nil,
                presencePenalty: withLogitProcessor ? 0.5 : nil,
                frequencyPenalty: withLogitProcessor ? 0.2 : nil,
            )

            var normalTokens: [Int] = []
            for await generation in try generateTokens(
                input: modelInput, parameters: parameters, context: mainContext
            ) {
                if let token = generation.token { normalTokens.append(token) }
            }

            var speculativeTokens: [Int] = []
            for await generation in try generateTokens(
                input: modelInput, parameters: parameters, context: mainContext,
                draftModel: draftContext.model, numDraftTokens: numDraftTokens
            ) {
                if let token = generation.token { speculativeTokens.append(token) }
            }

            print("input: \(modelInput.text.tokens.asArray(Int.self))")
            print("normal output: \(normalTokens))")

            #expect(!normalTokens.isEmpty)
            #expect(!speculativeTokens.isEmpty)
            #expect(normalTokens == speculativeTokens)
        }
    }
}
