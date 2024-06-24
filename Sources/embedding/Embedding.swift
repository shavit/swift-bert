import Hub
import Tokenizers
import CoreML
import Accelerate

public class BERTEmbedding {
    
    let shape: [NSNumber]

    private let weights: Weights

    private let positionEmbeddingType = "absolute"
    private let hiddenSize: Int = 128
    private let vocabSize: Int = 30522
    private let maxPositionEmbeddings: Int = 512
    private let typeVocabSize: Int = 2
    private let padTokenID: Int = 0
    private let normalizationEpsilon: Float = 1e-12
    private let dropoutRate: Float = 1e-1
    /// 0.5f * x * (1.0f + tanh(alpha * (x + beta * x * x * x)))
    private let hiddenActivation: BNNS.ActivationFunction = .geluApproximation2(alpha: 0.7978845608028654, beta: 44715e-6)

    private var allocations: [BNNSNDArrayDescriptor] = []

    private lazy var wordEmbedding: BNNS.EmbeddingLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int64.self, shape: .vector(maxPositionEmbeddings))
        allocations.append(input)
        let dictData: [Float32] = weights["bert.embeddings.word_embeddings.weight"]!.floats!
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(hiddenSize, vocabSize))
        allocations.append(dict)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)

        return BNNS.EmbeddingLayer(input: input, output: output, dictionary: dict, paddingIndex: 0, maximumNorm: 0, normType: .l2, scalesGradientByFrequency: false)!
    }()

    private lazy var positionEmbedding: BNNS.EmbeddingLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int64.self, shape: .vector(maxPositionEmbeddings))
        allocations.append(input)
        let dictData: [Float32] = weights["bert.embeddings.position_embeddings.weight"]!.floats!
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(dict)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)

        return BNNS.EmbeddingLayer(input: input, output: output, dictionary: dict, paddingIndex: -1, maximumNorm: 0, normType: .l2, scalesGradientByFrequency: false)!
    }()

    private lazy var tokenTypeEmbedding: BNNS.EmbeddingLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Int64.self, shape: .vector(maxPositionEmbeddings))
        allocations.append(input)
        let dictData: [Float32] = weights["bert.embeddings.token_type_embeddings.weight"]!.floats!
        let dict = BNNSNDArrayDescriptor.allocate(initializingFrom: dictData, shape: .matrixColumnMajor(hiddenSize, typeVocabSize))
        allocations.append(dict)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)

        return BNNS.EmbeddingLayer(input: input, output: output, dictionary: dict, paddingIndex: -1, maximumNorm: 0, normType: .l2, scalesGradientByFrequency: false)!
    }()

    private lazy var normalization: BNNS.NormalizationLayer = {
        let inputShape: BNNS.Shape = .imageCHW(1, hiddenSize, maxPositionEmbeddings)
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: inputShape)
        allocations.append(input)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: inputShape)
        allocations.append(output)
        let betaW: [Float32] = (weights["bert.embeddings.LayerNorm.beta"] ?? weights["bert.embeddings.LayerNorm.bias"])!.floats!
        let betaWA = Array(repeating: betaW, count: maxPositionEmbeddings).flatMap({ $0 })
        let beta = BNNSNDArrayDescriptor.allocate(initializingFrom: betaWA, shape: inputShape)
        allocations.append(beta)
        let gammaW: [Float32] = (weights["bert.embeddings.LayerNorm.gamma"] ?? weights["bert.embeddings.LayerNorm.weight"])!.floats!
        let gammaWA = Array(repeating: gammaW, count: maxPositionEmbeddings).flatMap({ $0 })
        let gamma = BNNSNDArrayDescriptor.allocate(initializingFrom: gammaWA, shape: inputShape)
        allocations.append(gamma)

        return BNNS.NormalizationLayer(type: .layer(normalizationAxis: 0), input: input, output: output, beta: beta, gamma: gamma, epsilon: normalizationEpsilon, activation: hiddenActivation)!
    }()

    private lazy var dropout: BNNS.DropoutLayer = {
        let input = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(input)
        let output = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        allocations.append(output)

        return BNNS.DropoutLayer(input: input, output: output, rate: dropoutRate, seed: 0, control: 0)!
    }()

    deinit {
        allocations.forEach({ $0.deallocate() })
    }

    init(config: Config, weights: Weights) {
        self.weights = weights
        self.shape = [NSNumber(value: maxPositionEmbeddings), NSNumber(value: hiddenSize)]
    }

    public func callAsFunction(inputIDs: [Int64],
                               tokenTypeIDs: [Int64]? = nil,
                               positionIDs: [Int64]? = nil,
                               phase: BNNS.LearningPhase = .inference) throws -> MLMultiArray {
        let inputLength = inputIDs.count
        let inputIDs: [Int64] = inputIDs.padded(length: maxPositionEmbeddings)
        let wordInput = BNNSNDArrayDescriptor.allocate(initializingFrom: inputIDs, shape: .vector(inputIDs.count))
        let wordOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, inputIDs.count))
        defer {
            wordInput.deallocate()
            wordOutput.deallocate()
        }
        try wordEmbedding.apply(batchSize: 1, input: wordInput, output: wordOutput)

        let positionIDs = positionIDs ?? Array<Int64>(stride(from: 0, through: Int64(inputLength - 1), by: 1))
        let positionInput = BNNSNDArrayDescriptor.allocate(initializingFrom: positionIDs.padded(length: maxPositionEmbeddings), shape: .vector(maxPositionEmbeddings))
        let positionOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        defer {
            positionInput.deallocate()
            positionOutput.deallocate()
        }
        try self.positionEmbedding.apply(batchSize: 1, input: positionInput, output: positionOutput)

        let tokenTypeIDs: [Int64] = tokenTypeIDs ?? Array(repeating: 0, count: maxPositionEmbeddings)
        let typeInput = BNNSNDArrayDescriptor.allocate(initializingFrom: tokenTypeIDs, shape: .vector(maxPositionEmbeddings))
        let typeOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
        defer {
            typeInput.deallocate()
            typeOutput.deallocate()
        }
        try self.tokenTypeEmbedding.apply(batchSize: 1, input: typeInput, output: typeOutput)

        let multiWord = try wordOutput.makeMultiArray(of: Float32.self, shape: shape)
        let multiPosition = try positionOutput.makeMultiArray(of: Float32.self, shape: shape)
        let multiType = try typeOutput.makeMultiArray(of: Float32.self, shape: shape)

        if phase == .inference {
            return multiWord + multiPosition + multiType
        } else {
            let normInputArray = (multiWord + multiPosition + multiType).floats!.padded(length: maxPositionEmbeddings)
            let normInput = BNNSNDArrayDescriptor.allocate(initializingFrom: normInputArray, shape: .imageCHW(1, hiddenSize, maxPositionEmbeddings))
            let normOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .imageCHW(1, hiddenSize, maxPositionEmbeddings))
            defer {
                normInput.deallocate()
                normOutput.deallocate()
            }
            try normalization.apply(batchSize: 1, input: normInput, output: normOutput, for: phase)

            let dropoutInput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
            let dropoutOutput = BNNSNDArrayDescriptor.allocateUninitialized(scalarType: Float32.self, shape: .matrixColumnMajor(hiddenSize, maxPositionEmbeddings))
            defer {
                dropoutInput.deallocate()
                dropoutOutput.deallocate()
            }
            try dropout.apply(batchSize: 1, input: dropoutInput, output: dropoutOutput)

            return try dropoutOutput.makeMultiArray(of: Float32.self, shape: shape)
        }
    }
}
