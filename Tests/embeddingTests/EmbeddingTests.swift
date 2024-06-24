import XCTest
import Hub
@testable import Embedding

final class EmbeddingTests: XCTestCase {

    let hub: HubApi = {
        let dest = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!.appending(component: "huggingface-tests")
        return HubApi(downloadBase: dest)
    }()

    func testInferenceBERT128() async throws {
        let repoName = "google/bert_uncased_L-2_H-128_A-2"
        let modelDir = try await hub.snapshot(from: repoName)
        let modelConfig = try hub.configuration(fileURL: modelDir.appending(path: "config.json"))

        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.isReadableKey])
        XCTAssertTrue(files.contains(where: { $0.lastPathComponent == "config.json" }))
        XCTAssertTrue(files.contains(where: { $0.lastPathComponent == "model.safetensors" }))

        let modelFile = modelDir.appending(path: "model.safetensors")
        let weights: Weights = try Weights.from(fileURL: modelFile)
        
        let model = BERTEmbedding(config: modelConfig, weights: weights)
        XCTAssertNoThrow(try model(inputIDs: [1, 2, 3, 4, 5, 6]))
    }
}
