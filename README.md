# Swift BERT

> BERT implementation using BNNS

```
// From the dependency swift-transformers
let hub = HubApi(downloadBase: dest)
let repoName = "google/bert_uncased_L-2_H-128_A-2"
let modelDir = try await hub.snapshot(from: repoName)
let config = try hub.configuration(fileURL: modelDir.appending(path: "config.json"))

let weights = try Weights.from(fileURL: modelFile)
let model = BERTEmbedding(config: config, weights: weights)
let embeddings = try model(inputIDs: tokenIDs)
```
