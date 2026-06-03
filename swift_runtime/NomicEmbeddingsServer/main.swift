import ArgumentParser
import Foundation
import HTTPTypes
import Hub
import Hummingbird
import MLX
import MLXEmbedders
import MLXLMCommon
import Tokenizers

private struct HubDownloader: Downloader, Sendable {
    let hub: HubApi
    func download(
        id: String, revision: String?, matching patterns: [String], useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        try await hub.snapshot(from: id, matching: patterns, progressHandler: progressHandler)
    }
}

private struct TransformersTokenizerLoader: TokenizerLoader, Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TransformersTokenizerBridge(try await AutoTokenizer.from(modelFolder: directory))
    }
}

private struct TransformersTokenizerBridge: MLXLMCommon.Tokenizer, Sendable {
    let upstream: any Tokenizers.Tokenizer
    init(_ upstream: any Tokenizers.Tokenizer) { self.upstream = upstream }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }
    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }
    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try upstream.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}

private struct EmbeddingsRequest {
    let input: [String]
    let model: String?
    let dimensions: Int?

    init(data: Data) throws {
        guard let payload = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ValidationError("Expected JSON object")
        }
        if let value = payload["input"] as? String {
            input = [value]
        } else if let values = payload["input"] as? [String] {
            input = values
        } else {
            throw ValidationError("input must be a string or string array")
        }
        model = payload["model"] as? String
        dimensions = payload["dimensions"] as? Int
    }
}

private struct ModelData: Encodable {
    let id: String
    let object = "model"
    let ownedBy = "lightrag"
    enum CodingKeys: String, CodingKey {
        case id, object
        case ownedBy = "owned_by"
    }
}

private struct ModelsResponse: Encodable {
    let object = "list"
    let data: [ModelData]
}

private struct EmbeddingData: Encodable {
    let object = "embedding"
    let index: Int
    let embedding: [Float]
}

private struct Usage: Encodable {
    let promptTokens: Int
    let totalTokens: Int
    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }
}

private struct EmbeddingsResponse: Encodable {
    let object = "list"
    let data: [EmbeddingData]
    let model: String
    let usage: Usage
}

private struct HealthResponse: Encodable {
    let status: String
    let model: String
    let loaded: Bool
    let activeRequests: Int
    let idleTimeoutS: Int

    enum CodingKeys: String, CodingKey {
        case status, model, loaded
        case activeRequests = "active_requests"
        case idleTimeoutS = "idle_timeout_s"
    }
}

private func jsonResponse<T: Encodable>(
    _ value: T, status: HTTPResponse.Status = .ok
) throws -> Response {
    let payload = try JSONEncoder().encode(value)
    return Response(
        status: status,
        headers: HTTPFields([HTTPField(name: .contentType, value: "application/json")]),
        body: .init(byteBuffer: ByteBuffer(data: payload)))
}

private func collectBody(_ request: Request) async throws -> Data {
    var body = try await request.body.collect(upTo: 100 * 1024 * 1024)
    return Data(body.readBytes(length: body.readableBytes) ?? [])
}

private actor EmbeddingServerState {
    private let configuration: ModelConfiguration
    private let downloader: HubDownloader
    private let tokenizerLoader: TransformersTokenizerLoader
    private let idleTimeoutS: Int
    private let servedModelName: String

    private var container: EmbedderModelContainer?
    private var loadTask: Task<EmbedderModelContainer, Error>?
    private var activeRequests = 0
    private var lastAccess = Date()

    init(
        configuration: ModelConfiguration,
        downloader: HubDownloader,
        tokenizerLoader: TransformersTokenizerLoader,
        servedModelName: String,
        idleTimeoutS: Int
    ) {
        self.configuration = configuration
        self.downloader = downloader
        self.tokenizerLoader = tokenizerLoader
        self.servedModelName = servedModelName
        self.idleTimeoutS = max(0, idleTimeoutS)
    }

    func healthPayload() -> HealthResponse {
        HealthResponse(
            status: "ok",
            model: servedModelName,
            loaded: container != nil,
            activeRequests: activeRequests,
            idleTimeoutS: idleTimeoutS)
    }

    func withContainer<R: Sendable>(
        _ action: @Sendable @escaping (EmbedderModelContainer) async throws -> sending R
    ) async throws -> sending R {
        let loadedContainer = try await ensureLoaded()
        activeRequests += 1
        lastAccess = Date()
        defer {
            activeRequests -= 1
            lastAccess = Date()
        }
        return try await action(loadedContainer)
    }

    func unloadIfIdle() {
        guard idleTimeoutS > 0 else {
            return
        }
        guard activeRequests == 0 else {
            return
        }
        guard loadTask == nil, container != nil else {
            return
        }
        let idleFor = Date().timeIntervalSince(lastAccess)
        guard idleFor >= Double(idleTimeoutS) else {
            return
        }
        container = nil
        MLX.Memory.clearCache()
        print(
            "[NomicEmbeddingsServer] unloaded \(servedModelName) after \(Int(idleFor))s idle")
    }

    private func ensureLoaded() async throws -> EmbedderModelContainer {
        if let container {
            return container
        }
        if let loadTask {
            return try await loadTask.value
        }

        let loadTask = Task {
            try await EmbedderModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration)
        }
        self.loadTask = loadTask

        do {
            let loadedContainer = try await loadTask.value
            container = loadedContainer
            self.loadTask = nil
            lastAccess = Date()
            print("[NomicEmbeddingsServer] loaded \(servedModelName)")
            return loadedContainer
        } catch {
            self.loadTask = nil
            throw error
        }
    }
}

@main
struct NomicEmbeddingsServer: AsyncParsableCommand {
    @Option(name: .long) var model: String
    @Option(name: .long) var servedModelName: String?
    @Option(name: .long) var host = "127.0.0.1"
    @Option(name: .long) var port = 11439
    @Option(name: .long) var maxTokens = 2048
    @Option(name: .long) var idleTimeoutS = 180

    mutating func run() async throws {
        let config: ModelConfiguration
        if FileManager.default.fileExists(atPath: model) {
            config = ModelConfiguration(directory: URL(filePath: model))
        } else {
            config = ModelConfiguration(id: model)
        }
        let downloader = HubDownloader(hub: HubApi(downloadBase: URL.applicationSupportDirectory))
        let tokenizerLoader = TransformersTokenizerLoader()
        let modelID = servedModelName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? servedModelName!.trimmingCharacters(in: .whitespacesAndNewlines)
            : model
        let maxTokenCount = maxTokens
        let state = EmbeddingServerState(
            configuration: config,
            downloader: downloader,
            tokenizerLoader: tokenizerLoader,
            servedModelName: modelID,
            idleTimeoutS: idleTimeoutS)
        let idleCheckIntervalS = max(5, min(30, max(1, idleTimeoutS / 2)))
        let idleMonitorTask = idleTimeoutS > 0
            ? Task.detached {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(idleCheckIntervalS))
                    await state.unloadIfIdle()
                }
            }
            : nil
        defer { idleMonitorTask?.cancel() }

        let router = Router()
        router.get("/health") { _, _ in
            try jsonResponse(await state.healthPayload())
        }
        router.get("/v1/models") { _, _ in
            try jsonResponse(ModelsResponse(data: [ModelData(id: modelID)]))
        }
        router.post("/v1/embeddings") { request, _ in
            do {
                let decoded = try EmbeddingsRequest(data: try await collectBody(request))
                let values = decoded.input
                let requestedDimensions = decoded.dimensions
                let result = try await state.withContainer { container in
                    try await container.perform { context in
                        let tokenizer = context.tokenizer
                        let encoded = values.map {
                            Array(tokenizer.encode(text: $0, addSpecialTokens: true).prefix(maxTokenCount))
                        }
                        let maxLength = encoded.map(\.count).max() ?? 1
                        let padToken = tokenizer.convertTokenToId("<pad>") ?? 1
                        let padded = stacked(encoded.map {
                            MLXArray($0 + Array(repeating: padToken, count: maxLength - $0.count))
                        })
                        let mask = padded .!= padToken
                        let tokenTypes = MLXArray.zeros(like: padded)
                        var output = context.pooling(
                            context.model(
                                padded, positionIds: nil, tokenTypeIds: tokenTypes,
                                attentionMask: mask),
                            normalize: true, applyLayerNorm: true)
                        if let dimensions = requestedDimensions,
                           dimensions > 0,
                           dimensions < output.dim(-1)
                        {
                            output = output[0..., ..<dimensions]
                            output = output / sqrt((output * output).sum(axis: -1, keepDims: true))
                        }
                        output.eval()
                        return output.map { $0.asArray(Float.self) }
                    }
                }
                let tokenCount = values.reduce(0) { $0 + $1.count }
                return try jsonResponse(EmbeddingsResponse(
                    data: result.enumerated().map {
                        EmbeddingData(index: $0.offset, embedding: $0.element)
                    },
                    model: decoded.model ?? modelID,
                    usage: Usage(promptTokens: tokenCount, totalTokens: tokenCount)))
            } catch {
                return try jsonResponse(
                    ["error": String(describing: error)], status: .internalServerError)
            }
        }
        let app = Application(
            router: router, configuration: .init(address: .hostname(host, port: port)))
        print("[NomicEmbeddingsServer] ready http://\(host):\(port)")
        try await app.runService()
    }
}
