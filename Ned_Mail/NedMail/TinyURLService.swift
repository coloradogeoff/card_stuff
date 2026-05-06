import Foundation

enum TinyURLError: Error, LocalizedError {
    case missingToken
    case requestFailed(Int, String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .missingToken:
            return "Missing TinyURL token. Set TINYURL_TOKEN env var or create ~/.tinyurl-token.txt"
        case .requestFailed(let code, let body):
            let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
            return "TinyURL request failed (HTTP \(code))" + (trimmed.isEmpty ? "" : ": \(trimmed)")
        case .invalidResponse:
            return "TinyURL returned an unexpected response."
        }
    }
}

enum TinyURLService {
    static func loadToken() throws -> String {
        if let env = ProcessInfo.processInfo.environment["TINYURL_TOKEN"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
           !env.isEmpty {
            return env
        }

        let candidates = [
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".tinyurl-token.txt"),
        ]
        for url in candidates {
            if let data = try? Data(contentsOf: url),
               let raw = String(data: data, encoding: .utf8) {
                let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { return trimmed }
            }
        }
        throw TinyURLError.missingToken
    }

    static func shorten(_ longURL: URL) async throws -> String {
        let token = try loadToken()
        var request = URLRequest(url: URL(string: "https://api.tinyurl.com/create")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload: [String: Any] = ["url": longURL.absoluteString]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw TinyURLError.invalidResponse
        }
        if !(200..<300).contains(http.statusCode) {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw TinyURLError.requestFailed(http.statusCode, body)
        }

        struct Wrapper: Decodable {
            struct Inner: Decodable { let tiny_url: String }
            let data: Inner
        }
        let decoded = try JSONDecoder().decode(Wrapper.self, from: data)
        return decoded.data.tiny_url
    }
}
