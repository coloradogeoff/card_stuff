import Foundation

enum LetterTrackLinks {
    static let awaitingShipping = URL(string: "https://www.ebay.com/sh/ord/?filter=status:AWAITING_SHIPMENT")!
    static let lettertrackpro = URL(string: "https://www.lettertrackpro.com/Process_Mail.asp")!

    static func trackingURL(for number: String) -> URL? {
        let trimmed = number.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)
        else { return nil }
        return URL(string: "https://neddog.com/t/\(encoded)")
    }
}

enum MessageBuilder {
    static func make(tinyURL: String) -> String {
        """
        Hello, and thank you for your business. Your order has been shipped via USPS First-Class Mail. I use LetterTrackPro rather than eBay's tracking system; the link below lets you follow its progress through USPS:

        \(tinyURL)

        Questions welcomed!
        Geoff
        """
    }
}
