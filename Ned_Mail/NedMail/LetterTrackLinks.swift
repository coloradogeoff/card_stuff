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
        Hello, and thank you for your business. Your order has been shipped via US First Class Mail. I've included a USPS Informed Visibility tracking link so you can follow its progress through the postal system:

        \(tinyURL)

        If you have any questions, I'm happy to help.
        Geoff
        """
    }
}
