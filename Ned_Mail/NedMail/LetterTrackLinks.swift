import Foundation

enum LetterTrackLinks {
    static let awaitingShipping = URL(string: "https://www.ebay.com/sh/ord/?filter=status:AWAITING_SHIPMENT")!
    static let lettertrackpro = URL(string: "https://www.lettertrackpro.com/Process_Mail.asp")!

    static func trackingURL(for number: String) -> URL? {
        let trimmed = number.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let encoded = trimmed.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed)
        else { return nil }
        return URL(string: "https://www.lettertrackpro.com/uspstracking/?TrackingNumber=\(encoded)")
    }
}

enum MessageBuilder {
    static func make(tinyURL: String) -> String {
        """
        Hello, and thank you for your business. I'm using a mail-tracking service called LetterTrackPro. You can track your order with this link: \(tinyURL)

        If you have any questions, please don't hesitate to reach out.
        Geoff
        """
    }
}
