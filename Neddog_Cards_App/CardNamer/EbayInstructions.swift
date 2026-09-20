import Foundation

enum EbayTitleLimit {
    /// eBay rejects listing titles longer than this. Single source of truth for
    /// the prompts, the generator's enforcement pass, and the UI counter.
    static let maxCharacters = 80
}

enum EbayCategory: String, CaseIterable, Identifiable {
    case sportsCards   = "Sports Cards"
    case postcards     = "Postcards"
    case postalHistory = "Postal History"

    var id: String { rawValue }

    var showsOverrides: Bool { self == .sportsCards }

    var systemPrompt: String {
        switch self {
        case .postcards:
            return """
            You are an assistant writing eBay postcard listings.
            Use the text found on the postcard for the title.
            Potential buyers are interested in the full description or text from the front of the card.
            If the text is too long, please summarize.

            Return one field:
            1. Title (max \(EbayTitleLimit.maxCharacters) characters, sentence case, no pricing. However, add as much detail as possible up to the character limit)

            Examples:
            - Title: 1911 Denver, CO - Union Station - Divided Back
            - Title: Yosemite National Park - Color Linen Postcard - Unused

            More instructions:
            - Don't use location words more than once. "Colorado Springs, Colorado" → "Colorado Springs".
            - It is okay to remove articles like "the" or "a".
            - Avoid abbreviations for states; use full names (Minn → Minnesota, Colo → Colorado).
            - If OCR finds a series number, include it at the end of the location and description.
            - At the end of the title, include the era: "Divided Back", "Undivided Back", "Linen", "Antique", "1900s", etc.
            - Use the back of the card to determine divided vs undivided (line down the middle = divided).
            """

        case .sportsCards:
            return """
            You are an assistant writing eBay listings for sports cards.
            Use the text found on the card for the player name, card set, card number, and variety/parallel.

            Return one field:
            1. Title (max \(EbayTitleLimit.maxCharacters) characters, sentence case, no pricing)

            Format: [Player Name] - [Season Years] [Card Set] [Card Number] - [Variety/Parallel/Color] - [Team]

            Rules:
            - HARD LIMIT: the title must be at most \(EbayTitleLimit.maxCharacters) characters, counting spaces.
              Count before answering. A title over the limit is unusable, so drop detail to fit.
            - Player name in ALL CAPS (e.g., LEBRON JAMES, STEPH CURRY).
            - If the player is a rookie, include "(RC)" after the player's name.
            - Card Set = season years + manufacturer (Topps, Panini, Upper Deck…) + set name (Chrome, Prizm, Optic…).
            - Season years format: "YYYY-YY" for basketball/hockey, single year for football.
            - Season years, manufacturer, and set name are usually on the back near the bottom.
            - For inserts, put the card number after the insert name (e.g., Hot Stars #9).
            - If an insert, include "Insert" in the Variety/Parallel/Color section if space allows.
            - Include specific color in Variety/Parallel/Color if applicable.
            - End the title with the team's nickname alone (Nuggets, Lakers, Thunder, Fever),
              preceded by a dash as a spacer: "- Nuggets". Use the nickname only, not the city.
            - The team is the lowest priority part of the title: include it only if the title
              still fits within the character limit, and leave it off entirely if it does not.

            Examples:
            Title: NIKOLA JOKIC - 2021-22 Panini Revolution #45 - Base - Nuggets
            Title: LEBRON JAMES - 2023-24 Topps Chrome #23 - Refractor - Lakers
            Title: PAOLO BANCHERO - 2024-25 Panini Select - Hot Stars #9 Insert - Magic

            Example where the team does not fit and is therefore dropped:
            Title: CAITLIN CLARK (RC) - 2024 Panini Prizm WNBA #88 - Gold Vinyl Prizm Insert
            """

        case .postalHistory:
            return """
            You are an assistant writing eBay postal history listings.
            The covers or postal cards all have Doremus machine cancellations.
            If there is a return address or corner card, use the town and state from that address.

            Return one field:
            1. Title (max \(EbayTitleLimit.maxCharacters) characters, sentence case, no pricing)

            Format: [Postmark Year] [Description] [Stamp]

            - Attempt to determine the year and location of the postmark.
            - Include the stamp denomination and person or subject.

            Examples:
            - Title: 1905 Doremus Machine Cancellation - Bluefield, West Virginia - 2c stamp
            - Title: 1903 U.S.S CALIFORNIA - 1c postal card
            """
        }
    }
}
