import re

from extractor import CandidateFact

SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "cool", "lol"}
JUNK_PREDICATES = {"greeting", "thanks", "mood", "feeling", "small_talk"}
SECRET_PREDICATES = {"password", "api_key", "card_number", "cvv", "pin", "iban"}

SECRET_PATTERNS = {
    "api_key": re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}|xox[bp]-[A-Za-z0-9-]{10,})"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}
CARD_LIKE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(digits))
    return total % 10 == 0


def has_card(text: str) -> bool:
    return any(luhn_ok(m.group()) for m in CARD_LIKE.finditer(text))


def redact(turn: str) -> str:
    """Runs before the LLM, so secrets never leave our server."""
    for name, pattern in SECRET_PATTERNS.items():
        turn = pattern.sub(f"[{name.upper()}]", turn)
    return CARD_LIKE.sub(lambda m: "[CARD_NUMBER]" if luhn_ok(m.group()) else m.group(), turn)


def reject_reason(fact: CandidateFact, user_name: str) -> str | None:
    all_text = f"{fact.subject} {fact.predicate} {fact.value} {fact.text}"

    if fact.predicate in SECRET_PREDICATES:
        return "secret"
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(all_text):
            return name
    if has_card(all_text):
        return "card_number"

    if fact.predicate in JUNK_PREDICATES or fact.value.strip().lower() in SMALL_TALK:
        return "small_talk"

    if fact.subject.strip().lower() != user_name.lower():
        return "not_about_user"

    return None


def gate(facts: list[CandidateFact], user_name: str) -> list[CandidateFact]:
    """Runs after the LLM. Secret checks stay here too, as a backup for anything redact() missed."""
    kept = []
    for fact in facts:
        reason = reject_reason(fact, user_name)
        if reason:
            print(f"  dropped ({reason}): {fact.text}")
        else:
            kept.append(fact)
    return kept


if __name__ == "__main__":
    from datetime import date

    from extractor import extract_facts

    for turn in [
        "Remember my OpenAI key is sk-proj-abc123def456ghi789",
        "My card is 4111 1111 1111 1111",
        "My brother Omar loves spicy food",
        "Actually I'm vegan now, not just vegetarian",
        "thanks!",
    ]:
        safe_turn = redact(turn)
        print(f"\n{turn}\n  sent to LLM: {safe_turn}")
        for fact in gate(extract_facts(safe_turn, "Ali", date(2026, 4, 5)), "Ali"):
            print(f"  kept: {fact.text}")

    print("\nHand-made facts (in case the LLM skips them):")
    fakes = [
        CandidateFact(subject="Ali", predicate="payment_card", value="4111-1111-1111-1111", text="Ali's card is 4111-1111-1111-1111."),
        CandidateFact(subject="Ali", predicate="openai_key", value="sk-proj-abc123def456", text="Ali's key is sk-proj-abc123def456."),
        CandidateFact(subject="Ali", predicate="mood", value="annoyed", text="Ali is annoyed today."),
        CandidateFact(subject="Omar", predicate="likes", value="spicy food", text="Omar likes spicy food."),
        CandidateFact(subject="Ali", predicate="phone_order_id", value="1234567890123", text="Ali's order id is 1234567890123."),
        CandidateFact(subject="Ali", predicate="diet", value="vegan", text="Ali is vegan."),
    ]
    for fact in gate(fakes, "Ali"):
        print(f"  kept: {fact.text}")
