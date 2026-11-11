"""The reader's checks on a model's reply, table-driven over made-up replies.

The model is a stub that returns a fixed reply; everything after it is the solution's own code:
the verdict and the rules that overrule it, the clean-up of the wording, the quotes kept beside the
answer, and the pages cited. Most cases read one made-up site; those about a product that is no
longer listed read two downloads of another (helpers.SPRING and NOW). No key, no network.
"""
import re

import pytest

from helpers import GONE, NOW, SMALL_SITE, SPRING, build, has_digit, model_reply, refused, scripted_reader, with_reader
from kb.text import digits_beside_words

SPECIALS = ("# Certificate specials\n\nOur current specials are the 14-month and the 22-month Plover certificates. Open "
            "one at any Fennick branch.\n")
SITE = {
    "specials": SPECIALS,
    "contact-us": "# Contact us\n\nCall us at 555-010-4477 or write to PO Box 12, Larkspur.\n",
    "about": "# About Larkspur Valley Credit Union\n\nThe credit union was founded in 1961 by nine orchard growers who "
             "pooled their savings. Today fourteen branches serve the valley.\n\nRouting Number: 271987654\n",
    "files/forms.pdf": "Direct deposit form\n\nGive your employer our routing number 271987654 and your account number.\n",
    "youth": "# Youth savings\n\nA youth savings account is for members under 18 and pays dividends every quarter.\n",
}
FACT = "Which certificate specials do you have now?"


def read(reply, question=FACT, as_of=None):
    """The answer the solution gives when the model replies `reply`, and the prompt it sent."""
    reader = scripted_reader(reply)
    ks = build(("2026-06-10", SITE), reader=reader)
    answer = ks.answer(question, as_of)
    return answer, ks.engine.answer(question, as_of), reader.backend.prompts


# --------------------------------------------------------------------------- an answer that says the site does not say

@pytest.mark.parametrize("said", [
    "The rate for the specials is not mentioned.", "The rate for the specials is not specified.",
    "The rate for the specials is not published.", "The rate for the specials is not provided.",
    "The rate for the specials is not given.", "The rate for the specials is not shown.",
    "The rate for the specials is not disclosed.", "The rate for the specials is not currently listed.",
    "No rates are listed for the specials.", "The site doesn't say what the specials pay.",
    "The page does not specify the rate.", "The passages do not mention a rate.", "The website does not publish it.",
    "I cannot confirm the rate of the specials.", "I can't determine which specials run now.",
    "I am unable to find the specials.", "There are no details about the specials.", "There is no mention of specials.",
    "The page does not provide the rates of the specials.", "The table doesn't include the specials.",
    "The documents don't give a rate for them.",
])
def test_an_answer_that_says_the_site_does_not_state_it_is_a_refusal(said):
    answer, reply, prompts = read(model_reply(said))
    assert len(prompts) == 1 and reply.path == "declined" and refused(answer)


@pytest.mark.parametrize("said", [
    "The specials are the 14-month and the 22-month Plover certificates.",
    "Two run now: 14 and 22 months; they do not include an IRA option.",      # what a product lacks is an answer
    "The 14-month and the 22-month: there are no fees to open one.",
])
def test_an_answer_that_states_the_fact_is_kept(said):
    answer, reply, _ = read(model_reply(said))
    assert reply.path == "reader" and answer.confident and answer.text.startswith(said[:20])


@pytest.mark.parametrize("said, kept", [
    ("The specials are the 14-month and the 22-month Plover certificates. Their rates are not given.", True),
    ("Two run now, at 14 and 22 months. When they end is not stated.", True),
    ("The specials run at any Fennick branch. Their rates are not given.", False),     # no figure before the gap
    ("Their rates are not given. The specials are the 14-month and the 22-month.", False),    # it opens with the gap
    ("The specials opened in 2025. No newer list is published.", False),                # a year says when, not what
])
def test_a_figure_before_the_gap_keeps_the_answer_only_when_it_comes_first_and_is_no_year(said, kept):
    answer, reply, _ = read(model_reply(said))
    assert (reply.path == "reader" and answer.confident and answer.text.startswith(said[:20])) if kept else refused(answer)


@pytest.mark.parametrize("question, said, kept", [
    ("Which specials do you have, and what does each need to open?", "The 14-month and the 22-month Plover. "
                                                                      "No business minimum is published.", True),
    ("How much does a box of Plover checks cost?", "A stop payment is $12. The price of a box of checks is not listed.",
     False),
    ("How much is the fee to move an IRA to another credit union?", "Closing an IRA costs $25. A fee for moving an "
                                                                     "IRA to another credit union is not listed.", False),
    ("What is your SWIFT code?", "Call 555-010-4477 for wire help. The SWIFT code is not stated on the site.", False),
    ("What is the founder's salary?", "The credit union was founded in 1961 by 9 growers. His salary is not published.",
     False),
])
def test_a_figure_before_a_gap_that_restates_the_question_is_still_a_refusal(question, said, kept):
    answer, reply, _ = read(model_reply(said), question)
    assert (reply.path == "reader" and answer.confident) if kept else refused(answer)


def test_a_figure_before_the_gap_is_no_answer_when_the_rate_tables_lack_what_was_asked():
    reader = scripted_reader(model_reply("The 14-month Plover pays 2.90%. Longer terms are not listed."))
    ks = build(("2026-06-10", {"rates": "## Rates\n\nPlover Certificates\n\n| Term | APY |\n| --- | --- |\n"
                                         "| 14 Month | 2.90% |\n"}), reader=reader)
    assert refused(ks.answer("What is the APY on a 30-month certificate?"))
    assert reader.backend.prompts                                   # the reader declines it, not a rule


@pytest.mark.parametrize("question", [
    "Do you still offer a 41-month special?", "Does any branch offer a 41-month special?",
    "Is there a 41-month special?", "Are 41-month specials offered?", "Can I still open a 41-month special?",
    "Will the 41-month special come back?", "Has the 41-month special ended?", "Was there a 41-month special?",
    "Should I expect a 41-month special?", "May I open a 41-month special?",
    "Could you tell me whether a 41-month special is offered?", "Can you let me know if a 41-month special exists?",
])
def test_a_yes_or_no_question_keeps_a_no_with_what_is_listed(question):
    answer, reply, _ = read(model_reply("No, a 41-month special is not listed; the specials are 14 and 22 months."), question)
    assert reply.path == "reader" and answer.confident and answer.text.startswith("No, a 41-month special is not listed")


@pytest.mark.parametrize("question", [
    "Could you give me the rate on the 41-month special?", "Would you show me the 41-month special's rate?",
    "Will you share the rate of the 41-month special?", "Can you list the rate on the 41-month special?",
    "Could you please provide the 41-month special's rate?", "Can you send the 41-month special's rate?",
])
def test_a_polite_request_for_a_fact_is_held_to_the_rule_for_facts(question):
    answer, _, _ = read(model_reply("The rate of a 41-month special is not listed."), question)
    assert refused(answer, question)


# --------------------------------------------------------------------------- a product that is no longer listed

@pytest.mark.parametrize("gone", ["is no longer listed", "is no longer offered or listed", "is no longer available or listed",
                                  "will no longer be listed", "is no longer being listed", "is No Longer Listed"])
def test_an_answer_that_says_the_product_is_no_longer_listed_is_kept(gone):
    ks = with_reader(model_reply(f"The 41-month special {gone}. Current specials are the 14-month and the 22-month."),
                     SPRING, NOW)
    answer = ks.answer(GONE)
    assert ks.engine.answer(GONE).path == "reader" and ks.engine.reader.backend.calls == 1
    assert answer.confident and f"{gone}. Current specials are the 14-month and the 22-month" in answer.text


@pytest.mark.parametrize("said", [
    "The site does not list the rate of the 41-month special.",
    "The rate of the 41-month special is not stated.",
    "There is no information about the 41-month special.",
    "The 41-month special is not listed; the 22-month special earns 2.83% APY.",       # another product's value
    "The 41-month special is no longer listed, and its last rate is not stated.",     # the rest is still read
    "Its fee is no longer listed.",                            # "no longer listed" that gives nothing else
    "No longer listed.",
    "No longer listed. The rate cannot be confirmed.",
    "The rate of the 41-month special isn't listed.",          # "n't" inside a word, with no word boundary
    "Its rate wasn't stated.",
    "Rates for specials aren't published.",
    "The rate of the 41-month special isn\u2019t listed.",     # a curly apostrophe: the reply is normalised first
    "The site doesn\u2019t list the rate of the 41\u2011month special.",
    "Its rate wasn\u2019t\u00a0stated.",
    "The site doesn't\u2009list the rate of the 41-month special.",   # spaces clean() leaves: thin,
    "Its rate isn't\u3000stated.",                                      # ideographic,
    "There is no\u2002information about the 41-month special.",        # en
    "The rate of the 41-month special is not\u205flisted.",            # and medium mathematical
])
def test_saying_the_site_does_not_state_it_is_a_refusal_whatever_its_apostrophes_and_spaces(said):
    ks = with_reader(model_reply(said), SPRING, NOW)
    answer = ks.answer(GONE)
    assert ks.engine.reader.backend.calls == 1                 # the model answered; the check turned it down
    assert refused(answer, GONE)


@pytest.mark.parametrize("rest, kept", [
    ("", False),                                                # its one figure, 41, is the question's own
    (" The 41-month special ran in the spring.", False),        # still nothing the question does not have
    (" The 22-month special earns 2.83% APY.", True),           # a figure the question does not have
    (" Specials now open with $750.", True),
    (" Current specials are the 14-month and the 22-month.", True),
])
def test_no_longer_listed_is_an_answer_only_beside_a_figure_the_question_does_not_have(rest, kept):
    ks = with_reader(model_reply("The 41-month special is no longer listed." + rest), SPRING, NOW)
    answer = ks.answer(GONE)
    assert ks.engine.reader.backend.calls == 1
    assert answer.confident is kept and refused(answer, GONE) is not kept


NOT_LISTED = "The website does not list an APY for the 41-month special."
NOT_ANY_MORE = "No, the 41-month special is not listed; the current specials are the 14-month and the 22-month."


@pytest.mark.parametrize("question, said, kept", [
    ("What is the APY on the 41-month special?", NOT_LISTED, False),
    ("Can you tell me the APY on the 41-month special?", NOT_LISTED, False),    # asks for a fact, politely
    ("Could you please give me the APY on the 41-month special?", NOT_LISTED, False),
    ("Do you know the APY on the 41-month special?", NOT_LISTED, False),
    ("Do you still offer the 41-month special?", NOT_ANY_MORE, True),          # yes or no: "no" and what there is
    ("Can you tell me if you still offer the 41-month special?", NOT_ANY_MORE, True),
    ("Do you know whether the 41-month special is still offered?", NOT_ANY_MORE, True),
])
def test_a_polite_question_is_held_to_the_rule_for_facts_or_for_a_yes_or_no(question, said, kept):
    ks = with_reader(model_reply(said), SPRING, NOW)
    answer = ks.answer(question)
    assert ks.engine.reader.backend.calls == 1
    assert answer.confident is kept and refused(answer, question) is not kept


@pytest.mark.parametrize("said, kept", [
    ("The current specials are the 14-month and the 22-month; they do not include a debit card.", True),
    ("The current specials are the 14-month and the 22-month; we do not provide a bonus on them.", True),
    ("The evidence names the 14-month and the 22-month specials but does not provide their rates.", False),
    ("The site's rate tables do not include the 41-month special.", False),
])
def test_does_not_include_is_a_refusal_only_when_said_of_the_source(said, kept):
    # What a product lacks is an answer; what the site leaves out is not.
    ks = with_reader(model_reply(said), SPRING, NOW)
    answer = ks.answer(GONE)
    assert ks.engine.reader.backend.calls == 1
    assert answer.confident is kept and refused(answer, GONE) is not kept


# --------------------------------------------------------------------------- the contact verdict

@pytest.mark.parametrize("question", [
    "How can I open a special?", "How should I open a special?", "How to open a special?", "Can I open a special online?",
    "Could I open a special by phone?", "May I open a special for my son?", "Where do I open a special?",
    "Where can I open a special?", "What do I need to open a special?", "What should I do to open a special?",
    "Who do I call to open a special?", "Who should I call about a special?", "Help me open a special.",
    "Is there a way to open a special online?", "What's the process to open a special?",
    "What is the easiest way to open a special?", "What steps open a special?", "How do I find the nearest branch?",
])
def test_a_how_to_is_shown_the_contact_paragraph_and_gets_the_contact_route_citing_the_contact_page_once(question):
    answer, reply, prompts = read(model_reply("Call us to open one.", verdict="contact", sources="none"), question)
    assert "General contact" in prompts[0]
    assert reply.path == "reader" and answer.confident and answer.text == "Call us to open one."
    urls = [s.url for s in answer.sources]
    assert urls.count("https://cu.test/contact-us") == 1


@pytest.mark.parametrize("question", ["What rate does the 22-month special pay?", "Which specials do you have now?",
                                      "When do the specials end?", "Who runs the specials?"])
def test_a_contact_verdict_gives_the_contact_route_whatever_the_question(question):
    # A question of fact is not shown the contact paragraph, but the model may still judge the contact
    # route its answer: the verdict stands.
    answer, reply, prompts = read(model_reply("Call us to ask.", verdict="contact", sources="none"), question)
    assert "General contact" not in prompts[0]
    assert reply.path == "reader" and answer.confident and answer.text == "Call us to ask."
    assert [s.url for s in answer.sources].count("https://cu.test/contact-us") == 1


# --------------------------------------------------------------------------- a decline

@pytest.mark.parametrize("said", [
    "The specials are the 14-month and the 22-month Plover certificates. Their rates are not given.",
    "Two specials run now, at 14 and at 22 months, at any Fennick branch. When they end is not stated.",
    "",
])
def test_a_decline_is_a_refusal_that_states_no_number_whatever_it_opens_with(said):
    answer, reply, _ = read(model_reply(said, verdict="decline"))
    assert reply.path == "declined" and refused(answer) and not has_digit(answer.text)


# --------------------------------------------------------------------------- wording clean-up

@pytest.mark.parametrize("said, shown", [
    ("**Two** specials run now.", "Two specials run now."),
    ("__Two__ specials run now [1].", "Two specials run now."),
    ("Two specials run now [1, 2].", "Two specials run now."),
    ("The evidence shows that two specials run now.", "Two specials run now."),
    ("This evidence states two specials run now.", "Two specials run now."),
    ("Two specials run now in the evidence.", "Two specials run now."),
    ("two specials run now.", "Two specials run now."),
    ("Fourteen branches serve the valley.", "Fourteen (14) branches serve the valley."),
    ("The 14‑month and the 22‑month specials.", "The 14-month and the 22-month specials."),
    ("Two specials run now.", "Two specials run now."),
])
def test_the_answer_is_cleaned_of_markup_and_talk_about_the_evidence(said, shown):
    answer, _, _ = read(model_reply(said))
    assert answer.text == shown


def test_an_as_of_answer_says_which_download_it_is_from():
    answer, _, _ = read(model_reply("Two specials run now."), as_of="2026-07-01")
    assert answer.text == "As of 2026-07-01 (the website as downloaded on 2026-06-10): Two specials run now."


@pytest.mark.parametrize("said, shown", [
    ("Forty-one volunteers", "Forty-one (41) volunteers"),
    ("Twelve first-time buyers", "Twelve (12) first-time buyers"),     # counts keep their digits
    ("a fifteen-second video", "a fifteen (15)-second video"),
    ("the twenty-first annual drive", "the twenty-first annual drive"),  # part of a larger number: left alone
    ("Seventy-fifth year", "Seventy-fifth year"),
    ("ten thousand members", "ten thousand members"),
    ("Ten Thousand", "Ten Thousand"),
    ("Twenty-five hundred", "Twenty-five hundred"),
])
def test_a_number_in_words_gets_its_digits_only_when_they_are_the_figure(said, shown):
    assert digits_beside_words(said) == shown


# --------------------------------------------------------------------------- a reply in the wrong form

@pytest.mark.parametrize("raw", [
    "", None, "Two specials run now.", "VERDICT: maybe\nSOURCES: 1\nANSWER: Two.", "ANSWER: Two specials run now.",
    "QUOTES: none\nVERDICT: answer\nANSWER: Two specials run now.",           # no SOURCES line
    "<html><body>Bad gateway</body></html>",
])
def test_a_reply_not_in_the_form_asked_is_no_reply_and_the_rules_answer(raw):
    answer, reply, prompts = read(raw)
    assert len(prompts) == 1 and reply.path in ("quote", "declined", "contact", "link")
    assert isinstance(answer.text, str) and isinstance(answer.confident, bool)


def test_a_reply_without_a_quotes_line_is_read_all_the_same():
    answer, reply, _ = read("VERDICT: answer\nSOURCES: 1\nANSWER: Two specials run now.")
    assert reply.path == "reader" and answer.text == "Two specials run now."


@pytest.mark.parametrize("verdict", ["ANSWER", "Answer", "answer"])
def test_the_verdict_is_read_in_any_case(verdict):
    answer, reply, _ = read(f"QUOTES: none\nVERDICT: {verdict}\nSOURCES: 1\nANSWER: Two specials run now.")
    assert reply.path == "reader" and answer.text == "Two specials run now."


def test_an_answer_verdict_with_an_empty_answer_is_a_refusal():
    answer, _, _ = read(model_reply(""))
    assert refused(answer)


# --------------------------------------------------------------------------- quotes

QUOTE = "Our current specials are the 14-month and the 22-month Plover certificates."


@pytest.mark.parametrize("quotes, shown", [
    (f'"{QUOTE}" [1]', QUOTE),                                     # on the page: kept beside the answer
    (f"“{QUOTE}” [1]", QUOTE),                           # typographic quote marks
    (f'"**{QUOTE}**" [1]', QUOTE),                                 # emphasis
    ('"Our current specials are the 15-month and the 22-month Plover certificates." [1]', None),   # not on the page
    ('"Open one at any branch." [1]', None),                       # shorter than 25 characters
    (f'"{QUOTE * 5}" [1]', None),                                  # longer than 320 characters
])
def test_a_quote_is_kept_only_when_it_is_on_the_page_word_for_word(quotes, shown):
    answer, _, _ = read(model_reply("Two specials run now: 14 and 22 months.", quotes=quotes))
    assert answer.text == "Two specials run now: 14 and 22 months." + (f' The site says: "{shown}"' if shown else "")
    assert answer.sources[0].quote == shown


def test_a_quote_the_answer_already_contains_is_not_repeated():
    answer, _, _ = read(model_reply(QUOTE, quotes=f'"{QUOTE}" [1]'))
    assert answer.text == QUOTE


def test_a_quote_with_a_figure_the_answer_does_not_have_is_not_shown():
    answer, _, _ = read(model_reply("Two specials run now.", quotes=f'"{QUOTE}" [1]'))
    assert answer.text == "Two specials run now."                      # 14 and 22 are not in the answer


def test_no_more_than_two_quotes_are_shown():
    quotes = " ".join(f'"{q}" [1]' for q in (
        "The credit union was founded in 1961 by nine orchard growers who pooled their savings.",
        "A youth savings account is for members under 18 and pays dividends every quarter.",
        "Give your employer our routing number 271987654 and your account number."))
    answer, _, _ = read(model_reply("It was founded in 1961; youth savings is for under 18; routing 271987654.", quotes=quotes),
                        "When was it founded, who can open youth savings, and what is the routing number?")
    assert answer.text.count('"') == 4


# --------------------------------------------------------------------------- which pages are cited

def numbered(prompt):
    """{passage number: page url} as the prompt numbers them."""
    return {int(n): url for n, url in re.findall(r"^\[(\d+)\] [^|\n]*\| (\S+)$", prompt, re.M)}


@pytest.mark.parametrize("cite", ["1", "2", "2, 1", "3,1"])
def test_the_passages_the_model_names_are_cited_first_in_its_order(cite):
    answer, _, prompts = read(model_reply("Two specials run now.", sources=cite),
                              "Which specials do you have now, and who can open youth savings?")
    pages = numbered(prompts[0])
    want = [pages[int(n)] for n in re.findall(r"\d+", cite)]
    assert [s.url for s in answer.sources][:len(want)] == want


def test_a_passage_number_that_was_not_shown_is_ignored():
    answer, _, _ = read(model_reply("Two specials run now.", sources="99, 0"))
    assert answer.confident and all(s.url.startswith("https://cu.test/") for s in answer.sources)


def test_a_number_given_in_the_answer_cites_each_page_that_prints_it_once():
    question = "Which number do I give my employer for direct deposit?"
    prompt = read(model_reply("Nothing."), question)[2][0]
    about = next(n for n, url in numbered(prompt).items() if url.endswith("/about"))
    answer, reply, _ = read(model_reply("Give them 271987654.", sources=str(about)), question)
    assert reply.path == "reader"
    urls = [s.url for s in answer.sources]
    assert len(urls) == len(set(urls)) and {"https://cu.test/about", "https://cu.test/files/forms.pdf"} <= set(urls)
    assert "https://cu.test/youth" not in urls


@pytest.mark.parametrize("rates", [
    "## Rates\n\nTern Mortgages\n\n| Term | APR |\n| --- | --- |\n| 30 Year | 6.10% |\n| 15 Year | 5.40% |\n",
    # The answer gives the row's second value, its APR, not its first.
    "## Rates\n\nTern Mortgages\n\n| Term | Rate | APR |\n| --- | --- | --- |\n| 30 Year | 5.90% | 6.10% |\n"
    "| 15 Year | 5.20% | 5.40% |\n",
])
def test_a_value_from_the_rows_cites_every_page_that_prints_its_row(rates):
    reader = scripted_reader(model_reply("For a first home, the 30-year Tern mortgage is 6.10% APR.", sources="none"))
    buyers = "# First-time buyers\n\nA first-time buyer's home loan guide: what a first-time buyer needs to know.\n"
    ks = build(("2026-06-10", {"rates": rates, "loans/tern": "# Tern\n\nOur table.\n\n" + rates,
                               "buyers": buyers, "learn/buyers": buyers.replace("guide", "checklist")}), reader=reader)
    answer = ks.answer("Which home loan suits a first-time buyer?")
    assert ks.engine.answer("Which home loan suits a first-time buyer?").path == "reader"
    urls = [s.url for s in answer.sources]
    assert {"https://cu.test/rates", "https://cu.test/loans/tern"} <= set(urls)


def test_a_link_given_in_the_answer_is_cited():
    answer, _, _ = read(model_reply("See https://cu.test/youth for youth savings.", sources="none"),
                        "Which page explains youth savings?")
    assert "https://cu.test/youth" in [s.url for s in answer.sources]


def test_an_answer_cites_no_more_than_ten_pages():
    pages = dict(SITE, **{f"branches/b{n}": f"# Branch {n}\n\nBranch {n} prints our routing number 271987654.\n"
                          for n in range(12)})
    reader = scripted_reader(model_reply("The routing number is 271987654.", sources="1, 2, 3"))
    answer = build(("2026-06-10", pages), reader=reader).answer("What is the routing number?")
    assert 1 <= len(answer.sources) <= 10 and len({s.url for s in answer.sources}) == len(answer.sources)


def test_only_the_newest_of_the_dated_articles_shown_is_tagged_the_newest():
    news = "1 min read\n\n{day}\n\n# Plover news\n\nPlover certificate specials opened at every Fennick branch {when}.\n"
    pages = dict(SITE, **{"news/plover-spring": news.format(day="March 2, 2026", when="this spring"),
                          "news/plover-winter": news.format(day="January 5, 2026", when="this winter")})
    reader = scripted_reader(model_reply("Nothing."))
    build(("2026-06-10", pages), reader=reader).answer("When did the Plover certificate specials open?")
    tags = re.findall(r"^\[\d+\] (article[^|]*)\|", reader.backend.prompts[0], re.M)
    assert sorted(t.strip() for t in tags) == ["article published 2026-01-05",
                                                "article published 2026-03-02 (the newest article here)"]


# --------------------------------------------------------------------------- the rules first, the model only when they cannot answer


def test_a_table_lookup_never_calls_the_model():
    ks = with_reader("VERDICT: answer\nSOURCES: 1\nANSWER: nonsense", SMALL_SITE)
    assert "3.50%" in ks.answer("6 month certificate APY?").text
    assert ks.engine.reader.backend.calls == 0


def test_model_decline_becomes_a_plain_refusal_with_no_numbers():
    ks = with_reader("QUOTES: none\nVERDICT: decline\nSOURCES: none\nANSWER: The site lists 2 pages but not that.", SMALL_SITE)
    answer = ks.answer("Who is the chief executive?")
    assert not answer.confident and not any(ch.isdigit() for ch in answer.text)
