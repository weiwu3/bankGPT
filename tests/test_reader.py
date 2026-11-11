"""What the reader shows the model, and what it makes of a reply, on a made-up site.

The model is a scripted reply: everything around it is the solution's own code. The tests here pin
what the model is shown (calculator figures, press-release copies, fine print, dates), the quotes
kept beside an answer, how a question is read (its typography, a wish, the year it asks about), and
what is said when a year or a rate is not published. test_reader_checks.py pins the other checks on
a reply, table-driven, and test_transport.py the requests sent to the API. No website snapshots, no
API key and no network.
"""
import time

import pytest

import scoring
from helpers import ASK, GONE, NOW, REFUSAL, SITE, SPRING, model_reply, refused, with_reader
from kb import reader
from kb.engine import _no_rate
from kb.reply import figures


def test_a_calculator_s_figures_are_not_given_to_the_reader_as_values_in_force():
    calculator = ("# Savings calculator\n\n## Savings breakdown\n\n| Point | Deposit | Dividends earned |\n"
                  "| --- | --- | --- |\n| Point 1 | $8,000 | $36 |\n| Point 2 | $16,000 | $72 |\n")
    ks = with_reader(model_reply("Nothing."), ("2026-06-10", dict(NOW[1], **{"calculators/savings": calculator})))
    ks.answer("What bonus dividend did savings members get last year?")
    [prompt] = ks.engine.reader.backend.prompts
    assert "Published values" not in prompt                 # its passages still come, tagged as illustrative


def test_an_older_press_release_s_copy_of_a_statement_stays_out_of_the_evidence():
    # Every release repeats the "About us" paragraph; only the newest copy is evidence, even when the
    # older release is the best page for the question and the rest of it is shown.
    about = ("Founded by mill workers, the Valley cooperative now counts {} owners in twelve river towns and offers "
             "savings, loans and plain advice to anyone who lives, works or studies in the valley. Its owners elect its "
             "board and share in its earnings every year, and its offices stay open six days a week.")
    old = ("1 min read\n\nMarch 2, 2024\n\n# Elmwood branch opens\n\nThe Elmwood branch opened its doors to members this "
           "week with a ribbon cutting and coffee for every visitor.\n\n## About us\n\n" + about.format("120,000") + "\n")
    new = ("1 min read\n\nMarch 2, 2026\n\n# New mobile app\n\nOur new app lets members deposit checks and pay bills "
           "from anywhere at any hour.\n\n## About us\n\n" + about.format("125,000") + "\n")
    ks = with_reader(model_reply("Nothing."), ("2026-06-10", {"newsroom/elmwood-opens": old, "newsroom/new-app": new}))
    ks.answer("When did the Elmwood branch open?")
    [prompt] = ks.engine.reader.backend.prompts
    assert "Elmwood branch opened its doors" in prompt and "120,000" not in prompt


def test_a_paragraph_of_fine_print_reaches_the_model_whole():
    # One line of about 1,300 characters is one passage: its last sentence must not be cut off.
    fine = " ".join(f"Clause {n} of this offer applies to every line of credit opened this year." for n in range(17))
    page = f"# Home equity\n\n## Fine print\n\n{fine} Flood cover is needed on every pledged home.\n"
    ks = with_reader(model_reply("Nothing."), ("2026-06-10", {"home-equity": page}))
    ks.answer("Is flood cover needed for a home equity line?")
    [prompt] = ks.engine.reader.backend.prompts
    assert len(fine) > 1100 and "Flood cover is needed on every pledged home." in prompt


def test_a_date_worked_out_from_a_change_is_not_given_as_an_effective_date():
    # The table changed between two undated downloads: it took effect after the first one, a lower
    # bound that the site never printed.
    old, new = (f"# Rates\n\nCertificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | {apy} |\n" for apy in ("2.95%", "3.50%"))
    ks = with_reader(model_reply("Nothing."), ("2026-01-05", {"rates": old}), ("2026-03-05", {"rates": new}))
    ks.answer("Tell me about your certificates.")
    [prompt] = ks.engine.reader.backend.prompts
    assert "- Certificates, 6 Month: 3.50% APY\n" in prompt
    ks.engine.reader = None
    assert ks.answer("What is the APY on a 6-month certificate?").text == "Certificates, 6 Month: 3.50% APY."


def test_a_quote_is_shown_word_for_word_and_brings_no_new_figure():
    # "twenty" in the quote is last year's count: given digits after the check, it would be a new figure.
    page = ("2026-06-10", {"foundation": "# Foundation\n\nThis year the foundation awards 25 scholarships of $1,000 "
                                         "each. Last year the foundation awarded twenty scholarships to local students.\n"})
    reply = ('QUOTES: "Last year the foundation awarded twenty scholarships to local students." [1]\n'
             "VERDICT: answer\nSOURCES: 1\nANSWER: The foundation awards 25 scholarships of $1,000 each this year.")
    answer = with_reader(reply, page).answer("How many scholarships does the foundation award?").text
    assert answer == ("The foundation awards 25 scholarships of $1,000 each this year. "
                      'The site says: "Last year the foundation awarded twenty scholarships to local students."')


@pytest.mark.parametrize("quoted", ["In the spring you had a 41-month certificate special",     # the member's question
                                    "Today is 2026-06-10, the date of the latest download"])   # the message's own line
def test_a_quote_from_the_question_or_the_message_itself_is_not_the_site_s(quoted):
    reply = (f'QUOTES: "{quoted}" [1]\nVERDICT: answer\nSOURCES: 1\n'
             "ANSWER: The 41-month special is no longer listed. Current specials are the 14-month and the 22-month.")
    answer = with_reader(reply, SPRING, NOW).answer(GONE)
    assert answer.confident and "The site says" not in answer.text
    assert all(s.quote is None for s in answer.sources)


def test_each_quote_goes_with_the_page_it_was_copied_from():
    site = ("2026-06-10", {"specials": NOW[1]["specials"],
                           "youth": "# Youth savings\n\nA youth savings account is for members under 18 and pays "
                                    "dividends quarterly.\n"})
    reply = ('QUOTES: "A youth savings account is for members under 18 and pays dividends quarterly." [2] '
             '"Our current specials are the 14-month and the 22-month certificates." [1]\n'
             "VERDICT: answer\nSOURCES: 1, 2\nANSWER: Two specials run now, and youth savings is for members under 18.")
    answer = with_reader(reply, site).answer("Which specials do you have now, and who can open youth savings?")
    quotes = {s.url.rsplit("/", 1)[-1]: s.quote for s in answer.sources}
    assert quotes["youth"].startswith("A youth savings account") and quotes["specials"].startswith("Our current specials")


# The page of NOW set with typographic characters: non-breaking hyphens, a registered mark, a curly
# apostrophe and a non-breaking space. What clean() makes of it is PLAIN_SENTENCE.
TYPESET_SENTENCE = ("Our current specials are the 14\u2011month and the 22\u2011month Maple\u00ae certificates; "
                    "they\u2019re open at\u00a0any Maple Falls branch.")
PLAIN_SENTENCE = ("Our current specials are the 14-month and the 22-month Maple certificates; "
                  "they're open at any Maple Falls branch.")
NOW_TYPESET = ("2026-06-10", {"specials": f"# Certificate specials\n\n{TYPESET_SENTENCE}\n"})
NOW_PLAIN = ("2026-06-10", {"specials": f"# Certificate specials\n\n{PLAIN_SENTENCE}\n"})
# The same with thin spaces, which clean() leaves as they are: the reply's become plain spaces, and
# its quote must still be found in the evidence, which keeps them.
THIN_SENTENCE = PLAIN_SENTENCE.replace("Maple Falls", "Maple\u2009Falls").replace("22-month", "22-month\u2009")
NOW_THIN = ("2026-06-10", {"specials": f"# Certificate specials\n\n{THIN_SENTENCE}\n"})


@pytest.mark.parametrize("page, quoted", [
    (NOW_PLAIN, "\u201cOur current specials are the 14\u2011month and the 22\u2011month Maple  certificates; "
                "they\u2019re open at any Maple Falls branch.\u201d"),      # the model's typography, a double space
    (NOW_TYPESET, f'"{TYPESET_SENTENCE}"'),                                   # copied exactly from a typeset page
    (NOW_TYPESET, f'"{PLAIN_SENTENCE}"'),                                     # copied plainly from a typeset page
    (NOW_THIN, f'"{THIN_SENTENCE}"'),                                         # copied exactly, thin spaces and all
])
def test_a_quote_is_matched_to_the_evidence_whatever_its_typography(page, quoted):
    reply = f"QUOTES: {quoted} [1]\nVERDICT: answer\nSOURCES: 1\nANSWER: Two specials run now: 14 and 22 months."
    ks = with_reader(reply, SPRING, page)
    answer = ks.answer(GONE)
    assert ks.engine.answer(GONE).path == "reader" and answer.confident
    # The quote was found in the evidence: it is kept, in its normalised wording, beside the answer.
    assert answer.text == f'Two specials run now: 14 and 22 months. The site says: "{PLAIN_SENTENCE}"'
    assert answer.sources[0].url == f"{SITE}/specials" and answer.sources[0].quote == PLAIN_SENTENCE


# The site of SPRING and NOW, now with a contact page: a how-to's message carries its paragraph.
REACH = ("2026-06-10", dict(NOW[1], **{"contact-us": "# Contact us\n\nCall us at 555-010-4477 or visit any branch.\n"}))
CONTACT_REPLY = "QUOTES: none\nVERDICT: contact\nSOURCES: none\nANSWER: Ask at any branch about that."


def shows_contact(asked):
    """Whether the message for `asked` carries the site's contact paragraph: whether it reads as a how-to."""
    ks = with_reader(CONTACT_REPLY, SPRING, REACH)
    ks.answer(asked)
    [prompt] = ks.engine.reader.backend.prompts
    return "General contact" in prompt


# One question set four ways: a curly apostrophe, a non-breaking space, doubled spaces. The reader's
# checks of the question (here: is it a "how do I", whose message carries the contact paragraph) must
# read them all alike.
WAYS = ["What's the best way to open one of the specials?", "What\u2019s the best way to open one of the specials?",
        "What's the\u00a0best way to open one of the specials?", "What's  the best way to open one of  the specials? "]


def test_a_question_is_read_the_same_whatever_its_apostrophes_and_spaces():
    reply = "QUOTES: none\nVERDICT: contact\nSOURCES: none\nANSWER: Ask at any branch to open one."
    seen = []
    for asked in WAYS:
        ks = with_reader(reply, SPRING, REACH)
        answer = ks.answer(asked)
        seen.append((answer.text, answer.confident, ks.engine.reader.backend.prompts))
    assert seen[0][:2] == ("Ask at any branch to open one.", True)
    assert "General contact" in seen[0][2][0]
    assert all(x == seen[0] for x in seen)                       # the same answer from the same prompt
    # Asked all four ways of one solution, it is one question: one answer kept, one call to the model.
    ks = with_reader(reply, SPRING, REACH)
    assert len({ks.answer(asked).text for asked in WAYS}) == 1
    assert len(ks.engine._answers) == 1 and ks.engine.reader.backend.calls == 1


@pytest.mark.parametrize("asked", ["I'd like to open one of the specials.", "I\u2019d like to open one of the specials.",
                                   "I would like to open one of the specials.", "I want to open one of the specials.",
                                   "I need to open one of the specials."])
def test_a_wish_to_get_something_done_is_a_how_to_however_it_is_worded(asked):
    # A "how do I ..." is shown the contact paragraph; "I'd like to" asks the same as "I would like to".
    assert shows_contact(asked)


def test_a_wish_to_know_a_fact_is_not_a_how_to():
    # "I'd like to know <fact>" asks for the fact, as "Can you tell me" does: no contact paragraph. It is a
    # how-to only when what it wants to know is one.
    assert not shows_contact("I'd like to know the APY on the 41-month special.")
    assert shows_contact("I'd like to know how to open an account.")


@pytest.mark.parametrize("asked, how_to", [
    ("I need to know what to do if my card is missing.", True),
    ("I'd like to know who to ring about a missing card.", True),
    ("I want to know who I should ring about a missing card.", True),
    ("I would like to know what number to ring about a missing card.", True),
    ("I'd like to know the best way to reach a branch.", True),
    ("I'd like to find out the APY on the 41-month special.", False)])
def test_a_wish_to_know_what_to_do_or_whom_to_call_is_a_how_to(asked, how_to):
    # Who to call or what to do is a how-to however the wish is worded. "Find out" asks for a fact as
    # "know" does. (The questions say "ring": "call" alone brings the contact paragraph to any question.)
    assert shows_contact(asked) is how_to


# A credit union that counts its volunteers each year and has no count yet for last year (2025).
VOLUNTEERS = ("2026-06-10", {"volunteering": "# Volunteering\n\nIn 2024, 41 staff members used paid volunteer days. "
                                            "In 2023, 37 staff members used them. In 2025 our staff gave 900 hours.\n"})
LAST_YEAR = "How many staff members used paid volunteer days last year?"
LATEST = "No count is published for 2025. The latest are 41 in 2024 and 37 in 2023."


@pytest.mark.parametrize("said", [LATEST, "No count of the staff members who used paid volunteer days is published "
                                          "for 2025; in 2024, 41 of them used them.",
                                   "There is no count for 2025. The latest are 41 (2024) and 37 (2023).",
                                   "No count is published for 2025; the latest are 41 in 2024 and 37 in 2023."])
@pytest.mark.parametrize("verdict", ["answer", "decline"])
def test_the_latest_years_figures_for_a_year_without_one_are_an_answer(verdict, said):
    # A count for 2024 is not superseded by 2025: it stays 2024's, and answers "last year" with its year.
    reply = f"QUOTES: none\nVERDICT: {verdict}\nSOURCES: 1\nANSWER: {said}"
    answer = with_reader(reply, VOLUNTEERS).answer(LAST_YEAR)
    assert answer.confident and answer.text == said


@pytest.mark.parametrize("said", [
    "No count is published for 2025.",                                     # no year's figure given
    "No count is published for 2025, and none is given for 2024 either.",
    "The count for last year is not stated; 900 hours were given.",       # a figure, but no year beside it
    "No count is published for 2025. The last count was published on March 3, 2025.",   # a date is no count
    "No count is published for 2025. The 2024 count was the 3rd one.",                   # nor is a rank
    "No count is published for 2026. The latest is 41 in 2024.",           # not the year asked about
])
def test_a_year_without_a_figure_and_no_other_year_s_figure_is_a_refusal(said):
    ks = with_reader(model_reply(said), VOLUNTEERS)
    answer = ks.answer(LAST_YEAR)
    assert ks.engine.reader.backend.calls == 1 and not answer.confident
    assert figures(answer.text) <= figures(LAST_YEAR)


@pytest.mark.parametrize("asked, said", [
    # A question that names no past year asks how things stand now: the 2024 count is no answer to it.
    ("How many staff members use paid volunteer days?", LATEST),
    ("How many staff members use paid volunteer days this year?", LATEST),
    ("How many staff members will use paid volunteer days in 2027?", LATEST),
    ("How many staff members have used paid volunteer days since 2023?", LATEST),
    ("How many staff members have used paid volunteer days since 2020, through 2025?", LATEST),
    # The site has a figure later than the year asked: for a standing fact it replaces the older ones.
    ("How many members did you have in 2025?", "No member count is published for 2025. The latest are 52,000 in 2026 "
                                                "and 49,000 in 2024."),
    # The only figure beside a year is the question's own.
    ("How many of our 41 staff members used paid volunteer days in 2025?", "No count is published for 2025. In 2024, 41 did."),
])
def test_an_older_year_s_figure_does_not_answer_a_question_about_a_standing_fact(asked, said):
    ks = with_reader(model_reply(said), VOLUNTEERS)
    answer = ks.answer(asked)
    assert ks.engine.reader.backend.calls == 1 and not answer.confident
    assert figures(answer.text) <= figures(asked)


@pytest.mark.parametrize("asked, gap", [
    ("How many staff members used paid volunteer days in the past year?", "No count is published for 2025."),
    ("Over the prior year, how many staff members used paid volunteer days?", "No count is published for 2025."),
    ("A year ago, how many staff used paid volunteer days?", "No count is published for 2025."),
    ("how many staff used volunteer days last yr", "No count is published for 2025."),
    ("How many staff members used paid volunteer days in the previous year?", "No count is published for the previous year."),
    ("How many staff members used paid volunteer days in the past year?", "There is no count for the past year."),
])
def test_a_past_year_in_other_words_gets_the_latest_years_figures(asked, gap):
    # "The past year", "the prior year", "a year ago" and "last yr" all mean 2025 here, as "last year" does,
    # and the gap sentence may name the year in the question's own words.
    said = f"{gap} The latest are 41 in 2024 and 37 in 2023."
    answer = with_reader(f"QUOTES: none\nVERDICT: decline\nSOURCES: 1\nANSWER: {said}", VOLUNTEERS).answer(asked)
    assert answer.confident and answer.text == said


@pytest.mark.parametrize("about", ["In 2025 our staff gave 900 hours.", "Last year our staff gave 900 hours.",
                                   "(For 2025 the site gives only hours: 900.)"])
@pytest.mark.parametrize("verdict", ["answer", "decline"])
def test_a_sentence_about_the_asked_year_itself_is_dropped_from_the_latest_years_figures(verdict, about):
    # Asked for a count, the reply also gives the year's hours: not the count, and no later value, so it goes.
    said = f"No count is published for 2025. {about} The latest are 41 in 2024 and 37 in 2023."
    answer = with_reader(f"QUOTES: none\nVERDICT: {verdict}\nSOURCES: 1\nANSWER: {said}", VOLUNTEERS).answer(LAST_YEAR)
    assert answer.confident and answer.text == LATEST


def test_a_sentence_that_names_a_year_after_the_one_asked_still_makes_a_refusal():
    said = "No count is published for 2025. In 2025 and 2026 our staff gave 900 hours. The latest are 41 in 2024."
    answer = with_reader(model_reply(said), VOLUNTEERS).answer(LAST_YEAR)
    assert not answer.confident


@pytest.mark.parametrize("asked, years", [
    ("How many in 2024?", [(2024, "2024", False)]),
    ("How many last year?", [(2025, "last year", False)]),
    ("Last yr, how many?", [(2025, "Last yr", False)]),
    ("How many in the past year?", [(2025, "past year", False)]),
    ("How many in the prior year?", [(2025, "prior year", False)]),
    ("How many a year ago?", [(2025, "a year ago", False)]),
    ("How many this year?", [(2026, "this year", False)]),
    ("How many since last year?", [(2025, "last year", True)]),
    ("How many since 2016?", [(2016, "2016", True)]),
    # Beside a year in figures, "the previous year" is the year before that one, not before today's.
    ("How many in 2024, and in the previous year?", [(2024, "2024", False)]),
    ("How many in the past few years?", []),
])
def test_the_year_a_question_asks_about_is_read_from_its_own_words(asked, years):
    assert reader.asked_years(asked, 2026) == years


LOANS = ("2026-06-10", {
    "loans": "# Bridge loans\n\nBorrow against your home while you sell it. Talk to a loan officer.\n\n"
             "# Adjustable-rate mortgages\n\nOur 7/1 adjustable-rate mortgage is fixed for seven years.\n\n"
             "# Premium checking\n\nPremium checking pays dividends on higher balances.\n\n"
             "# Harbor certificates\n\nHarbor certificates lock in a rate.\n",
    "rates": "# Rates\n\nAuto Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | 4.85% |\n\n"
             "Harbor Certificates\n\n| Term | APY |\n| --- | --- |\n| 12 Month | 2.90% |\n"})


@pytest.mark.parametrize("asked, words", [
    ("What interest rate do you charge on a bridge loan?", "bridge loans"),
    ("bridge loan apr?", "bridge loans"),
    ("What's the APR on your Bridge Loans?", "Bridge Loans"),
    ("Would you show me the bridge loan's rate?", "bridge loans"),
    ("What is your rate on a 7/1 adjustable-rate mortgage?", "7/1 adjustable-rate mortgages"),
    ("What APY does premium checking pay?", "premium checking"),
    ("What is the APY on a 13-month harbor certificate?", None),  # the product has rates, not this term: say nothing false
])
def test_a_rate_the_site_does_not_publish_is_refused_in_the_question_s_words(asked, words):
    ks = with_reader(None, LOANS)
    answer = ks.answer(asked)
    assert ks.engine.reader.backend.calls == 0                 # the rules decided it
    assert refused(answer, asked) and answer.text == (f"The website does not publish a rate for {words}. {ASK}" if words else REFUSAL)



def test_no_rate_is_said_to_be_missing_when_a_passage_found_by_the_product_s_words_states_one():
    # The question's other words can keep the passage out of the evidence; the claim would still be false.
    ks = with_reader(None, ("2026-06-10", {**LOANS[1], "equity": "# Home equity\n\nHome equity loans are as low as 6.10% APR.\n"}))
    view = ks.engine.store.latest
    said = _no_rate("What equity loan rates do you have?", {"equity"}, view.cache["rows"], view.cache["passages"])
    assert said == ""
    said = _no_rate("What bridge loan rates do you have?", {"bridge"}, view.cache["rows"], view.cache["passages"])
    assert said == "The website does not publish a rate for bridge loans."
    said = _no_rate("Where was the Harbor branch in Apr 2026?", {"harbor"}, view.cache["rows"], view.cache["passages"])
    assert said == ""                                                   # no branch has a rate


def test_a_word_that_only_describes_listed_rows_leaves_the_rate_to_the_reader():
    # "Used" names no rate table, but the auto-loan rows may be what is asked: the reader sees them.
    site = ("2026-06-10", {
        "rates": "# Rates\n\nAuto Loans\n\n| Model Year | Term | APR |\n| --- | --- | --- |\n"
                 "| 2024-27 | 60 Months | 4.85% |\n| 2017-20 | 60 Months | 6.15% |\n",
        "cars": "# Buying a car\n\nNew or used, we can help. Auto loans are as low as 4.85% APR on a 2024 or newer model.\n"})
    ks = with_reader(model_reply("A 2018 model is 6.15% APR over 60 months."), site)
    asked = "What is the APR on a used auto loan for a 2018 model?"
    answer = ks.answer(asked)
    assert ks.engine.reader.backend.calls == 1 and "6.15%" in ks.engine.reader.backend.prompts[0]
    assert answer.confident and "6.15%" in answer.text
    # Without a reader nothing tells which row is meant, and a rate is published: the plain refusal, not
    # the article's rate for a newer model.
    ks.engine.reader = None
    ks.engine._answers.clear()
    assert refused(ks.answer(asked), asked) and ks.answer(asked).text == REFUSAL


def test_a_word_one_of_the_rows_accounts_for_is_listed_though_tied_rows_match_other_words():
    # The boat row matches "boat", the auto row "used": neither word is missing, so the reader decides.
    site = ("2026-06-10", {"rates": "# Rates\n\nBoat Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | 7.25% |\n\n"
                                    "Auto Loans\n\n| Term | APR |\n| --- | --- |\n| New 60 Months | 4.85% |\n"
                                    "| Used 60 Months | 5.40% |\n"})
    ks = with_reader(model_reply("A boat loan is 7.25% APR over 60 months."), site)
    asked = "What is the APR on a used boat loan?"
    answer = ks.answer(asked)
    assert ks.engine.reader.backend.calls == 1 and "7.25%" in ks.engine.reader.backend.prompts[0]
    assert answer.confident and "7.25%" in answer.text
    ks.engine.reader = None                     # without one: not "the website does not publish a rate for used boat loans"
    ks.engine._answers.clear()
    assert refused(ks.answer(asked), asked) and ks.answer(asked).text == REFUSAL


def test_a_kind_no_row_lists_is_declined_whatever_the_word_before_it_matches():
    site = ("2026-06-10", {"rates": "# Rates\n\nHome Equity Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | 6.10% |\n",
                           "build": "# Building\n\nAsk a lender about construction loans.\n"})
    ks = with_reader(None, site)
    answer = ks.answer("What is the rate on a home construction loan?")
    assert ks.engine.reader.backend.calls == 0
    assert answer.text == f"The website does not publish a rate for construction loans. {ASK}"

DESCRIBED = ("2026-06-10", {
    "cards": "# Credit cards\n\nCredit Cards\n\n| Card | APR |\n| --- | --- |\n| Everyday Visa | 13.00% |\n"
             "| Cashback Visa | 16.00% |\n",
    "savings": "# Savings\n\nOrdinary Wren Certificates\n\n| Term | APY |\n| --- | --- |\n| 12 Month | 3.00% |\n\n"
               "Fir Savings\n\n| Balance | APY |\n| --- | --- |\n| $5 - $1,000 | 2.00% |\n| $1,000.01+ | 2.50% |\n",
    "mortgages": "# Mortgages\n\nMortgages\n\n| Loan | APR |\n| --- | --- |\n| 30-Year Fixed $100,000 or less | 6.00% |\n"
                 "| 30-Year Fixed $300,000 or less | 6.20% |\n",
    "about": "# About us\n\nA small, regular, basic and plain credit union: start saving here. Ask about builder loans.\n"})


@pytest.mark.parametrize("asked", ["What's the rate on your ordinary credit card?", "basic credit card APR?",
                                   "What rate would I get to start saving?", "plain savings APY?"])
def test_a_describing_word_or_a_verb_names_no_kind_of_product(asked):
    ks = with_reader(model_reply("Everyday Visa is 13.00% APR."), DESCRIBED)
    reply = ks.engine.answer(asked)
    assert reply.path in ("rows", "reader") and reply.confident


def test_a_describing_word_leaves_the_tier_to_the_amount():
    ks = with_reader(None, DESCRIBED)
    assert ks.answer("30 year mortgage rate for a small loan like 90k?").text == "Mortgages, 30-Year Fixed $100,000 or less: 6.00% APR."
    # A row on the same page with the word in its name makes it no kind either.
    assert "2.50% APY" in ks.answer("If I keep $5,000 in ordinary fir savings, what APY do I earn?").text


BUILDER = ("2026-06-10", {
    "rates": "# Rates\n\nHome Equity Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | 6.10% |\n\n"
             "Cedar Checking\n\n| Balance | APY |\n| --- | --- |\n| $5,000+ | 0.20% |\n",
    "builder": "# Builder loans\n\nA builder loan converts to an adjustable mortgage. Its rate may adjust up to 1% a year, "
               "never more than 5% above the start rate.\n",
    "checking": "# Cedar checking\n\nCedar checking pays interest. Its monthly fee of $8 is waived at $5,000.\n"})


@pytest.mark.parametrize("asked", ["How much can my builder loan rate go up?", "How high can the builder loan rate go?",
                                   "What are the rate caps on a builder loan?",
                                   "Can the rate on a builder loan go up, and by how much?",
                                   "How much can my builder loan APR go up?",
                                   "What's the highest my builder loan rate could ever go?",
                                   "Is there a monthly fee on the cedar checking that pays interest?"])
def test_a_question_about_a_fee_or_how_a_rate_changes_asks_no_published_rate(asked):
    ks = with_reader(model_reply("Up to 1% a year, never more than 5% above the start rate."), BUILDER)
    assert ks.engine.answer(asked).path == "reader"
    assert "no row for" not in ks.engine.reader.backend.prompts[0]


@pytest.mark.parametrize("asked", [
    "What is the builder loan rate?", "What interest rate do you charge on a builder loan?",
    # A fee or a minimum that describes the product, or a change that is not the rate's, asks the rate still.
    "What APR do you charge on a builder loan with no fee?", "What's the APR on your no-fee builder loan?",
    "What APR does a builder loan with a $1,000 minimum carry?", "What APR would I get on a builder loan if rates go down?",
    "How much is the rate on a builder loan and can it change?", "What's the APR on the low-fee builder loan?",
    "What is the minimum APR on a builder loan?", "How much would the builder loan rate be if rates drop?",
    # A rate asked beside a fee or a minimum is asked still; "this fall" is a season.
    "What's the APR on a builder loan, and is there an origination fee?",
    "What is the rate on a builder loan; what is the minimum deposit?", "What will builder loan rates be this fall?",
])
def test_a_rate_for_a_kind_no_table_lists_is_still_declined_by_rule(asked):
    ks = with_reader(None, BUILDER)
    assert ks.answer(asked).text == f"The website does not publish a rate for builder loans. {ASK}"
    assert ks.engine.reader.backend.calls == 0


def test_an_adjustable_rate_product_asks_its_rate():
    from kb.lookup import asks_rate, parse_question
    assert asks_rate(parse_question("Can you tell me the rate on an adjustable mortgage?"))


@pytest.mark.parametrize("said, shown", [
    ("The website does not list the rate of the 41-month special.", True),
    ("", False),                                                          # nothing said: the fixed sentence
    ("The website does not list the rate of the 22-month special.", False),          # a figure not in the question
    ("The website does not list its rate, though the 14-month special pays more.", False),    # a second statement
    ("The website does not list the rate; it lists two specials.", False),
    ("Branch hours are not listed and the special ended in April.", False),
    ("The website does not list a rate for jumbo loans but it offers the 41-month special.", False),
    ("The website does not list the rate of the 41-month special" + " for members" * 10 + ".", False),   # too long
    ("No APY is published for the 41-month special.", True),
    ("The evidence does not list the rate of the 41-month special.", False),         # about the message, not the site
    ("Ask at a branch about the 41-month special.", False),                           # not about what the site says
])
def test_a_decline_says_what_the_site_does_not_state_only_when_that_is_all_it_says(said, shown):
    ks = with_reader(f"QUOTES: none\nVERDICT: decline\nSOURCES: none\nANSWER: {said}", SPRING, NOW)
    asked = "What is the APY on the 41-month special?"
    answer = ks.answer(asked)
    assert ks.engine.reader.backend.calls == 1 and refused(answer, asked)
    assert answer.text == (f"{said} {ASK}" if shown else REFUSAL)
    assert figures(answer.text) <= figures("What is the APY on the 41-month special?")


@pytest.mark.parametrize("said", ["$ 2,468.13 or 6.15 % and 6.150%", "11 percent of $97 for 97 months",
                                  "  $ 2,468.13  4,02 .17% 43."])
def test_a_figure_is_a_number_as_the_scorer_counts_one(said):
    assert figures(said) == scoring._numbers(said)


def test_looking_for_figures_does_not_run_away_on_a_long_run_of_spaces():
    start = time.perf_counter()
    assert reader._not_stated("No longer listed." + " " * 30_000, "Which specials" + " " * 30_000 + "are open?")
    assert time.perf_counter() - start < 2            # with a \s* tried at every space, this took 18 s
