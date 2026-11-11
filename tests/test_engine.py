"""The routes a question can take (rows, hours, link, contact, quote, declined), each reason a
question leaves the rows route, the answer cache, and which pages an answer cites.

Every answer here is given without a language model, the way the solution answers when no key is
set or the model gives no reply. Made-up sites; no downloads, no network.
"""
import re

import pytest

from helpers import REFUSAL, build, has_digit, model_reply, path_of, scripted_reader, snapshot, table

BRANCH = ("1 min read\n\nMarch 3, 2025\n\n# Credit union opens {town} branch\n\n{TOWN}, {st}-The new branch at {street} in "
          "{town} is open.\n\nOpening times are:\n\n**Lobby**\nMonday-Thursday: 9:15 a.m.-4:45 p.m.\nFriday: 9:15 a.m.-{friday}\n\n"
          "**Drive-up**\nMonday-Friday: 7:45 a.m.-6:15 p.m.\n")
SITE = {
    "rates": table("Wren Certificates", [("6 Month", "1.65%"), ("12 Month", "2.05%")], "02/02/2026"),
    "files/fee-schedule.pdf": "Schedule of Charges\n\nCard services\n\nLost card replacement $6.75\nCoin wrapping $3.00\n",
    "newsroom/elm": BRANCH.format(town="Elmwood", TOWN="ELMWOOD", st="Kan.", street="12 Elm Street", friday="6 p.m."),
    "newsroom/oak": BRANCH.format(town="Oakdale", TOWN="OAKDALE", st="N.C.", street="40 Oak Street", friday="5:30 p.m."),
    "contact-us": "# Contact us\n\nCall us at 555-010-4477, write to PO Box 12, Larkspur, or visit any branch during "
                  "business hours.\n",
    "about": "# About Larkspur Valley Credit Union\n\nThe credit union was founded in 1961 by nine orchard growers who "
             "pooled their savings.\n",
    "loans": "# Loans\n\nWe finance construction of new homes. Ask a lender about construction loans today.\n\n"
             "[Apply for a Wren certificate](https://cu.test/apply/wren)\n",
    "kayak": "# Kayak Saver\n\nOpen with $5. The Kayak Saver earns an APY of 1.25% on every balance.\n",
}


@pytest.fixture(scope="module")
def ks():
    return build(("2026-02-10", SITE))


@pytest.mark.parametrize("question, route", [
    ("What is the APY on a 6-month Wren certificate?", "rows"),
    ("How much is a lost card replacement?", "rows"),
    ("What are your lobby hours?", "hours"),
    ("What are the lobby hours at the Elmwood branch?", "hours"),
    ("Elm Street hours?", "hours"),                                      # a street right at the start
    ("What are the branch hours in North Carolina?", "hours"),          # a state of two words, and nothing else
    ("What link do I use to apply for a Wren certificate?", "link"),
    ("How do I reorder paper checks?", "contact"),
    ("What's the best way to reorder paper checks?", "contact"),
    ("When was the credit union founded?", "quote"),
    ("What is the APY on the Kayak Saver?", "quote"),                   # "an APY of 1.25%" states the rate
    ("What is the rate on a construction loan?", "declined"),           # a product the site names but prices nowhere
    ("What is the rate on a canoe loan?", "declined"),                  # a product the site never names
    ("What is the APY on an 18-month Wren certificate?", "declined"),   # a term the table does not list
    ("What are the hours at the Zorbleton branch?", "declined"),        # a branch the site never names
    ("Reorder paper checks?", "declined"),                              # not a how-to: no contact route
    ("How much paint covers a barn?", "declined"),                      # nor is "how much": it asks for no steps
])
def test_each_question_takes_its_route(ks, question, route):
    assert path_of(ks, question) == route


def test_a_rows_answer_names_the_tier_and_the_date_once(ks):
    text = ks.answer("What are the Wren certificate APYs?").text
    assert text == ("Wren Certificates, 6 Month: 1.65% APY; Wren Certificates, 12 Month: 2.05% APY "
                    "(rates effective February 2, 2026).")


@pytest.mark.parametrize("question", ["What is the APY on an 18-month Wren certificate?", "What is the rate on a canoe loan?",
                                      "What are the hours at the Zorbleton branch?"])
def test_a_declined_question_is_the_fixed_refusal_with_no_sources(ks, question):
    answer = ks.answer(question)
    assert (answer.text, answer.confident, answer.sources) == (REFUSAL, False, [])


# --------------------------------------------------------------------------- why a question leaves the rows route

@pytest.mark.parametrize("question", [
    "What is the fee for a Wren certificate?",                 # a fee asked; the rows give rates
    "How much does the Wren certificate earn?",               # no measure named, and the rows are rates
    "Business APY?",                                           # only "business" matched: that names no product
    "What is the 6-month Wren certificate rate for my construction business?",   # a kind the rows do not list
])
def test_a_question_the_rows_do_not_settle_is_not_answered_from_them(ks, question):
    assert path_of(ks, question) != "rows"


def test_a_value_inside_an_article_is_read_not_looked_up():
    story = "1 min read\n\nMarch 1, 2026\n\n# Plover news\n\n" + table("Plover Certificates", [("6 Month", "1.95%")])
    ks = build(("2026-03-05", {"news/plover": story}))
    assert path_of(ks, "What is the APY on a 6-month Plover certificate?") != "rows"


def test_a_value_in_a_sentence_needs_most_of_the_product_words():
    page = "# Specials\n\nSpring promo: 6-month Plover certificate APY 2.95%.\n"
    ks = build(("2026-03-05", {"specials": page}))
    assert path_of(ks, "What is the APY on a 6-month Plover certificate?") == "rows"
    assert path_of(ks, "What is the APY on a 6-month Plover jumbo harvest certificate?") != "rows"


def test_a_link_question_about_a_product_named_with_a_measure_word_gets_a_link():
    # "Gold Rate Checking" has "rate" in its name; "What link shows its rates?" still asks for a link.
    gold = ("# Gold Rate Checking\n\nEarn more on every balance. [See details](https://cu.test/gold-rate-checking)\n\n"
            "## Rates\n\nGold Rate Checking\n\n| Balance | APY |\n| --- | --- |\n| $5,000+ | 2.15% |\n| $0 - $4,999.99 | 0.05% |\n")
    pages = {"checking": gold, "gold-rate-checking": "# Gold Rate Checking details\n\nOpen it online in minutes.\n"}
    for n in range(20):                                 # enough pages for "gold" to be a telling word
        pages[f"news/item-{n}"] = f"# Community note {n}\n\nVolunteers planted trees in park number {n} this spring.\n"
    answer = build(("2026-03-05", pages)).answer("What link shows the Gold Rate Checking rates?").text
    assert answer == 'Use https://cu.test/gold-rate-checking (the site\'s "See details" link).'


# --------------------------------------------------------------------------- the answer cache

def test_the_same_question_however_spaced_is_answered_once_and_alike(ks):
    first = ks.engine.answer("What is the APY on a 6-month Wren certificate?")
    for again in ("What is the APY on a 6-month Wren certificate?",
                  "  What is the APY on a 6-month Wren   certificate?  "):
        assert ks.engine.answer(again) is first


def test_a_question_written_in_other_capitals_is_not_given_the_cached_answer_of_its_twin():
    # A name in capitals is read as written and one in small letters is spell-checked (see kb.quote), so
    # the two can have different answers, whichever was asked first.
    pages = {"newsroom/elm": SITE["newsroom/elm"]}
    pages.update({f"news/fair-{n}": f"# Fair {n}\n\nMembers met at the harvest fair.\n" for n in range(3)})
    for first, then in (("what is the address of the harvesta branch?", "What is the address of the Harvesta branch?"),
                        ("What is the address of the Harvesta branch?", "what is the address of the harvesta branch?")):
        ks = build(("2026-03-05", pages))
        ks.answer(first)
        assert ks.answer(then).text == build(("2026-03-05", pages)).answer(then).text


def test_a_new_download_or_another_date_is_another_answer():
    ks = build(("2026-01-05", {"rates": table("Wren Certificates", [("6 Month", "2.10%")])}))
    before = ks.answer("What is the APY on a 6-month Wren certificate?").text
    then = ks.answer("What is the APY on a 6-month Wren certificate?", as_of="2026-01-10").text
    assert "2.10%" in before and then.startswith("As of 2026-01-10")
    ks.ingest(snapshot(2, "2026-03-05", {"rates": table("Wren Certificates", [("6 Month", "2.60%")])}))
    assert "2.60%" in ks.answer("What is the APY on a 6-month Wren certificate?").text
    assert "2.10%" in ks.answer("What is the APY on a 6-month Wren certificate?", as_of="2026-01-10").text


# --------------------------------------------------------------------------- sources

def test_a_rows_answer_cites_every_page_that_prints_the_value_most_trusted_first_and_at_most_ten():
    wren = table("Wren Certificates", [("6 Month", "2.60%")])
    pages = {f"wren-{n:02d}": wren for n in range(12)}
    pages.update({"files/wren-sheet.pdf": wren, "wren-lp": wren, "rates": wren})
    sources = [s.url for s in build(("2026-03-05", pages)).answer("What is the APY on a 6-month Wren certificate?").sources]
    # Thirteen product pages print it: the ten cited are product pages, not the document or the campaign page.
    assert len(sources) == 10 and not any(u.endswith((".pdf", "-lp")) for u in sources)


def test_a_labelled_number_cites_every_page_that_prints_it_most_trusted_first():
    pages = {"about": "# About\n\nRouting Number: 271987654\n", "files/forms.pdf": "Direct deposit form\n\nUse 271987654.\n",
             "promo-lp": "# Switch\n\nOur routing number is 271987654.\n", "help/faq": "# FAQ\n\nAsk us anything.\n"}
    answer = build(("2026-03-05", pages)).answer("What is the routing number?")
    assert "271987654" in answer.text
    assert [s.url for s in answer.sources] == ["https://cu.test/about", "https://cu.test/files/forms.pdf",
                                               "https://cu.test/promo-lp"]


def test_an_hours_answer_cites_the_branch_page_and_the_contact_page(ks):
    sources = [s.url for s in ks.answer("What are the lobby hours at the Elmwood branch?").sources]
    assert sources == ["https://cu.test/newsroom/elm", "https://cu.test/contact-us"]


# --------------------------------------------------------------------------- the answers without a model

def test_a_link_answer_gives_the_link_and_the_words_the_site_puts_on_it(ks):
    answer = ks.answer("What link do I use to apply for a Wren certificate?")
    assert answer.text == 'Use https://cu.test/apply/wren (the site\'s "Apply for a Wren certificate" link).'
    assert [s.url for s in answer.sources] == ["https://cu.test/loans"]


def test_a_link_question_no_link_matches_gets_no_link(ks):
    assert path_of(ks, "Which link has the Zorbleton form?") != "link"


def test_a_how_to_with_nothing_on_the_site_gets_the_contact_paragraph(ks):
    answer = ks.answer("How do I reorder paper checks?")
    assert answer.text == ("The website doesn't give specific steps for that. Call us at 555-010-4477, write to PO Box 12, "
                           "Larkspur, or visit any branch during business hours.")
    assert answer.confident and [s.url for s in answer.sources] == ["https://cu.test/contact-us"]


def test_a_quote_carries_the_date_of_the_download_it_is_from(ks):
    text = ks.answer("When was the credit union founded?", as_of="2026-03-01").text
    assert text.startswith("As of 2026-03-01 (the website as downloaded on 2026-02-10): The credit union was founded in 1961")
    assert len(text) <= 900


def test_a_rate_stated_for_another_product_is_not_quoted_for_this_one():
    pages = {"loans": "# Loans\n\nAuto loans as low as 4.99% APR for members.\n", "rates": SITE["rates"]}
    answer = build(("2026-03-05", pages)).answer("What is the rate on a kayak loan?")
    assert answer.text == REFUSAL


def test_a_tier_the_table_lacks_is_declined_even_when_a_passage_states_other_tiers_rates():
    page = table("Wren Certificates", [("6 Month", "1.65% APY"), ("12 Month", "2.05% APY")])
    page += "\nOur 12-month Wren certificate earns 2.05% APY.\n"
    for question in ("What is the APY on an 18-month Wren certificate?", "Wren certificate APY for 48 months?"):
        answer = build(("2026-03-05", {"rates": page})).answer(question)
        assert answer.text == REFUSAL and not has_digit(answer.text), question


@pytest.mark.parametrize("question", ["What is the APR on a 60-month Wren loan?", "What is the APR on a 60-month Wren mortgage?"])
def test_a_kind_of_product_the_brand_does_not_have_is_not_answered_with_another_kind(question):
    answer = build(("2026-03-05", {"wren/certificates": table("Wren Certificates", [("60 Month", "3.58%")])})).answer(question)
    assert "3.58%" not in answer.text


@pytest.mark.parametrize("question, want", [("What is the APR on a 60-month Wren loan?", "5.10%"),
                                            ("What is the rate on a 60-month Wren certificate?", "3.58%")])
def test_of_a_brand_s_loan_and_certificate_each_question_gets_its_own_kind(question, want):
    pages = {"wren/certificates": table("Wren Certificates", [("60 Month", "3.58%")]),
             "wren/loans": table("Wren Loans", [("60 Month", "5.10%")], header="| Term | APR |")}
    assert set(re.findall(r"\d\.\d\d%", build(("2026-03-05", pages)).answer(question).text)) == {want}


@pytest.mark.parametrize("question", ["Tree house savings rate on $500?", "APY on tree-house savings with $500"])
def test_two_words_the_rows_write_as_one_are_that_word(question):
    rates = table("Treehouse Savings", [("$5 - $1,000", "3.25%"), ("$1,000.01+", "0.10%")], header="| Balance | APY |")
    ks = build(("2026-06-10", {"rates": rates, "kids": "# Kids\n\nHouse rules for young savers: start early.\n"}))
    assert path_of(ks, question) == "rows" and ks.answer(question).text == "Treehouse Savings, $5 - $1,000: 3.25% APY."


LOCKERS = ("2026-06-10", {"fees": "# Fee schedule\n\n## Lockers\n\n| Service | Fee |\n| --- | --- |\n"
                                  "| Storage locker 2 X 2 | $10.00 |\n| Storage locker 4 X 4 | $30.00 |\n"
                                  "| Storage locker late fee | $5.00 |\n",
                          "rates": table("Auto Loans", [("36 Months", "4.10%"), ("60 Months", "4.60%")], header="| Term | APR |")})


@pytest.mark.parametrize("question", ["What does the smallest storage locker cost?", "cheapest storage locker fee?",
                                      "What does the biggest storage locker cost?"])
def test_the_highest_or_lowest_of_rows_with_no_rate_is_left_to_be_read(question):
    ks = build(LOCKERS, reader=scripted_reader(model_reply("The 2 X 2 locker is $10.00.")))
    assert path_of(ks, question) == "reader"


def test_the_lowest_rate_is_still_taken_from_the_rows():
    assert build(LOCKERS).answer("What is the lowest auto loan APR?").text == "Auto Loans, 36 Months: 4.10% APR."


def test_a_row_that_states_an_apr_is_a_loan_s_even_when_no_word_says_loan():
    pages = {"wren": table("Wren", [("60 Month", "5.10%")], header="| Term | APR |")}
    assert "5.10%" not in build(("2026-03-05", pages)).answer("What is the APY on a 60-month Wren certificate?").text


def test_a_product_the_site_names_without_a_rate_is_declined_even_beside_another_rate():
    page = ("# Loans\n\nWe finance construction of new homes; ask a lender about construction loans. "
            "Auto loans are as low as 4.99% APR.\n")
    answer = build(("2026-03-05", {"loans": page})).answer("What is the rate on a construction loan?")
    assert not answer.confident
    assert answer.text == "The website does not publish a rate for construction loans. Please contact the credit union for details."


@pytest.mark.parametrize("page", [
    "# Loans\n\n## Construction loans\n\nBuild the home you planned. Rates start as low as 6.25% APR.\n",
    "# Loans\n\nA construction loan pays your builder. It's a short-term, interest-only loan. With draws as the work "
    "goes on and a low 6.25% APR, you pay for what is built.\n",
    # "It" goes on about the loan just named, though the sentence names another kind of loan.
    "# Loans\n\nA construction loan builds your home. It costs less than our motorcycle loans: just 6.25% APR.\n"])
def test_a_rate_said_under_a_heading_or_in_sentences_that_go_on_about_the_product_is_its_rate(page):
    assert "6.25%" in build(("2026-03-05", {"loans": page})).answer("What is the rate on a construction loan?").text


def test_a_how_to_is_not_answered_with_a_passage_that_shares_one_incidental_word():
    pages = {"contact-us": SITE["contact-us"], "rates": SITE["rates"]}
    answer = build(("2026-03-05", pages)).answer("How do I change my mailing address?")
    assert "%" not in answer.text and answer.text.startswith("The website doesn't give specific steps")


# --------------------------------------------------------------------------- hours

def test_a_general_hours_question_lists_each_distinct_schedule_and_says_they_vary(ks):
    text = ks.answer("What are your lobby hours?").text
    assert text.startswith("Lobby hours vary by branch. Published schedules: ")
    assert "Friday: 9:15 a.m.-6 p.m." in text and "Friday: 9:15 a.m.-5:30 p.m." in text and "Drive-up" not in text


def test_branches_with_the_same_schedule_are_listed_once_with_their_count():
    pages = {f"newsroom/{t.lower()}": BRANCH.format(town=t, TOWN=t.upper(), st="Kan.", street="1 Main Street", friday="6 p.m.")
             for t in ("Elmwood", "Oakdale")}
    text = build(("2026-03-05", pages)).answer("What are your drive-up hours?").text
    assert text.startswith("Drive-up hours: Monday-Friday: 7:45 a.m.-6:15 p.m. (2 branches).")


def test_hours_volunteered_at_a_branch_are_not_its_opening_hours(ks):
    assert path_of(ks, "How many hours did the Elmwood branch staff volunteer?") != "hours"


def test_a_branch_with_no_published_hours_is_not_given_another_branch_s_hours():
    pages = {"newsroom/elm": SITE["newsroom/elm"],
             "branches/cedarton": "# Cedarton branch\n\nOur Cedarton branch sits beside the river park and has a coin "
                                  "machine and two drive-up lanes for members.\n"}
    answer = build(("2026-03-05", pages)).answer("What are the hours at the Cedarton branch?")
    assert "9:15" not in answer.text and "7:45" not in answer.text


def test_a_general_hours_answer_cites_at_most_ten_pages():
    pages = {f"newsroom/b{n}": BRANCH.format(town=f"Town{chr(65 + n)}", TOWN="TOWN", st="Kan.", street="1 Main Street",
                                             friday=f"{n % 5 + 1}:30 p.m.") for n in range(12)}
    pages["contact-us"] = SITE["contact-us"]
    answer = build(("2026-03-05", pages)).answer("What are your lobby hours?")
    assert answer.text.startswith("Lobby hours vary by branch.") and len(answer.sources) == 10


def test_a_street_named_after_a_state_at_the_start_of_the_question_names_the_branch():
    pages = {"newsroom/oak": BRANCH.format(town="Oakdale", TOWN="OAKDALE", st="Kan.", street="40 Ohio Street", friday="5:30 p.m.")}
    ks = build(("2026-03-05", pages))
    assert path_of(ks, "Ohio Street branch hours?") == "hours"
    assert "5:30 p.m." in ks.answer("Ohio Street branch hours?").text


def test_no_published_schedule_gives_no_hours_answer():
    from kb import hours
    assert hours.answer("What are your hours?", []) is None


def test_a_rate_a_passage_states_is_quoted_even_when_some_rows_match_a_word_of_the_question():
    pages = {"kayak": "# Kayak Paddle Saver\n\nOpen with $5. The Kayak Paddle Saver earns an APY of 1.25% on every balance.\n",
             "saver": table("Wren Saver Certificates", [("6 Month", "2.20%")])}
    answer = build(("2026-03-05", pages)).answer("What is the APY on the Kayak Paddle Saver?")
    assert "1.25%" in answer.text and "2.20%" not in answer.text


def test_a_kind_of_product_named_only_in_a_section_heading_still_keeps_another_kind_out():
    # "Mortgage" is only the heading over the loan table; the jumbo rows are certificates.
    pages = {"certificates": table("Jumbo Certificates", [("12 Month", "4.40%")]),
             "lending": "# Lending\n\n## Mortgage rates\n\nFixed\n\n| Term | APR |\n| --- | --- |\n| 30 Year | 6.25% |\n"}
    answer = build(("2026-03-05", pages)).answer("What is the APR on a jumbo mortgage?")
    assert "4.40%" not in answer.text


def test_a_count_question_is_answered_by_the_passage_with_a_number_beside_the_thing_counted():
    pages = {"about": "# About us\n\nOpen 6 days a week, our branches are friendly places where the credit union staff know your name and your "
                      "branch manager answers the phone.\n",
             "facts": "# Facts\n\nThe credit union now runs 14 branches across the valley.\n"}
    answer = build(("2026-03-05", pages)).answer("How many branches does the credit union have?")
    assert answer.text == "The credit union now runs 14 branches across the valley."


@pytest.mark.parametrize("question, shapes", [
    ("Where is the Elmwood branch?", 1), ("What is the address of the Elmwood branch?", 1), ("Where's the branch?", 1),
    ("What are the Elmwood hours?", 1), ("What is the phone number for lending?", 1), ("Tell me about savings.", 0),
])
def test_a_question_is_read_for_the_kind_of_answer_it_needs(question, shapes):
    from kb.passages import answer_shapes
    assert len(answer_shapes(question)) == shapes
