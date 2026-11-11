"""How an answer from table rows is worded: the order of the tiers and the names of the rows, on a
made-up site. No website snapshots and no language model."""
import dataclasses
import os
import subprocess
import sys

import pytest

from helpers import ROOT, build, table
from kb import compose, reader
from kb.lookup import find, parse_question

# The site lists its loan-amount tiers from the largest down, and its terms from the longest down.
MORTGAGES = ("# Mortgage rates\n\n## Mortgages - Buying\n\n| Loan | Rate | APR |\n| --- | --- | --- |\n"
             "| Buying: 30-Year Fixed $400,000 or less | 6.50% | 6.61% |\n"
             "| Buying: 30-Year Fixed $150,000 or less | 6.25% | 6.38% |\n"
             "| Buying: 15-Year Fixed $400,000 or less | 5.75% | 5.90% |\n"
             "| Buying: 15-Year Fixed $150,000 or less | 5.50% | 5.66% |\n")
SAVINGS = ("# Savings rates\n\n## Sunrise Savings\n\n| Balance | APY |\n| --- | --- |\n"
           "| $20,000+ | 0.30% |\n| $1,000 - $19,999.99 | 0.20% |\n| $0 - $999.99 | 0.05% |\n\n"
           "## Harbor Certificates\n\n| Term | APY |\n| --- | --- |\n| 48 Month | 3.10% |\n| 12 Month | 2.90% |\n"
           "| Jumbo: 12 Month | 3.00% |\n")
# More tiers than an answer lists (compose.MAX_ROWS), the largest first.
LADDER = "# Ladder savings\n\n## Ladder Savings\n\n| Balance | APY |\n| --- | --- |\n" + "".join(
    f"| ${n},000+ | {n / 100:.2f}% |\n" for n in range(30, 0, -1))


@pytest.fixture(scope="module")
def ks():
    return build(("2026-06-10", {"mortgages": MORTGAGES, "savings": SAVINGS, "ladder": LADDER}))


def in_order(text, *parts):
    """Whether every part is in text, in this order."""
    at = [text.find(p) for p in parts]
    return -1 not in at and at == sorted(at)


def test_amount_tiers_are_listed_from_the_smallest_up_when_none_is_asked(ks):
    text = ks.answer("What are your mortgage rates?").text
    # Within each term, and the terms in the site's order.
    assert in_order(text, "30-Year Fixed $150,000", "30-Year Fixed $400,000", "15-Year Fixed $150,000", "15-Year Fixed $400,000")
    text = ks.answer("What APY does Sunrise Savings pay?").text
    assert in_order(text, "$0 - $999.99", "$1,000 - $19,999.99", "$20,000+")


def test_terms_keep_the_site_s_order(ks):
    text = ks.answer("What APY do Harbor Certificates pay?").text
    assert in_order(text, "48 Month", "12 Month")


def test_a_row_s_name_does_not_repeat_its_table_title(ks):
    text = ks.answer("What are your mortgage rates?").text
    assert "Mortgages - Buying, 30-Year Fixed $150,000 or less: 6.25% rate, 6.38% APR" in text
    assert "Buying: " not in text


@pytest.mark.parametrize("asked, said", [
    ("Do any Sunrise Savings tiers pay 0.25% APY?", "No. "),
    ("Does any Sunrise Savings tier pay 0.20% APY?", "Yes. "),
    ("Is Sunrise Savings paying 0.30 percent APY?", "Yes. "),
    ("What APY does Sunrise Savings pay?", ""),                 # not a yes-or-no question
    ("Does Sunrise Savings pay dividends monthly?", ""),        # no rate in the question to compare
])
def test_a_yes_or_no_question_about_a_rate_is_answered_yes_or_no_first(ks, asked, said):
    text = ks.answer(asked).text
    assert text.startswith(said) and not text[len(said):].startswith(("Yes", "No"))


def rows_for(ks, question):
    q = parse_question(question)
    return q, find(ks.engine.store.view_at(None).cache["rows"], q).rows


def test_a_question_that_names_an_amount_keeps_the_site_s_order(ks):
    q, rows = rows_for(ks, "What are your 30-year fixed mortgage rates?")
    assert compose.in_order(q, rows) != rows
    q.quals = dataclasses.replace(q.quals, amount=200_000)        # as if it had named one
    assert compose.in_order(q, rows) == rows


def test_the_tiers_shown_are_the_site_s_first_ones_put_in_order(ks):
    text = ks.answer("What APY does Ladder Savings pay?").text
    assert in_order(text, "$7,000+", "$30,000+") and "$6,000+" not in text        # 24 of 30 rows, the site's first


def test_the_tiers_are_in_order_in_what_the_model_is_shown_too(ks):
    question = "Which mortgage suits a first home?"
    view = ks.engine.store.view_at(None)
    q, rows = parse_question(question), view.cache["rows"]
    user = reader.Reader(None).prompt(question, None, view, q, find(rows, q), view.cache["passages"].search(question, k=14),
                                   rows)[0]
    assert in_order(user, "30-Year Fixed $150,000", "30-Year Fixed $400,000")


def test_a_row_name_keeps_words_its_title_does_not_have(ks):
    assert "Harbor Certificates, Jumbo: 12 Month: 3.00% APY" in ks.answer("What APY do Harbor Certificates pay?").text


@pytest.mark.parametrize("asked", [
    "Does any Sunrise Savings tier pay more than 0.10% APY?",   # a comparison: an equal rate does not answer it
    "Are Sunrise Savings rates above 0.25%?",
    "Do any Sunrise Savings tiers pay at least 0.20% APY?",
    "Is 0.05% the dividend rate on Sunrise Savings?",           # no row states a dividend rate
    "Sunrise Savings 0.20% APY?",                               # not a yes-or-no question
])
def test_no_yes_or_no_where_a_row_s_rate_cannot_decide_it(ks, asked):
    q, rows = rows_for(ks, asked)
    assert rows and compose.yes_or_no(q, rows) == ""


def test_the_reason_given_explains_the_copies_that_competed():
    # The answer's first row (Silver) had no rival: the reason is about the Travel card's copies.
    h = "| Card | APR |"
    ks = build(("2026-03-05", {"a-cards": table("Cards", [("Silver Card", "11.25%")], header=h),
                               "rates": table("Cards", [("Travel Card", "13.75%")], "03/01/2026", header=h),
                               "guide": "# Card guide\n\n" + table("Cards", [("Travel Card", "12.80%")], "01/02/2026", header=h)}))
    note = ks.answer("What are the APRs on your cards?").conflict_note
    assert note.endswith("(its copy is dated 2026-01-02, this one 2026-03-01).")


HIGHEST = """
import sys
sys.path[:0] = ["tests", "harness", "solution"]
from helpers import build, table
rates = table("Tern Certificates", [("12 Month", "3.00%", "3.04%"), ("24 Month", "2.90%", "3.10%")],
              header="| Term | Dividend Rate | APY |")
print(build(("2026-06-10", {"rates": rates})).answer("Which Tern certificate has the highest dividend rate or APY?").text)
"""


def test_with_two_measures_named_the_highest_is_taken_by_the_first_in_a_fixed_order_in_any_process():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("BANKGPT_", "ANTHROPIC_"))}
    for seed in ("1", "2", "3", "4", "5"):
        out = subprocess.run([sys.executable, "-c", HIGHEST], cwd=ROOT, env=dict(env, PYTHONHASHSEED=seed, BANKGPT_LLM="off"),
                             capture_output=True, text=True, timeout=60)
        assert out.stdout.strip() == "Tern Certificates, 12 Month: 3.00% dividend rate, 3.04% APY.", (seed, out.stderr[-500:])


# A copy that loses but states only figures the answer gives is no rival: its page prints the answer.
FEES_AS_TABLE = ("Schedule of Charges, Effective March 9, 2026\n\nOther fees\n\n| Item charged | Amount |\n| --- | --- |\n"
                 "| Courier fee for a lost card | $45.00 |\n| Coin wrapping | $3.00 |\n")
FEES_AS_LINES = "Schedule of Charges, Effective March 9, 2026\n\nOther fees\n\nCourier fee for a lost card $45.00\nCoin wrapping $3.00\n"
KESTREL = table("Kestrel Money Market", [("$50,000+", "3.90%", "3.97%", "2.10%", "2.12%")],
                header="| Balance | Dividend Rate for First 3 Months | APY for First 3 Months | Dividend Rate After | APY After |")
KESTREL_ASKED = "What is the APY on the Kestrel Money Market for 60000 dollars?"


def kestrel_copy(dividend, apy, title="Kestrel offer"):
    """Another page's copy of the Kestrel tier, with one dividend rate and one APY."""
    return f"# {title}\n\n" + table("Kestrel Money Market", [("$50,000+", dividend, apy)], header="| Balance | Dividend Rate | APY |")


def urls(answer):
    return [s.url for s in answer.sources]


def test_a_fee_read_as_a_table_in_one_copy_and_as_a_line_in_another_is_one_value():
    ks = build(("2026-03-20", {"files/fees.pdf": FEES_AS_LINES, "files/fees-1.pdf": FEES_AS_TABLE}))
    answer = ks.answer("What is the courier fee for a lost card?")
    assert "$45.00" in answer.text and answer.conflict_note is None
    assert sorted(urls(answer)) == ["https://cu.test/files/fees-1.pdf", "https://cu.test/files/fees.pdf"]


def test_an_older_table_with_the_same_numbers_is_cited_and_not_called_different():
    rates = table("Heron Card", [("Heron Card Rate", "11.25% - 17.25%"), ("Intro Transfer Rate", "1.90%")], "03/01/2026",
                  header="| Rate | APR |")
    older = "# Heron Card\n\n" + table("Heron Card", [("Standing rate", "11.25%-17.25% APR")], "01/02/2026", header="| Rate | APR |")
    answer = build(("2026-03-05", {"rates": rates, "heron-card": older})).answer("What is the APR on the Heron Card?")
    assert "11.25% - 17.25% APR" in answer.text and answer.conflict_note is None
    assert "https://cu.test/heron-card" in urls(answer)


def test_a_copy_that_repeats_one_of_the_figures_given_is_cited_and_not_listed():
    answer = build(("2026-03-05", {"kestrel": KESTREL, "kestrel-lp": kestrel_copy("2.10%", "2.12%")})).answer(KESTREL_ASKED)
    assert answer.conflict_note is None
    assert urls(answer) == ["https://cu.test/kestrel", "https://cu.test/kestrel-lp"]


@pytest.mark.parametrize("dividend, apy", [("2.10%", "2.15%"), ("2.05%", "2.12%")])
def test_a_copy_that_differs_in_one_figure_is_still_listed_and_not_cited(dividend, apy):
    # It repeats one of its two figures but not the other: a real difference, as before.
    answer = build(("2026-03-05", {"kestrel": KESTREL, "kestrel-lp": kestrel_copy(dividend, apy)})).answer(KESTREL_ASKED)
    assert answer.conflict_note.startswith(f"Other copies of this item show different values: {apy} on https://cu.test/kestrel-lp")
    assert urls(answer) == ["https://cu.test/kestrel"]


def test_a_fee_in_words_only_is_still_listed():
    # "Free" states no figure, so it cannot repeat the $3.00 the answer gives.
    old = FEES_AS_TABLE.replace("March 9", "January 9").replace("| Coin wrapping | $3.00 |", "| Coin wrapping | Free |")
    answer = build(("2026-03-20", {"files/fees.pdf": FEES_AS_TABLE, "files/fees-old.pdf": old})).answer("How much is coin wrapping?")
    assert "$3.00" in answer.text and "Free on https://cu.test/files/fees-old.pdf" in answer.conflict_note
    assert "https://cu.test/files/fees-old.pdf" not in urls(answer)


def test_of_two_copies_only_the_one_that_differs_is_listed_and_its_win_explained():
    ks = build(("2026-03-05", {"kestrel": KESTREL, "kestrel-lp": kestrel_copy("2.10%", "2.12%"),
                               "blog/kestrel": kestrel_copy("2.05%", "2.07%", title="Kestrel")}))
    answer = ks.answer(KESTREL_ASKED)
    assert answer.conflict_note == ("Other copies of this item show different values: 2.07% on https://cu.test/blog/kestrel "
                                    "(article). Chosen because a product page outranks an article.")
    assert "https://cu.test/kestrel-lp" in urls(answer) and "https://cu.test/blog/kestrel" not in urls(answer)


def test_an_old_rate_that_another_term_now_pays_is_still_listed():
    # The old 12-month rate is the 24-month rate today: the copy repeats a figure the answer gives,
    # but not for its own term, so it is a real difference.
    now = table("Finch Certificates", [("12 Month", "3.00%"), ("24 Month", "3.20%")], "03/01/2026")
    old = "# Finch guide\n\n" + table("Finch Certificates", [("12 Month", "3.20%")], "01/02/2026")
    answer = build(("2026-03-05", {"rates": now, "finch-guide": old})).answer("What APY do Finch Certificates pay?")
    assert "3.20% on https://cu.test/finch-guide" in answer.conflict_note
    assert "https://cu.test/finch-guide" not in urls(answer)


def test_an_old_rate_for_other_model_years_is_still_listed():
    # The old copy's 2019-20 models paid what 2025-26 models pay today: not the same item.
    header = "| Model Year | Term | APR |"
    now = table("Grebe Boat Loans", [("2026-2025", "36 Months", "7.00%"), ("2024-2023", "36 Months", "7.50%")], "03/01/2026",
                header=header)
    old = "# Grebe boats\n\n" + table("Grebe Boat Loans", [("2025-2024", "36 Months", "7.25%"), ("2020-2019", "36 Months", "7.00%")],
                                      "01/02/2026", header=header)
    answer = build(("2026-03-05", {"rates": now, "grebe-boats": old})).answer("What is the 36 month rate on Grebe Boat Loans?")
    assert "7.00% on https://cu.test/grebe-boats" in answer.conflict_note
    assert "https://cu.test/grebe-boats" not in urls(answer)


def test_an_old_fee_equal_to_today_s_minimum_balance_is_still_listed():
    # Today the statement copy costs $5.00 and its minimum is $25.00; the old copy charged $25.00.
    now = ("Schedule of Charges, Effective March 9, 2026\n\nOther fees\n\n| Item charged | Amount | Opening min |\n"
           "| --- | --- | --- |\n| Statement copy | $5.00 | $25.00 |\n")
    old = "Schedule of Charges, Effective January 9, 2026\n\nOther fees\n\nStatement copy $25.00\n"
    answer = build(("2026-03-20", {"files/fees.pdf": now, "files/fees-old.pdf": old})).answer("What is the statement copy fee?")
    assert "$25.00 on https://cu.test/files/fees-old.pdf" in answer.conflict_note
    assert "https://cu.test/files/fees-old.pdf" not in urls(answer)
