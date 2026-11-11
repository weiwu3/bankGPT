"""How a page's tables, labelled values, fee lines and rate sentences are read into rows.

Table-driven over made-up page layouts. Most cases read one page with kb.tables.extract and check
the rows; a few ask the whole system, where the layout decides an answer. No downloads, no model.
"""
from datetime import date

import pytest

from helpers import CERTIFICATES, build, table
from kb import pages as P
from kb.tables import extract
from kb.text import effective_date, parse_amount_range, parse_term, parse_years

INF = float("inf")


def rows_of(markdown, url="https://cu.test/rates"):
    page = P.PageVersion(url=url, key=P.url_key(url), title="t", markdown=markdown, fetched="2026-03-01",
                         first_seen=0, last_seen=0, kind=P.classify(url, markdown))
    return extract(page)


def values(rows):
    """(row name, [value texts]) for each row."""
    return [(r.name, [v.text for v in r.values]) for r in rows]


# --------------------------------------------------------------------------- pipe tables

CERT_ROWS = "| 6 Month | 1.50% |\n| 12 Month | 2.00% |\n"


@pytest.mark.parametrize("layout", [
    "| Term | APY |\n| --- | --- |\n",               # the usual header
    "| Term | APY |\n|:---|---:|\n",                  # aligned columns
    "| Term | APY |\n|---|---|\n",                    # no spaces
    "| | |\n| --- | --- |\n",                         # an empty header row
])
def test_a_pipe_table_is_read_with_any_header_row(layout):
    rows = rows_of("Finch Certificates\n\n" + layout + CERT_ROWS)
    assert [(r.name, r.quals.term, r.values[0].num) for r in rows] == [("6 Month", 6, 1.5), ("12 Month", 12, 2.0)]
    assert {r.label for r in rows} == {"Finch Certificates"}


def test_a_table_without_outer_pipes_is_read():
    rows = rows_of("Finch Certificates\n\nTerm | APY\n--- | ---\n6 Month | 1.50%\n12 Month | 2.00%\n")
    assert [r.values[0].num for r in rows] == [1.5, 2.0]


def test_lines_with_pipes_but_no_separator_row_are_no_table():
    assert rows_of("Finch Certificates\n\n| 6 Month | 1.50% |\n| 12 Month | 2.00% |\n") == []


@pytest.mark.parametrize("cell, num, text", [
    ("1.50%1", 1.5, "1.50%"),             # a footnote mark glued to the value
    ("1.50%*", 1.5, "1.50%"),             # an asterisk
    ("1.50% APY2", 1.5, "1.50% APY"),
    ("1.50% APY*1", 1.5, "1.50% APY"),
    ("**1.50%**", 1.5, "1.50%"),          # emphasis
    ("1.50 %", 1.5, "1.50 %"),
])
def test_a_value_cell_keeps_its_number_whatever_marks_it_carries(cell, num, text):
    [row] = rows_of(f"Finch Certificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | {cell} |\n")
    assert (row.values[0].num, row.values[0].text) == (num, text)


@pytest.mark.parametrize("cell, name", [
    ("Stop payment &amp; reissue", "Stop payment & reissue"),
    ("Non&amp;#45;member wire", "Non-member wire"),         # escaped twice, as some PDF tables arrive
    ("Wire<br>to another bank", "Wire to another bank"),
    ("[Wire to another bank](https://cu.test/wires)", "Wire to another bank"),
])
def test_a_cell_is_read_as_the_text_it_stands_for(cell, name):
    [row] = rows_of(f"Fees\n\n| Service | Fee |\n| --- | --- |\n| {cell} | &#36;30.00 |\n")
    assert (row.name, row.values[0].text, row.values[0].num) == (name, "$30.00", 30.0)


def test_a_row_with_no_name_or_no_value_is_left_out():
    rows = rows_of("Finch Loans\n\n| Term | APR |\n| --- | --- |\n| 36 Months | 5.10% |\n|  | 5.30% |\n"
                   "| 48 Months | - |\n| 60 Months | 5.90% |\n")
    assert values(rows) == [("36 Months", ["5.10%"]), ("60 Months", ["5.90%"])]


def test_a_fee_column_of_mostly_words_is_still_a_fee_column():
    rows = rows_of("Fees\n\n| Service | Fee |\n| --- | --- |\n| Teller withdrawal | Free |\n| Balance inquiry | N/A |\n"
                   "| Coin count | - |\n| Wire out | $22.00 |\n")
    assert values(rows) == [("Teller withdrawal", ["Free"]), ("Wire out", ["$22.00"])]


def test_a_price_given_in_words_is_kept_as_its_words():
    rows = rows_of("Fees\n\n| Service | Fee |\n| --- | --- |\n| Wire to a member | $10.00 |\n| Wire to a bank | $22.00 |\n"
                   "| Foreign draft | Call for pricing |\n")
    assert values(rows)[-1] == ("Foreign draft", ["Call for pricing"])
    assert "Call for pricing" in build(("2026-03-05", {"fees": "# Fees\n\n" + table(
        "Fees", [("Wire to a member", "$10.00"), ("Foreign draft", "Call for pricing")], header="| Service | Fee |")}
    )).answer("How much is a foreign draft?").text


@pytest.mark.parametrize("markdown", [
    "Savings growth\n\n| Year | Balance |\n| --- | --- |\n| Year one | $1,000 |\n| Year two | $2,050 |\n",  # chart data
    "Branches\n\n| Branch | Town |\n| --- | --- |\n| Main | Larkspur |\n| North | Fennick |\n",           # no values
    "Members\n\n| 1.50% | 2.00% |\n| --- | --- |\n| 3.00% | 4.00% |\n",                                   # nothing names a row
])
def test_a_table_that_publishes_no_priced_value_gives_no_rows(markdown):
    assert rows_of(markdown) == []


@pytest.mark.parametrize("page, question, said", [
    ("# Specials\n\n6-month Wren certificate APY: 2.95%\n", "What is the 6-month Wren certificate APY?",
     "6-month Wren certificate APY: 2.95%."),                                # the measure is not said twice
    ("## Rates\n\nFinch Money Market\n\n| Balance | Intro Dividend Rate1 | Dividend Rate |\n| --- | --- | --- |\n"
     "| $10,000+ | 3.90% | 3.15% |\n", "Finch money market dividend rate on $20,000?",
     "Finch Money Market, $10,000+: 3.90% (Intro Dividend Rate), 3.15% (Dividend Rate)."),   # a header's footnote mark
    ("# Gold Rate Checking\n\nPaper statement fee: $3.00\n\n## Rates\n\nGold Rate Checking\n\n| Balance | APY |\n"
     "| --- | --- |\n| $5,000+ | 2.15% |\n", "What is the paper statement fee for Gold Rate Checking?",
     "Paper statement fee: $3.00."),                                         # "rate" in a product's name asks no rate
])
def test_a_value_is_worded_as_the_page_prints_it(page, question, said):
    assert build(("2026-03-05", {"page": page})).answer(question).text == said


def test_two_columns_of_one_measure_keep_their_own_headers_in_the_answer():
    page = ("## Rates\n\nFinch Money Market\n\n| Balance | Intro APY2 | APY |\n| --- | --- | --- |\n"
            "| $10,000+ | 3.90% | 3.15% |\n")
    text = build(("2026-03-05", {"money-market": page})).answer("Finch money market APY on $20,000?").text
    assert "3.90% (Intro APY), 3.15% (APY)" in text and "APY2" not in text


# --------------------------------------------------------------------------- labels, sections, dates

@pytest.mark.parametrize("markdown, label", [
    ("## Deposit rates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | 4.10% |\n", "Deposit rates"),        # the heading
    ("## Rates\n\nClassic Certificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | 4.10% |\n", "Classic Certificates"),
    ("## Rates\n\nCertificates\n\nClassic\n\n| Term | APY |\n| --- | --- |\n| 6 Month | 4.10% |\n", "Classic"),  # nearest line
])
def test_a_table_is_labelled_by_the_short_line_above_it_else_its_heading(markdown, label):
    assert [r.label for r in rows_of(markdown)] == [label]


@pytest.mark.parametrize("blank_lines", [0, 1, 3, 7])
def test_the_date_line_under_a_table_dates_it(blank_lines):
    [row] = rows_of("Finch Certificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | 1.50% |\n" + "\n" * blank_lines
                    + "Rates effective as of 03/01/2026.\n")
    assert row.effective == date(2026, 3, 1)


def test_a_date_line_nine_lines_under_a_table_still_dates_it():
    [row] = rows_of("Finch Certificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | 1.50% |\n" + "\n" * 8
                    + "Rates effective as of 03/01/2026.\n")
    assert row.effective == date(2026, 3, 1)


@pytest.mark.parametrize("between", ["## Fine print\n", "\nFinch Loans\n\n| Term | APR |\n| --- | --- |\n| 36 Months | 5.10% |\n"])
def test_a_date_line_under_the_next_heading_or_table_is_not_this_table_s(between):
    rows = rows_of("Finch Certificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | 1.50% |\n" + between
                   + "Rates effective as of 03/01/2026.\n")
    assert rows[0].effective is None


def test_a_document_s_printed_date_dates_its_fee_lines_and_a_product_page_s_does_not():
    top = "Schedule of Charges, Effective March 9, 2026\n\nAccount services\n\nStatement copy $2.00\n"
    [fee] = rows_of(top, url="https://cu.test/files/fee-schedule.pdf")
    assert (fee.kind, fee.label, fee.effective) == ("fee", "Account services", date(2026, 3, 9))
    assert rows_of(top.replace("Statement copy $2.00", "Statement copy: $2.00"), url="https://cu.test/fees")[0].effective is None


def test_a_document_s_printed_date_dates_its_tables():
    fees = ("Schedule of Charges, Effective March 9, 2026\n\nOther fees\n\n| Service | Fee |\n| --- | --- |\n"
            "| Coin wrapping | $3.00 |\n| Wire to another bank | $22.00 |\n")
    ks = build(("2026-03-20", {"files/fee-schedule.pdf": fees}))
    assert ks.answer("How much is coin wrapping?").text == "Other fees, Coin wrapping: $3.00 (rates effective March 9, 2026)."


def test_an_escaped_table_cell_is_read_as_the_text_it_stands_for():
    # A table read from a PDF can arrive HTML-escaped, some of it twice: "&amp;#45;" is a hyphen.
    fees = ("# Fee schedule\n\nOther fees\n\n| Service | Fee |\n| --- | --- |\n"
            "| Non&amp;#45;member ATM usage | $1.75 |\n| Coin counting &amp;amp; wrapping | $4.00 |\n")
    ks = build(("2026-03-05", {"files/fee-schedule.pdf": fees}))
    assert ks.answer("What is the fee for non-member ATM usage?").text == "Other fees, Non-member ATM usage: $1.75."
    assert ks.answer("How much is coin counting and wrapping?").text == "Other fees, Coin counting & wrapping: $4.00."


# --------------------------------------------------------------------------- labelled values, fee lines, sentences

@pytest.mark.parametrize("line, name, text, num, metric", [
    ("Routing Number: 271987654", "Routing Number", "271987654", None, "value"),     # an identity, kept as text
    ("Phone: (555) 010-2233", "Phone", "(555) 010-2233", None, "value"),
    ("- **Wire fee:** $18.00", "Wire fee", "$18.00", 18.0, "fee"),
    ("Minimum to open: $25", "Minimum to open", "$25", 25.0, "min_open"),
    ("6-month certificate APY: 2.95%", "6-month certificate APY", "2.95%", 2.95, "apy"),
    ("Overdraft fee: $25.00", "Overdraft fee", "$25.00", 25.0, "fee"),
])
def test_a_labelled_value_is_one_row(line, name, text, num, metric):
    [row] = rows_of(f"# Finch Checking\n\n{line}\n")
    assert (row.kind, row.name, row.values[0].text, row.values[0].num, row.values[0].metric) == ("labelled", name, text, num, metric)
    assert row.label == "Finch Checking"


@pytest.mark.parametrize("line", ["Monday: 9 a.m. - 5 p.m.", "Note: see the table below", "Hours: by appointment",
                                  "Founded: in the spring", "Extension: 12"])
def test_a_line_with_a_colon_but_no_published_value_is_no_row(line):
    assert rows_of(f"# Finch\n\n{line}\n") == []


def test_a_labelled_value_on_a_product_page_is_labelled_by_its_heading_not_a_plain_line():
    [row] = rows_of("# Finch Checking\n\nEveryday perks\n\nOverdraft fee: $25.00\n")
    assert row.label == "Finch Checking"


def test_a_label_that_starts_like_a_weekday_is_still_a_labelled_value():
    page = ("# Gold Rate Checking\n\nMonthly fee: $5.00\n\n## Rates\n\nGold Rate Checking\n\n| Balance | APY |\n"
            "| --- | --- |\n| $5,000+ | 2.15% |\n| $0 - $4,999.99 | 0.05% |\n")
    assert rows_of(page)[0].kind == "labelled"
    text = build(("2026-03-05", {"gold-rate-checking": page})).answer("What is the monthly fee for Gold Rate Checking?").text
    assert "$5.00" in text and "2.15%" not in text


def test_fee_lines_are_read_in_documents_only_and_a_wrapped_name_is_joined():
    lines = "Account services\n\nStatement copy $2.00 per page\nReturned deposit item\n(each item) $12.00\n"
    rows = rows_of("Fee Schedule\n\n" + lines, url="https://cu.test/files/fees.pdf")
    assert values(rows) == [("Statement copy", ["$2.00 per page"]), ("Returned deposit item (each item)", ["$12.00"])]
    assert rows_of("# Fees\n\n" + lines) == []


@pytest.mark.parametrize("line, expired", [
    ("Summer special: 6-month certificate APY 2.95% (offer ended 2025-09-30)", date(2025, 9, 30)),
    ("Spring promo: 6-month certificate APY 2.95%. Promotion has ended June 1, 2026.", date(2026, 6, 1)),
    ("Spring promo: 6-month certificate APY 2.95%. This special expires 07/01/2026.", date(2026, 7, 1)),
    ("Spring promo: 6-month certificate APY 2.95%.", None),
])
def test_a_rate_in_a_sentence_is_a_weak_row_that_knows_when_its_offer_ends(line, expired):
    [row] = rows_of(f"# Specials\n\n{line}\n")
    assert (row.kind, row.quals.term, row.values[0].num, row.expired) == ("claim", 6, 2.95, expired)


def test_an_offer_ended_heading_expires_the_table_under_it():
    ended = "# Specials\n\n## This offer ended January 31, 2026\n\n" + table("Wren Certificates", [("6 Month", "4.10%")], heading="")
    ks = build(("2026-03-05", {"specials": ended, "rates": table("Wren Certificates", [("6 Month", "2.60%")])}))
    text = ks.answer("What is the APY on a 6-month Wren certificate?").text
    assert "2.60%" in text and "4.10%" not in text


def test_a_fee_schedule_read_as_lines_answers_as_its_table_does():
    # The same schedule as a table or as text lines: two of the three words asked are enough for either.
    as_table = "Schedule of Charges\n\nOther fees\n\n| Service | Fee |\n| --- | --- |\n| Wage order fee | $12.00 |\n| Coin wrapping | $3.00 |\n"
    as_lines = "Schedule of Charges\n\nOther fees\n\nWage order fee $12.00\nCoin wrapping $3.00\n"
    for fees in (as_table, as_lines):
        ks = build(("2026-03-20", {"files/fee-schedule.pdf": fees}))
        assert ks.answer("What's the cost of wage order processing?").text == "Wage order fee: $12.00.", fees


def test_a_fee_line_priced_in_words_does_not_title_the_lines_under_it():
    # "Withdrawals at our ATMs Free" states its fee in a word; the card fee printed under it is not an ATM fee.
    fees = "# Fee schedule\n\nCard services\n\nWithdrawals at our ATMs Free\nCourier fee for a lost card $45.00\n"
    ks = build(("2026-03-05", {"files/fee-schedule.pdf": fees}))
    assert "Free" in ks.answer("What does an ATM withdrawal cost?").text
    assert ks.answer("What is the courier fee for a lost card?").text == "Courier fee for a lost card: $45.00."


def test_a_fee_line_does_not_make_the_question_s_measure_part_of_a_name():
    # "Gold Rate Savings" has "rate" in its name; "Courier fee for a lost card" states a fee.
    fees = "# Fee schedule\n\nCard services\n\nCourier fee for a lost card $45.00\nCoin wrapping $3.00\nWire to another bank $22.00\n"
    ks = build(("2026-03-05", {"files/fee-schedule.pdf": fees}))
    assert ks.answer("What is the courier fee for a lost card?").text == "Courier fee for a lost card: $45.00."


def test_a_dollar_sign_with_no_amount_after_it_breaks_nothing():
    assert parse_amount_range("Wire fee ($, domestic)") is None and parse_amount_range("$1MM+") is None
    ks = build(("2026-03-05", {"fees": "# Fees\n\nWire fee ($, domestic): $25\n", "rates": CERTIFICATES}))    # read at ingest
    assert "3.50%" in ks.answer("What is the APY on a 6-month certificate?").text


# --------------------------------------------------------------------------- rows that read alike

def test_rows_that_read_alike_are_told_apart_by_their_values_and_a_repeat_stays_one():
    rows = rows_of("Cards\n\n| Card | Rate |\n| --- | --- |\n| Standard Rate | 12.00% |\n| Standard Rate | 14.00% |\n"
                   "| Standard Rate | 12.00% |\n")
    assert [r.variant for r in rows] == [0, 1, 0]


FEES = "# Fees\n\n" + table("Wires", [("Wire in", "$10.00"), ("Wire out", "$22.00")], header="| Service | Fee |") \
    + table("Checks", [("Check with stub", "$2.00"), ("Check without stub", "$3.50")], header="| Service | Fee |")


@pytest.mark.parametrize("question, want, other", [
    ("What is the wire out fee?", "$22.00", "$10.00"),
    ("What was the wire out fee in January 2026?", "$22.00", "$10.00"),     # the date's "in" is no row's word
])
def test_a_question_about_one_of_two_rows_that_differ_by_a_small_word_gets_that_row(question, want, other):
    text = build(("2026-03-05", {"fees": FEES})).answer(question).text
    assert want in text and other not in text


def test_a_small_word_that_a_negation_turns_round_picks_no_row():
    text = build(("2026-03-05", {"fees": FEES})).answer("How much is a check with no stub?").text
    assert "$3.50" in text or "$2.00" not in text


def test_or_less_tiers_of_one_product_nest_into_ranges():
    rows = rows_of("Home Loans\n\n| Program | Rate |\n| --- | --- |\n| Jumbo $600,000 or less | 6.5% |\n"
                   "| Jumbo $300,000 or less | 6.4% |\n| Jumbo $600,000.01 and up | 6.9% |\n")
    assert [r.quals.amount for r in rows] == [(300000.01, 600000.0), (0.0, 300000.0), (600000.01, INF)]


def test_the_same_amount_printed_two_ways_is_one_value_not_a_conflict():
    fee = "# Fees\n\nFees\n\n| Service | Fee |\n| --- | --- |\n| Wire to another bank | {} |\n"
    ks = build(("2026-03-05", {"fees": fee.format("$25"), "wires": fee.format("$25.00")}))
    answer = ks.answer("What is the fee for a wire to another bank?")
    assert answer.conflict_note is None and ks.conflicts() == []


def test_two_rows_of_one_page_that_read_alike_are_two_facts():
    page = ("# Awards\n\n## University Award Program\n\nAwards available: 6 awards of $3,000 each, paid over three years.\n\n"
            "## Apprenticeship Award Program\n\nAwards available: 6 awards of $1,500 each, paid over two years.\n")
    ks = build(("2026-03-05", {"awards": page}))
    answer = ks.answer("How much is the apprenticeship award?")
    assert "$1,500" in answer.text and "$3,000" not in answer.text
    assert answer.conflict_note is None                  # neither is an out-of-date copy of the other


def test_a_rate_with_its_conditions_is_its_own_row():
    heloc = ("## Rates\n\nHome Equity Line of Credit\n\n| Plan | APR |\n| --- | --- |\n"
             "| Home Equity Line of Credit (HELOC) | 7.15% APR |\n"
             "| HELOC Promotional Rate | 2.49% intro APR for 9 months, then the standard rate of 7.15% APR. |\n")
    # Two downloads of the same table, so that the report has an earlier copy it could list.
    heloc += "\nRates effective as of {}.\n"
    ks = build(("2026-01-05", {"rates": heloc.format("01/02/2026")}), ("2026-03-05", {"rates": heloc.format("03/01/2026")}))
    assert "2.49%" in ks.answer("Is there an intro rate on a home equity line of credit?").text
    assert "7.15%" in ks.answer("What is the HELOC rate?").text
    assert ks.conflicts() == []                        # the intro rate and the standard rate do not contradict


# --------------------------------------------------------------------------- the qualifiers in a cell

@pytest.mark.parametrize("cell, amount", [
    ("$3,000 - $11,999.99", (3000.0, 11999.99)), ("$12,000+", (12000.0, INF)), ("$12,000 or more", (12000.0, INF)),
    ("Over $50,000", (50000.0, INF)), ("At least $1,000", (1000.0, INF)), ("Minimum of $500", (500.0, INF)),
    ("$640,000 or less", (0.0, 640000.0)), ("Up to $25k", (0.0, 25000.0)), ("Under $1,000", (0.0, 1000.0)),
    ("$1M to $2M", (1e6, 2e6)), ("$10k - $50k", (1e4, 5e4)), ("$500", (500.0, 500.0)),
    ("60 Months", None), ("1,000 points", None), ("$,", None),
])
def test_an_amount_cell_reads_as_a_range(cell, amount):
    assert parse_amount_range(cell) == amount


@pytest.mark.parametrize("cell, years", [
    ("2020-23 Model Year", (2020.0, 2023.0)), ("Model Year 2025-21", (2021.0, 2025.0)), ("2019 - 2023", (2019.0, 2023.0)),
    ("2012 Model Year (and older)", (-INF, 2012.0)), ("2024 or newer", (2024.0, INF)), ("2025", (2025.0, 2025.0)),
    ("New", None),
])
def test_a_model_year_cell_reads_as_a_range(cell, years):
    assert parse_years(cell) == years


@pytest.mark.parametrize("cell, months", [("6 Month", 6), ("12-month", 12), ("3 Mos", 3), ("1 Year", 12), ("5 yrs", 60),
                                          ("36mo", 36), ("Share Savings", None), ("2025", None)])
def test_a_term_cell_reads_as_months(cell, months):
    assert parse_term(cell) == months


@pytest.mark.parametrize("line, day", [
    ("Rates effective as of 03/01/2026.", date(2026, 3, 1)), ("Effective March 9, 2026", date(2026, 3, 9)),
    ("Effective date: 2026-03-09", date(2026, 3, 9)), ("Rates effective on Mar. 9, 2026", date(2026, 3, 9)),
    ("effective from 3/9/26", date(2026, 3, 9)),
    ("Effective immediately; reviewed March 9, 2026", None),        # the date must follow the word
    ("Rates effective as of 02/30/2026", None),                     # no such day
])
def test_an_effective_line_reads_the_date_right_after_effective(line, day):
    assert effective_date(line) == day
