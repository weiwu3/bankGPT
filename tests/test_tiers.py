"""Which tier a question asks for: terms, balances, loan amounts, model years, personal or business.

A row for another tier is a different fact, and a value from it in the answer scores zero. Each case
names the one tier that must be answered; every other tier's value of the same table must be absent.
The site is made up. No downloads, no model.
"""
import pytest

from helpers import CERTIFICATES, REFUSAL, build, has_digit, model_reply, path_of, table, with_reader

TERMS = [("3 Month", "1.15%"), ("6 Month", "1.65%"), ("12 Month", "2.05%"), ("24 Month", "2.45%"),
         ("36 Month", "2.80%"), ("60 Month", "3.10%")]
BALANCES = [("$0 - $999.99", "0.05%"), ("$1,000 - $9,999.99", "0.15%"), ("$10,000 - $49,999.99", "0.35%"),
            ("$50,000+", "0.55%")]
BUSINESS = [("$0 - $9,999.99", "0.10%"), ("$10,000+", "0.40%")]
HOME = [("30 Year Fixed $300,000 or less", "6.40%"), ("30 Year Fixed $600,000 or less", "6.55%"),
        ("30 Year Fixed $600,000.01 and up", "6.90%")]
ROADSTER = ("## Rates\n\nRoadster Loans\n\n| Model Year | Term | APR |\n| --- | --- | --- |\n"
            "| 2025-27 Model Year | 60 Months | 4.15% |\n| 2021-24 Model Year | 60 Months | 4.65% |\n"
            "| 2020 Model Year (and older) | 60 Months | 6.25% |\n")
SPECIALS = ("## Rates\n\nSpecial Certificates\n\n| Certificate | APY | Min. to Earn |\n| --- | --- | --- |\n"
            "| 9 Month Bulk Special | 3.80% | $50,000.00 |\n| 9 Month Special | 3.70% | $500.00 |\n")
MORTGAGE = ("## Rates\n\nOsprey Mortgages\n\n| Program | Rate | APR | Points |\n| --- | --- | --- | --- |\n"
            "| 30 Year Fixed | 6.35% | 6.47% | 0.250 |\n| 15 Year Fixed | 5.70% | 5.88% | 0.125 |\n")
SHARE_SAVINGS = table("Share Savings", [("$12,000+", "0.42%"), ("$3,000 - $11,999.99", "0.27%"),
                                       ("$0 - $2,999.99", "0.06%")], "03/01/2026", header="| Balance Tier | APY |")
AUTO = ("## Rates\n\nAuto Loans\n\n| Model Year | Term | APR |\n| --- | --- | --- |\n"
        "| 2025-27 Model Year | 72 Months | 4.85% |\n| 2022-24 Model Year | 72 Months | 5.35% |\n")
PERSONAL = ("## Rates\n\nPersonal Loan\n\n| Term | APR |\n| --- | --- |\n"
            "| 24 Months | 8.75% |\n| 36 Months | 9.25% |\n| 48 Months | 9.85% |\n")
SLEDS = ("## Rates\n\nSnowmobile Loans\n\n| Model Year | Term | APR |\n| --- | --- | --- |\n"
         "| 2021-22 Model Year | 48 Months | 7.45% |\n| 2016 Model Year (and older) | 48 Months | 9.15% |\n")
HELOC = ("## Rates\n\nHome Equity Line of Credit\n\n| Plan | APR |\n| --- | --- |\n| HELOC | 7.15% APR |\n"
         "| HELOC Introductory Rate | 2.49% APR |\n")


@pytest.fixture(scope="module")
def ks():
    return build(("2026-02-10", {
        "certificates": table("Wren Certificates", TERMS, "02/02/2026"),
        "savings": table("Heron Savings", BALANCES, "02/02/2026", header="| Balance | APY |"),
        "business/savings": table("Business Heron Savings", BUSINESS, "02/02/2026", header="| Balance | APY |"),
        "home-loans": table("Osprey Home Loans", HOME, "02/02/2026", header="| Program | Rate |"),
        "roadster-loans": ROADSTER, "specials": SPECIALS, "mortgages": MORTGAGE, "heloc": HELOC}))


def given(ks, question, rows):
    """The values of `rows` that the answer gives."""
    text = ks.answer(question).text
    return {value for _, value in rows if value in text}


# --------------------------------------------------------------------------- terms

@pytest.mark.parametrize("asked, want", [
    ("a 1-year", {"2.05%"}), ("a 2 year", {"2.45%"}), ("a 60 month", {"3.10%"}), ("a 5-year", {"3.10%"}),
    ("a 36-mo", {"2.80%"}), ("a 1 yr", {"2.05%"}), ("a 3 mo", {"1.15%"}), ("a twelve month", {"2.05%"}),
    ("a six-month", {"1.65%"}), ("a thirty-six month", {"2.80%"}), ("a 24-month", {"2.45%"}),
])
def test_a_term_written_any_way_gets_that_term_only(ks, asked, want):
    assert given(ks, f"What is the APY on {asked} Wren certificate?", TERMS) == want


@pytest.mark.parametrize("asked, want", [("a 12 mos", {"2.05%"}), ("a 6mo", {"1.65%"})])
def test_a_term_abbreviated_as_mos_or_glued_to_its_number_gets_that_term(ks, asked, want):
    assert given(ks, f"What is the APY on {asked} Wren certificate?", TERMS) == want


@pytest.mark.parametrize("bound, want", [
    ("under 12 months", {"1.15%", "1.65%"}), ("below 6 months", {"1.15%"}), ("fewer than 6 months", {"1.15%"}),
    ("less than 1 year", {"1.15%", "1.65%"}), ("shorter than 2 years", {"1.15%", "1.65%", "2.05%"}),
    ("more than 24 months", {"2.80%", "3.10%"}), ("longer than 2 years", {"2.80%", "3.10%"}),
    ("greater than 36 months", {"3.10%"}),
])
def test_a_term_bound_gets_every_term_inside_it_and_no_other(ks, bound, want):
    assert given(ks, f"Wren certificate APY for terms {bound}?", TERMS) == want


@pytest.mark.parametrize("bound, want", [
    ("at least 24 months", {"2.45%", "2.80%", "3.10%"}), ("24 months or longer", {"2.45%", "2.80%", "3.10%"}),
    ("no less than 3 years", {"2.80%", "3.10%"}), ("of 36 months and up", {"2.80%", "3.10%"}),
    ("at most 6 months", {"1.15%", "1.65%"}), ("up to 1 year", {"1.15%", "1.65%", "2.05%"}),
    ("no more than 6 months", {"1.15%", "1.65%"}), ("6 months or less", {"1.15%", "1.65%"}),
    ("at least 12 months but under 36 months", {"2.05%", "2.45%"}),       # every bound counts
])
def test_a_bound_that_includes_its_own_term_gets_that_term_too(ks, bound, want):
    assert given(ks, f"Wren certificate APY for terms of {bound}?", TERMS) == want


@pytest.mark.parametrize("question, product", [
    ("Wren certificate APY for terms less than 1 year?", ["wren", "certificate"]),
    ("Wren certificate APY for 12 mos on 10k?", ["wren", "certificate"]),     # units glued to their numbers
    ("What is the rate on a 401k rollover?", ["401k", "rollover"]),           # a plan, not $401,000
])
def test_the_words_of_a_bound_or_a_unit_name_no_product(question, product):
    from kb.lookup import parse_question
    assert parse_question(question).product == product


def test_a_term_the_table_does_not_list_is_declined_without_another_term_s_value(ks):
    answer = ks.answer("What is the APY on an 18-month Wren certificate?")
    assert answer.text == REFUSAL and not answer.confident


@pytest.mark.parametrize("word, want", [("highest", "3.10%"), ("maximum", "3.10%"), ("best", "3.10%"), ("top", "3.10%"),
                                        ("most", "3.10%"), ("lowest", "1.15%"), ("cheapest", "1.15%"), ("least", "1.15%"),
                                        ("smallest", "1.15%")])
def test_a_superlative_picks_the_extreme_row(ks, word, want):
    assert given(ks, f"What is the {word} Wren certificate APY?", TERMS) == {want}


@pytest.mark.parametrize("asked, want, other", [
    ("shorter than 12 months", "3.50%", "3.10%"),       # a bound, not the 12-month term
    ("under 12 months", "3.50%", "3.10%"),
    ("longer than 6 months", "3.10%", "3.50%"),
    ("of 12 months", "3.10%", "3.50%"),                 # a term named
    ("paid over 12 months", "3.10%", "3.50%"),          # so is "over": "spread it over 72 months" names a term
])
def test_a_term_bound_is_not_a_term(asked, want, other):
    answer = build(("2026-03-05", {"rates": CERTIFICATES})).answer(f"What is the APY on a certificate {asked}?").text
    assert want in answer and other not in answer, answer


def test_a_term_in_words_is_read_as_its_number():
    ks = build(("2026-03-05", {"rates": CERTIFICATES}))
    assert "3.10%" in ks.answer("apy on a twelve month certificate").text


# --------------------------------------------------------------------------- balances and amounts

@pytest.mark.parametrize("amount, want", [
    ("$0", "0.05%"), ("$999.99", "0.05%"), ("$1,000", "0.15%"), ("$9,999.99", "0.15%"), ("$10,000", "0.35%"),
    ("$49,999.99", "0.35%"), ("$50,000", "0.55%"), ("$2,000,000", "0.55%"),
])
def test_each_balance_boundary_belongs_to_its_own_tier(ks, amount, want):
    assert given(ks, f"Heron Savings APY on a balance of {amount}?", BALANCES) == {want}


@pytest.mark.parametrize("amount", ["$10k", "$10K", "10k", "10 grand", "ten thousand dollars", "10,000 dollars",
                                    "10 thousand", "$10.5k", "$ 10,000", "$10,000.00", "$12k"])
def test_an_amount_written_any_way_is_the_same_amount(ks, amount):
    assert given(ks, f"Heron Savings APY on {amount}?", BALANCES) == {"0.35%"}


@pytest.mark.parametrize("asked, want", [
    ("over $49,999.99", "0.55%"), ("more than $9,999.99", "0.35%"), ("under $1,000", "0.05%"),
    ("less than $10,000", "0.15%"), ("between $1,000 and $5,000", "0.15%"), ("1.5 million dollars", "0.55%"),
])
def test_an_amount_with_a_bound_or_a_range_gets_the_tier_it_falls_in(ks, asked, want):
    assert given(ks, f"Heron Savings APY for {asked}?", BALANCES) == {want}


@pytest.mark.parametrize("asked", ["I have 12000 in Heron Savings, what APY?", "Heron Savings APY with a balance of 12000?",
                                   "What APY if I deposit 12000 in Heron Savings?"])
def test_a_plain_number_after_a_word_for_holding_money_is_the_amount(ks, asked):
    assert given(ks, asked, BALANCES) == {"0.35%"}


@pytest.mark.parametrize("asked", ["With 720 credit score, what is the Heron Savings APY?",
                                   "I put 3000 down on a roadster, what APR?", "I can put 6000 toward a roadster, APR?"])
def test_a_plain_number_that_is_a_score_or_a_down_payment_is_no_amount(asked):
    from kb.lookup import parse_question
    assert parse_question(asked).quals.amount is None


def test_a_plain_number_to_borrow_is_the_amount_before_one_held():
    from kb.lookup import parse_question
    assert parse_question("I have 8000 and want to borrow 20000 for a roadster, APR?").quals.amount == (20000.0, 20000.0)


def test_of_two_amounts_the_one_next_to_a_balance_word_is_the_tier(ks):
    assert given(ks, "What APY does Heron Savings pay on a $12,000 balance with the $150 offer?", BALANCES) == {"0.35%"}


def test_of_two_amounts_with_no_balance_word_the_larger_is_the_tier(ks):
    assert given(ks, "Heron Savings: what APY if I add $300 today and $12,000 in June?", BALANCES) == {"0.35%"}


def test_a_bonus_named_before_the_balance_is_not_the_tier(ks):
    assert given(ks, "Heron Savings APY with the $150 offer on a $12,000 balance?", BALANCES) == {"0.35%"}


@pytest.mark.parametrize("amount, want", [
    ("$250,000", "6.40%"), ("$300,000", "6.40%"), ("$300,000.01", "6.55%"), ("$450k", "6.55%"),
    ("$600,000", "6.55%"), ("$600,001", "6.90%"), ("1.2 million", "6.90%"),
])
def test_or_less_loan_tiers_nest_and_each_edge_belongs_to_the_lower_one(ks, amount, want):
    assert given(ks, f"Osprey home loan rate, 30 year, for {amount}?", HOME) == {want}


def test_a_30_year_fixed_rate_question_gets_only_its_amount_tier(ks):
    assert given(ks, "What is the 30-year fixed rate on a $250,000 loan?", HOME) == {"6.40%"}


@pytest.mark.parametrize("question, metrics", [("Do you offer fixed-rate mortgages?", set()),
                                               ("Is a variable rate HELOC available?", set()),
                                               ("What is the 30-year fixed rate?", {"rate"})])
def test_fixed_rate_before_a_product_names_the_product_and_alone_asks_the_rate(question, metrics):
    from kb.lookup import parse_question
    assert parse_question(question).metrics == metrics


@pytest.mark.parametrize("amount, want", [("$500", {"3.70%"}), ("$5,000", {"3.70%"}), ("$50,000", {"3.80%"}),
                                          ("$60,000", {"3.80%"})])
def test_an_amount_that_reaches_a_stated_minimum_exactly_buys_that_tier(ks, amount, want):
    rows = [("", "3.70%"), ("", "3.80%")]
    assert given(ks, f"What APY does {amount} earn in the 9-month special?", rows) == want


def test_an_amount_below_every_minimum_is_declined(ks):
    answer = ks.answer("What APY does $499 earn in the 9-month special?")
    assert answer.text == REFUSAL and not answer.confident


def test_the_highest_tier_is_the_top_balance_tier():
    ks = build(("2026-03-05", {"rates": SHARE_SAVINGS}))
    assert "0.42%" in ks.answer("What is the highest Share Savings APY?").text


def test_with_no_amount_asked_every_minimum_is_given(ks):
    either = ks.answer("What APY does the 9-month special pay?").text
    assert "3.70%" in either and "3.80%" in either      # every tier, each with its minimum


def test_a_dollar_amount_shaped_like_a_year_is_an_amount():
    ks = build(("2026-03-05", {"rates": SHARE_SAVINGS, "loans": SLEDS}))
    for asked in ("$2000", "$1999", "$ 2050"):          # a dollar sign makes it money, never a model year
        answer = ks.answer(f"What APY does Share Savings pay on {asked}?").text
        assert "0.06%" in answer and "0.27%" not in answer and "0.42%" not in answer, asked
    answer = ks.answer("Rate on a $2000 loan for a 2021 snowmobile over 48 months?").text
    assert "7.45%" in answer and "9.15%" not in answer


def test_a_down_payment_is_not_the_loan_amount():
    homes = table("Home Loans", [("30-Year Fixed $400,000 or less", "6.80%"), ("30-Year Fixed $100,000 or less", "6.60%")],
                  "03/01/2026", header="| Program | Rate |")
    ks = build(("2026-03-05", {"home-loans": homes}))
    for asked in ("with $2000 down", "with a $4,500 down payment", "with a down payment of $2000"):
        answer = ks.answer(f"What is the rate on a 30-year mortgage {asked}?").text
        assert "6.80%" in answer and "6.60%" in answer, asked
    answer = ks.answer("What is the rate on a 30-year mortgage for a $250,000 house with $2000 down?").text
    assert "6.80%" in answer and "6.60%" not in answer


def test_the_size_a_question_gives_picks_that_size_of_rows_that_read_alike():
    fees = ("# Fee schedule\n\nStorage lockers\n\n| Service | Fee |\n| --- | --- |\n| Storage locker 2 X 4 X 18 | $30.00 |\n"
            "| Storage locker 4 X 4 X 18 | $45.00 |\n| Storage locker 2 X 8 X 18 | $55.00 |\n")
    ks = build(("2026-03-05", {"files/fee-schedule.pdf": fees}))
    assert ks.answer("What is the fee for a 2 x 4 storage locker?").text == "Storage locker 2 X 4 X 18: $30.00."
    assert ks.answer("How much is the 4 by 4 storage locker?").text == "Storage locker 4 X 4 X 18: $45.00."
    every = ks.answer("What is the fee for a storage locker?").text      # no size given: every size
    assert all(fee in every for fee in ("$30.00", "$45.00", "$55.00"))


# --------------------------------------------------------------------------- model years

@pytest.mark.parametrize("year, want", [(2027, "4.15%"), (2025, "4.15%"), (2024, "4.65%"), (2021, "4.65%"),
                                        (2020, "6.25%"), (2012, "6.25%"), (1999, "6.25%")])
def test_each_model_year_edge_belongs_to_its_own_range(ks, year, want):
    rows = [("", "4.15%"), ("", "4.65%"), ("", "6.25%")]
    assert given(ks, f"Roadster loan APR for a {year} model year, 60 months?", rows) == {want}


def test_a_model_year_newer_than_every_range_is_declined(ks):
    answer = ks.answer("Roadster loan APR for a 2028 model year, 60 months?")
    assert answer.text == REFUSAL and not answer.confident


def test_a_model_year_range_of_the_last_century_is_read_without_the_words_model_year():
    page = ("## Rates\n\nClassic Roadster Loans\n\n| Vehicle | APR |\n| --- | --- |\n| 1995-2009 | 7.75% |\n"
            "| 2010-2018 | 6.95% |\n")
    answer = build(("2026-02-10", {"classic-loans": page})).answer("Classic roadster loan APR for a 2001 model?").text
    assert "7.75%" in answer and "6.95%" not in answer


def test_a_car_older_than_every_listed_model_year_is_declined():
    page = "## Rates\n\nRoadster Loans\n\n| Model Year | APR |\n| --- | --- |\n| 2024 and newer | 4.15% |\n| 2019-2023 | 4.65% |\n"
    answer = build(("2026-02-10", {"roadster-loans": page})).answer("Roadster loan APR for a 2018 model year?")
    assert not answer.confident and "4.15%" not in answer.text and "4.65%" not in answer.text


def test_a_car_for_sale_is_no_tier_of_a_model_year_rate():
    cars = "# Lot cars\n\n| Year | Make | Model | Price |\n| --- | --- | --- | --- |\n| 2022 | Roadster | Coupe | $18,500 |\n"
    pages = {"roadster-loans": ROADSTER, "cars-for-sale": cars}
    text = build(("2026-02-10", pages)).answer("What is the APR on a 2022 Roadster Coupe?").text
    assert "4.65%" in text and "$18,500" not in text


def test_model_year_written_with_a_hyphen_is_the_same_qualifier():
    ks = build(("2026-03-05", {"auto-loans": AUTO}))
    for asked in ("model year", "model-year"):
        answer = ks.answer(f"What's the 72-month APR for financing a 2022 {asked} truck?").text
        assert answer == "Auto Loans, 2022-24 Model Year, 72 Months: 5.35% APR.", asked


@pytest.mark.parametrize("day", ["05/31/2026", "5/31/26", "2026/05/31", "05/2026"])
def test_a_date_written_in_numbers_is_no_model_year(day):
    answer = build(("2026-03-05", {"auto-loans": AUTO})).answer(f"What was the 72-month auto loan APR on {day}?").text
    assert "4.85%" in answer and "5.35%" in answer            # no model year was asked: both tiers


def test_a_rate_question_is_not_answered_with_cars_for_sale():
    # A car's make ("Buick") makes it an "auto" row too, but its minimum bid is no loan rate.
    cars = ("## Vehicles for sale\n\nRepossessed vehicles\n\n| Year | Make | Model | Minimum Bid |\n| --- | --- | --- | --- |\n"
            "| 2018 | Buick | Encore | $8,300.00 |\n| 2017 | Ford | Fusion | $6,950.00 |\n")
    ks = build(("2026-03-05", {"auto-loans": AUTO, "vehicles-for-sale": cars}))
    answer = ks.answer("What are your auto loan rates?").text
    assert "4.85%" in answer and "5.35%" in answer and "$8,300" not in answer and "$6,950" not in answer


@pytest.mark.parametrize("question, want, other", [
    ("Personal loan APR 36 months?", "9.25%", ("8.75%", "9.85%")),             # "APR 36" is no April 36
    ("What was the June 36 month personal loan APR?", "9.25%", ("8.75%", "9.85%")),       # "June 36 month" is a term
    ("72-month auto loan APR 2022 model?", "5.35%", ("4.85%",)),              # "APR 2022" is a model year
])
def test_apr_before_a_number_is_a_rate_not_april(question, want, other):
    answer = build(("2026-03-05", {"loans": PERSONAL, "auto-loans": AUTO})).answer(question).text
    assert want in answer and not any(o in answer for o in other), answer


def test_april_with_a_full_date_is_still_a_date():
    from kb.lookup import parse_question
    q = parse_question("As of Apr 15, 2026, what was the 36-month personal loan APR?")
    assert q.quals.term == 36 and q.quals.years is None


@pytest.mark.parametrize("question, want, other", [
    ("In Apr 2026, what was the 72-month auto loan APR for a 2023 car?", ("5.35%",), ("4.85%",)),   # 2023 is the model year
    ("Apr 2026: 72 month auto loan rate for a 2023 car?", ("5.35%",), ("4.85%",)),
    ("What was the 72-month auto loan APR in mid-Apr 2026?", ("4.85%", "5.35%"), ()),            # no model year: both tiers
])
def test_apr_before_a_year_alone_is_still_april(question, want, other):
    answer = build(("2026-03-05", {"auto-loans": AUTO})).answer(question).text
    assert all(w in answer for w in want) and not any(o in answer for o in other), answer


@pytest.mark.parametrize("when", ["in Apr 2026", "on Apr 5", "as of Apr 15", "in mid-Apr", "in late Apr"])
def test_apr_as_a_month_asks_no_rate(when):
    # A branch question dated in April is no APR question, so no rule declines it for want of a rate.
    harbor = "# Harbor branch\n\nThe Harbor branch lobby opens Saturdays 9-noon.\n"
    ks = with_reader(model_reply("Yes, Saturdays 9-noon."), ("2026-03-05", {"auto-loans": AUTO, "harbor": harbor}))
    assert path_of(ks, f"Was the Harbor branch open on Saturdays {when}?") == "reader"


@pytest.mark.parametrize("question", [
    "What's the APR 7/1 ARM?", "APR 3 vs 5 year certificate", "best apr 4 wheeler loan",
    "What is the APR 2 percent card?", "APR 7 days a week?", "what is the late apr on a card",
    "Is the HELOC APR 6 and fixed?", "Is the card APR 13, or higher?", "What is the APR 15 and 30 year mortgage?",
    "Auto APR 6, 2024 model?",
])
def test_apr_before_a_number_that_is_no_day_asks_the_rate(question):
    # A day makes "apr" April only after a date word ("on Apr 5?") or before a year ("Apr 5, 2026"); a
    # part of a month only after a date word ("in late Apr").
    from kb.lookup import parse_question
    assert "apr" in parse_question(question).metrics


def test_apr_as_a_month_is_no_search_word():
    from kb.passages import PassageIndex
    words = PassageIndex(build(("2026-03-05", {"auto-loans": AUTO})).engine.store.view_at(None)).query_tokens
    assert "apr" not in words("Where was the Harbor branch in mid-Apr 2026?")
    assert "apr" in words("What's the APR 7/1 ARM in May 2026?")


@pytest.mark.parametrize("question", ["APR 2023 auto loan 72 months as of May 2026", "auto loan APR 2023 72 months in 2026?"])
def test_apr_before_a_model_year_is_the_rate_when_the_later_year_is_a_date(question):
    # The year after "APR" is the model year: the later one dates the question ("May 2026", "in 2026").
    answer = build(("2026-03-05", {"auto-loans": AUTO})).answer(question).text
    assert "5.35%" in answer and "4.85%" not in answer, answer


@pytest.mark.parametrize("vehicle", ["snowmobiles", "snowmobile loans", "a 2021 snowmobile"])
def test_a_year_before_a_vehicle_is_its_model_year(vehicle):
    asked = vehicle if vehicle.startswith("a ") else f"2021 {vehicle}"
    answer = build(("2026-03-05", {"loans": SLEDS})).answer(f"What is the 48-month APR for {asked}?").text
    assert "7.45%" in answer and "9.15%" not in answer               # "for 2021 ..." is no date here


# --------------------------------------------------------------------------- personal and business

@pytest.mark.parametrize("question, want", [
    ("Business Heron Savings APY on $20,000?", "0.40%"),
    ("Heron Savings APY for my business with $20,000?", "0.40%"),
    ("Heron Savings APY for business with $20,000?", "0.40%"),
    ("Heron Savings APY for businesses with $20,000?", "0.40%"),
    ("Heron Savings APY for small businesses with $20,000?", "0.40%"),
    ("Small business Heron Savings APY on $5k?", "0.10%"),
    ("Biz Heron Savings APY on $5k?", "0.10%"),
    ("Commercial Heron Savings APY on $15,000?", "0.40%"),
])
def test_a_business_question_gets_the_business_tier_and_never_a_personal_one(ks, question, want):
    text = ks.answer(question).text
    assert want in text and not given(ks, question, BALANCES)


def test_a_personal_question_never_gets_a_business_value(ks):
    text = ks.answer("Heron Savings APY on $20,000?").text
    assert "0.35%" in text and not any(v in text for _, v in BUSINESS)


def test_a_fee_every_member_pays_is_the_answer_for_businesses_too():
    # The night drop key fee is charged to every member and has no business row; the deposit bag is a business fee.
    fees = ("# Fee schedule\n\nAccount services\n\nNight drop key $7.00\nCoin wrapping $3.00\n\n"
            "Business services\n\nBusiness deposit bag $9.00\nBusiness statement copy $4.00\n")
    ks = build(("2026-03-05", {"files/fee-schedule.pdf": fees}))
    assert ks.answer("What is the night drop key fee for businesses?").text == "Night drop key: $7.00."
    assert ks.answer("What is the deposit bag fee for businesses?").text == "Business deposit bag: $9.00."


# --------------------------------------------------------------------------- what else a question can name

def test_points_are_given_when_asked_for(ks):
    assert "0.250 points" in ks.answer("How many points on the 30 year Osprey mortgage?").text
    assert "points" not in ks.answer("What is the 30 year Osprey mortgage rate?").text


@pytest.mark.parametrize("question", ["What is the non-introductory HELOC rate?", "What is the HELOC rate, not the intro one?"])
def test_a_tier_the_question_excludes_is_left_out(ks, question):
    text = ks.answer(question).text
    assert "7.15%" in text and "2.49%" not in text


def test_a_refusal_for_a_missing_tier_states_no_number_but_the_question_s(ks):
    for question in ("Wren certificate APY for 48 months?", "Roadster loan APR for a 2031 model year, 60 months?"):
        answer = ks.answer(question)
        assert path_of(ks, question) == "declined"
        assert not has_digit(answer.text)
