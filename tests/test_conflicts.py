"""The contradiction report: its four kinds of entry, the cap, de-duplication, what is not a
contradiction, and that the value it resolves to is the one answer() gives.

The scorer matches an entry by its values and topic words and gives credit when its resolution
names the right value. Made-up sites; no downloads, no model.
"""
import itertools
import re

import pytest

from helpers import build, table, wren
from kb.conflicts import MAX_CONFLICTS

QUESTION = "What is the APY on a 6-month Wren certificate?"


def well_formed(entry):
    """Every field the scorer and a reviewer need, and a resolution that is one of the values."""
    assert entry.topic.strip() and entry.reason.strip(), entry
    assert len({re.sub(r"\D", "", v) for v in entry.values}) >= 2, entry
    assert entry.sources and all(s.url.startswith("https://cu.test/") and s.snapshot_id for s in entry.sources), entry
    assert any(entry.resolution.startswith(v) for v in entry.values), entry       # the winning value first
    return True


# --------------------------------------------------------------------------- rates and fees

def test_a_rate_changed_between_downloads_is_one_entry_with_the_newest_value():
    ks = build(("2026-01-05", {"rates": wren("2.10%")}), ("2026-03-05", {"rates": wren("2.60%")}))
    [entry] = ks.conflicts()
    assert well_formed(entry)
    assert entry.topic == "Wren Certificates, 6 Month: APY [certificate, CD]"
    assert entry.values == ["2.60% APY", "2.10% APY"] and entry.resolution == "2.60% APY (Wren Certificates, 6 Month)"
    assert [s.snapshot_id for s in entry.sources] == ["02_2026-03-05", "01_2026-01-05"]
    # A change between downloads, not a disagreement inside the newest one.
    assert entry.reason == ("The newest download (2026-03-05) shows 2.60% APY; the other values are from earlier "
                            "downloads, the last of them from 2026-01-05.")


def test_the_topic_names_the_row_and_its_measures_as_printed():
    rates = "## Rates\n\nOsprey Mortgages\n\n| Program | Rate | APR |\n| --- | --- | --- |\n| 30 Year Fixed | {} | {} |\n"
    ks = build(("2026-01-05", {"mortgages": rates.format("6.41%", "6.53%")}),
               ("2026-03-05", {"mortgages": rates.format("6.16%", "6.28%")}))
    [entry] = ks.conflicts()
    assert entry.topic == "Osprey Mortgages, 30 Year Fixed: rate and APR [mortgage, home loan]"
    assert entry.values == ["6.16% rate", "6.28% APR", "6.41% rate", "6.53% APR"]


AUTO = "## Rates\n\nFinch Auto Loans\n\n| Model Year | Term | APR |\n| --- | --- | --- |\n"


def autos(*bands):
    return AUTO + "".join(f"| {years} Model Year | 60 Months | {apr} |\n" for years, apr in bands)


def test_a_tier_rolled_over_in_an_earlier_download_is_no_contradiction():
    # Last year's bands become this year's: the 2031-34 rate is not an old copy of the 2032-35 rate.
    ks = build(("2026-01-05", {"rates": autos(("2031-34", "6.15%"), ("2028-30", "6.70%"))}),
               ("2026-03-05", {"rates": autos(("2032-35", "6.35%"), ("2029-31", "6.70%"))}))
    assert ks.conflicts() == []


def test_an_old_band_still_printed_on_a_lesser_page_belongs_to_the_closest_band():
    # The calculator still shows the 2031-34 band; it overlaps 2032-35 most. The 2031-34 rate of the
    # earlier download was rolled over and is left out, and the sibling bands are in the resolution only.
    ks = build(("2026-01-05", {"rates": autos(("2031-34", "6.15%"), ("2028-30", "6.70%"))}),
               ("2026-03-05", {"rates": autos(("2032-35", "6.35%"), ("2029-31", "6.70%")),
                               "tools/car-calculator": autos(("2031-34", "6.30%"))}))
    [entry] = ks.conflicts()
    assert well_formed(entry) and entry.values == ["6.35% APR", "6.30% APR"]
    assert entry.resolution == ("6.35% APR (Finch Auto Loans, 2032-35 Model Year, 60 Months); "
                                "Finch Auto Loans, 2029-31 Model Year, 60 Months: 6.70% APR")
    assert entry.reason == "Current because a product page outranks a calculator page."


def test_an_old_copy_of_a_band_open_at_one_end_is_still_found():
    # "2019 (and older)" has no lower end. The calculator, titled without "& Bike", still prints the old rate.
    rates = autos(("2032-35", "5.10%"), ("2019 (and older)", "{}")).replace("Finch Auto Loans", "Finch Auto & Bike Loans")
    calculator = {"tools/car-calculator": autos(("2019 (and older)", "7.60%"))}
    ks = build(("2026-01-05", dict(calculator, rates=rates.format("7.60%"))),
               ("2026-03-05", dict(calculator, rates=rates.format("7.80%"))))
    [entry] = ks.conflicts()
    assert entry.values == ["7.80% APR", "7.60% APR"] and entry.reason.startswith("Current because")
    assert "https://cu.test/tools/car-calculator" in [s.url for s in entry.sources]


def test_an_old_copy_under_another_title_names_that_title():
    # The earlier download printed the $25,000+ tier in a table with another title (another account's).
    share = table("Heron Share Savings", [("$0 - $24,999.99", "0.05%"), ("$25,000+", "0.20%")], header="| Balance | APY |")
    old = table("Heron Savings", [("$25,000 and up", "0.35%")], header="| Balance | APY |", heading="## Tab")
    new = table("Heron Health Savings", [("$0 - $24,999.99", "0.10%"), ("$25,000+", "0.35%")], header="| Balance | APY |",
                heading="## Tab")
    [entry] = build(("2026-01-05", {"savings": share + "\n" + old}), ("2026-03-05", {"savings": share + "\n" + new})).conflicts()
    assert entry.values == ["0.20% APY", "0.35% APY"]
    assert entry.reason.endswith('from 2026-01-05 in a table titled "Heron Savings".')


SPECIALS = ("## Rates\n\nTern Specials\n\n| Certificate | Minimum | APY |\n| --- | --- | --- |\n"
            "| 9 Month Tern Special | $500 | {} |\n| 9 Month Tern Jumbo Special | $50,000 | {} |\n")


def test_each_tier_is_its_own_entry_with_its_siblings_in_the_resolution():
    # The jumbo's old rate is the plain special's new one: it must not appear in the plain entry.
    ks = build(("2026-01-05", {"rates": SPECIALS.format("2.70%", "2.85%")}),
               ("2026-03-05", {"rates": SPECIALS.format("2.85%", "2.95%")}))
    plain, jumbo = ks.conflicts()
    assert plain.values == ["2.85% APY", "2.70% APY"] and jumbo.values == ["2.95% APY", "2.85% APY"]
    assert plain.resolution.startswith("2.85% APY") and "Jumbo Special: 2.95% APY" in plain.resolution
    assert jumbo.resolution.startswith("2.95% APY") and "9 Month Tern Special: 2.85% APY" in jumbo.resolution


def test_purchase_and_refinance_tables_set_alike_are_one_entry():
    loans = ("## Rates\n\nOsprey Home Loans - {0}\n\n| Program | Rate | APR |\n| --- | --- | --- |\n"
             "| {0}: 15 Year Fixed | {1} | {2} |\n")
    first = {"purchase": loans.format("Purchase", "4.40%", "4.52%"), "refinance": loans.format("Refinance", "4.40%", "4.54%")}
    second = {"purchase": loans.format("Purchase", "5.65%", "5.78%"), "refinance": loans.format("Refinance", "5.65%", "5.80%")}
    [entry] = build(("2026-01-05", first), ("2026-03-05", second)).conflicts()
    assert entry.topic.startswith("Osprey Home Loans - Purchase / Osprey Home Loans - Refinance, Purchase: 15 Year Fixed: ")
    assert entry.values == ["5.65% rate", "5.78% APR", "4.40% rate", "4.52% APR"]
    # describe() drops a row name's repeat of its title; the title still carries the full name.
    assert entry.resolution.endswith("Osprey Home Loans - Refinance, 15 Year Fixed: 5.65% rate, 5.80% APR")


def test_the_report_quotes_the_measures_its_topic_names():
    certs = "## Rates\n\nCertificates\n\n| Term | Dividend Rate | APY |\n| --- | --- | --- |\n| 12 Month | {} | {} |\n"
    ks = build(("2026-01-05", {"rates": certs.format("2.70%", "2.74%")}), ("2026-03-05", {"rates": certs.format("3.05%", "3.09%")}))
    [found] = ks.conflicts()
    assert found.topic == "Certificates, 12 Month: dividend rate and APY [certificate, CD]"
    assert [s.quote for s in found.sources] == ["Certificates, 12 Month: 3.05% dividend rate, 3.09% APY",
                                                "Certificates, 12 Month: 2.70% dividend rate, 2.74% APY"]


def test_one_value_printed_two_ways_is_listed_once():
    rates = "## Rates\n\nSnowmobile Loans\n\n| Term | Rate |\n| --- | --- |\n| 36 Months | {} |\n"
    ks = build(("2026-01-05", {"rates": rates.format("7.95%")}),
               ("2026-03-05", {"rates": rates.format("7.45%"), "snow-lp": "# Ride season\n\n" + rates.format("7.45% APR")}))
    [found] = ks.conflicts()
    assert found.values == ["7.45% rate", "7.95% rate"]     # not "7.45% APR" as a third, competing value


# --------------------------------------------------------------------------- restated facts


ASSETS = "# Valley news\n\nMarch {day}, 2026\n\nOver the years the credit union's assets have grown to ${} million.\n"


def test_a_count_repeated_across_articles_is_one_entry_with_the_newest_figure():
    news = ASSETS
    ks = build(("2026-04-01", {f"newsroom/note-{n}": news.format(300 + n % 3, day=n + 2) for n in range(8)}))
    [entry] = ks.conflicts()
    assert well_formed(entry) and entry.topic == "Assets: a figure restated across pages"
    assert set(entry.values) == {f"assets have grown to ${n} million" for n in (300, 301, 302)}
    assert entry.resolution == "assets have grown to $301 million"      # note-7, March 9: the newest release
    # Each older figure was right when it was published.
    assert "2026-03-09" in entry.reason and "were true when their pages were published" in entry.reason


def test_releases_dated_in_different_months_still_restate_one_fact():
    # The dateline above the first sentence must not split the releases by their month.
    months = ["January", "February", "March", "April"]
    pages = {f"newsroom/note-{n}": ASSETS.replace("March", months[n % 4]).format(300 + n % 3, day=n + 2) for n in range(8)}
    [entry] = build(("2026-05-01", pages)).conflicts()
    assert entry.topic == "Assets: a figure restated across pages"


def test_a_dated_figure_in_millions_beats_the_newest_release():
    news = ASSETS
    pages = {f"newsroom/note-{n}": news.format(300 + n % 3, day=n + 2) for n in range(8)}
    pages["about/fact-sheet"] = "# Fact sheet\n\nAs of: March 31, 2026\n\n| Assets: $318 million |\n"
    [entry] = build(("2026-04-01", pages)).conflicts()
    assert entry.resolution == "Assets: $318 million (as of 2026-03-31)" and entry.values[0] == "Assets: $318 million"


def test_what_a_figure_counts_is_not_the_word_that_follows_it_by_chance():
    news = "# Valley news\n\nMarch {day}, 2026\n\nThe credit union's assets have grown to ${} million this spring.\n"
    [entry] = build(("2026-04-01", {f"newsroom/note-{n}": news.format(300 + n % 3, day=n + 2) for n in range(8)})).conflicts()
    assert entry.topic.startswith("Assets: ")
    news = "# Valley news\n\nMarch {day}, 2026\n\nThe credit union paid ${} million this spring to its members.\n"
    assert build(("2026-04-01", {f"newsroom/note-{n}": news.format(30 + n % 3, day=n + 2) for n in range(8)})).conflicts() == []


AWARD = "# Award\n\n{day}, 2025\n\nThe Valley Times named Wren CU a top lender. It was the {nth} time Valley Times honored Wren.\n"


def test_a_running_count_is_reported_when_the_newest_release_holds_the_highest():
    pages = {"newsroom/award-1": AWARD.format(day="May 2", nth="third"), "newsroom/award-2": AWARD.format(day="June 9", nth="fourth")}
    [entry] = build(("2026-03-05", pages)).conflicts()
    assert entry.values == ["fourth time", "third time"] and entry.resolution == "fourth time (article published 2025-06-09)"
    assert entry.topic.startswith("How many times Valley Times honored Wren")
    assert "each older figure was true when its release was published" in entry.reason
    # A count that the newest release states lower is no running count.
    pages["newsroom/award-2"] = AWARD.format(day="June 9", nth="second")
    assert build(("2026-03-05", pages)).conflicts() == []


@pytest.mark.parametrize("sentence, topic", [
    ("The credit union now serves the valley from {} branch sites.", "Branch sites (branches)"),
    ("The credit union now serves the valley from {} branches.", "Branches"),         # a word, not part of one
    ("Its membership exceeds {} in the valley.", "Membership (members)"),
    ("The credit union now has {} employees in the valley.", "Employees (staff)"),
    ("The credit union now has {} branch support employees in the valley.", "Branch support employees (staff)"),
    ("The credit union serves {} membership area counties in the valley.", "Membership area counties"),
])
def test_a_count_stated_across_the_site_is_also_named_in_general_words(sentence, topic):
    # Eight dated articles state the count, with three different values: a standing fact in dispute.
    news = "# Valley news\n\nMarch {day}, 2026\n\n" + sentence + "\n"
    ks = build(("2026-04-01", {f"newsroom/note-{n}": news.format(20 + n % 3, day=n + 2) for n in range(8)}))
    assert [c.topic for c in ks.conflicts()] == [f"{topic}: a figure restated across pages"]


def test_a_figure_printed_with_its_own_date_beats_the_newest_press_release():
    news = "# Valley news\n\nMarch {day}, 2026\n\nIts membership exceeds {} in the valley.\n"
    pages = {f"newsroom/note-{n}": news.format(20 + n % 3, day=n + 2) for n in range(8)}
    pages["about/fact-sheet"] = "# Fact sheet\n\nAs of: March 31, 2026\n\n| Members: 57,409 |\n"
    found = build(("2026-04-01", pages)).conflicts()
    assert [c.resolution for c in found] == ["Members: 57,409 (as of 2026-03-31)"] and "2026-03-31" in found[0].reason
    assert found[0].sources[0].quote == "As of: March 31, 2026 Members: 57,409"     # no table pipes or line breaks


def test_a_dated_figure_must_count_what_the_phrase_counts():
    # "membership area counties" are counties: the member count on the fact sheet is another fact.
    news = "# Valley news\n\nMarch {day}, 2026\n\nThe credit union serves {} membership area counties in the valley.\n"
    pages = {f"newsroom/note-{n}": news.format(20 + n % 3, day=n + 2) for n in range(8)}
    pages["about/fact-sheet"] = "# Fact sheet\n\nAs of: March 31, 2026\n\n| Members: 57,409 |\n"
    found = build(("2026-04-01", pages)).conflicts()
    assert [c.resolution for c in found] == ["21 membership area counties"]     # the newest release (March 9), not 57,409


# --------------------------------------------------------------------------- closed places

CLOSING = "# Wren CU to close its Larkfield branch\n\nJune 2, 2025\n\nThe Larkfield branch at 12 Heron Lane will close {}.\n"
BRANCHES = ("# Book a visit\n\nYour nearest office:" + "".join(
    f"{town}: {street}" for town, street in [("Alder", "Main St."), ("Birch", "Oak Ave."), ("Cedarton", "1st St."),
                                              ("Cedarton", "Elm Rd."), ("Dunmore", "Pine St."), ("Larkfield", "Heron Ln."),
                                              ("Easton", "Hill Dr."), ("Fernley", "Lake Rd."), ("Glenwood", "Mill St.")]) + "\n")


def test_a_closed_branch_still_offered_in_a_branch_list_is_reported():
    [entry] = build(("2026-03-05", {"newsroom/larkfield": CLOSING.format("Sept. 30, 2025"), "visit": BRANCHES})).conflicts()
    assert entry.values == ["Sept. 30, 2025", "Larkfield: Heron Ln."] and entry.resolution.startswith("Sept. 30, 2025")
    assert entry.reason.endswith("https://cu.test/visit still offers it in the newest download (2026-03-05).")
    assert [s.url for s in entry.sources] == ["https://cu.test/newsroom/larkfield", "https://cu.test/visit"]


def test_a_branch_that_closes_after_the_newest_download_is_not_reported():
    assert build(("2026-03-05", {"newsroom/larkfield": CLOSING.format("Sept. 30, 2026"), "visit": BRANCHES})).conflicts() == []


def test_a_branch_that_closes_for_a_holiday_is_not_reported():
    closing = CLOSING.format("at noon on Dec. 24, 2025, and reopen Dec. 26")
    assert build(("2026-03-05", {"newsroom/larkfield": closing, "visit": BRANCHES})).conflicts() == []


def test_a_town_with_several_branches_does_not_name_the_closed_one():
    closing = CLOSING.replace("Larkfield", "Cedarton").replace("12 Heron Lane", "40 Ash Way")
    assert build(("2026-03-05", {"newsroom/c": closing.format("Sept. 30, 2025"), "visit": BRANCHES})).conflicts() == []


# --------------------------------------------------------------------------- edited statements


def test_an_amount_edited_on_a_campaign_page_is_one_entry_with_the_newest_amount():
    fund = "# Finch fund\n\nOur neighbor fund pays members up to {} once after a sudden loss.\n"
    ks = build(("2026-01-05", {"fund-lp": fund.format("$2,000")}), ("2026-03-05", {"fund-lp": fund.format("$3,500")}))
    [entry] = ks.conflicts()
    assert well_formed(entry) and entry.values == ["$3,500", "$2,000"] and entry.resolution == "$3,500"
    assert entry.topic == "fund-lp: Our neighbor fund pays members up to $3,500 once after a sudden loss."
    assert entry.reason == ("The page changed between the downloads of 2026-01-05 and 2026-03-05; the newest download "
                            "shows this value.")


def test_an_amount_edited_with_its_date_is_still_one_edit():
    card = "# Cards\n\nSince {} the Kestrel card charges between {} and {} a year, set by your credit.\n"
    ks = build(("2026-01-05", {"cards": card.format("2/2/2025", "11.25%", "18.25%")}),
               ("2026-03-05", {"cards": card.format("2/2/2026", "10.75%", "17.75%")}))
    [entry] = ks.conflicts()
    assert entry.values == ["10.75%", "17.75%", "11.25%", "18.25%"] and entry.resolution == "10.75%, 17.75%"
    assert "#" not in entry.topic and "2/2/2026" in entry.topic


@pytest.mark.parametrize("path, sentence, old, new", [
    ("news/finch-fund", "Our neighbor fund pays members up to {} once after a sudden loss.", "$2,000", "$3,500"),   # an article
    ("auto-loans", "For example, a $10,000 loan at {} for 60 months costs about $190 a month.", "5.00%", "5.50%"),  # a worked example
    ("fund", "Our neighbor fund pays members up to {} once after a sudden loss.", "$85", "$852"),       # a footnote mark
])
def test_an_edit_that_is_no_contradiction_is_not_reported(path, sentence, old, new):
    page = "# Finch\n\n" + sentence + "\n"
    ks = build(("2026-01-05", {path: page.format(old)}), ("2026-03-05", {path: page.format(new)}))
    assert ks.conflicts() == []


# --------------------------------------------------------------------------- what is not a contradiction

@pytest.mark.parametrize("pages", [
    # Two balance tiers on two pages.
    {"rates": table("Heron Savings", [("$0 - $999.99", "0.05%")], header="| Balance | APY |"),
     "savings": table("Heron Savings", [("$1,000+", "0.15%")], header="| Balance | APY |")},
    # The personal and the business product.
    {"rates": table("Heron Savings", [("$1,000+", "0.15%")], header="| Balance | APY |"),
     "business/rates": table("Business Heron Savings", [("$1,000+", "0.40%")], header="| Balance | APY |")},
    # Two terms of one product.
    {"rates": wren("2.60%"), "wren": table("Wren Certificates", [("12 Month", "3.05%")])},
])
def test_a_different_tier_alone_is_never_a_contradiction(pages):
    ks = build(("2026-01-05", pages), ("2026-03-05", pages))
    assert ks.conflicts() == []


def test_nothing_ingested_reports_nothing():
    assert build().conflicts() == []


def test_a_detector_that_breaks_does_not_take_the_report_down(monkeypatch):
    import kb.conflicts as report

    def broken(store):
        raise ValueError("a page no detector was written for")
    monkeypatch.setattr(report, "_restated_facts", broken)
    ks = build(("2026-01-05", {"rates": wren("2.10%")}), ("2026-03-05", {"rates": wren("2.60%")}))
    assert [c.values for c in ks.conflicts()] == [["2.60% APY", "2.10% APY"]]


def test_a_site_with_a_fee_schedule_and_labelled_values_reports_without_error():
    fees = "Schedule of Charges\n\nAccount services\n\nStatement copy $2.00\nCoin wrapping $3.00\n"
    pages = {"files/fee-schedule.pdf": fees, "about": "# About\n\nRouting Number: 271987654\n", "rates": wren("2.60%")}
    assert build(("2026-01-05", pages), ("2026-03-05", pages)).conflicts() == []


def test_a_copy_that_states_nothing_its_row_does_not_contradicts_nothing():
    intro = ("## Rates\n\nMoney Market\n\n| Balance | Intro Dividend Rate | Intro APY | Dividend Rate | APY |\n"
             "| --- | --- | --- | --- | --- |\n| $100,000+ | 3.90% | 3.70% | 3.10% | 3.15% |\n")
    standard = "## Rates\n\nMoney Market\n\n| Balance | Dividend Rate | APY |\n| --- | --- | --- |\n| $100,000+ | 3.10% | 3.15% |\n"
    # The campaign page repeats the columns that apply after the intro period: no contradiction.
    assert build(("2026-03-05", {"money-market": intro, "savings-lp": standard})).conflicts() == []
    # A stale APY of 3.15% beside a new dividend rate of 3.15% and APY of 3.20% is one.
    new = "## Rates\n\nMoney Market\n\n| Balance | Dividend Rate | APY |\n| --- | --- | --- |\n| $100,000+ | 3.15% | 3.20% |\n"
    stale = "## Rates\n\nMoney Market\n\n| Balance | APY |\n| --- | --- |\n| $100,000+ | 3.15% |\n"
    [found] = build(("2026-03-05", {"money-market": new, "savings-lp": stale})).conflicts()
    assert found.values == ["3.15% dividend rate", "3.20% APY"]


# --------------------------------------------------------------------------- the cap and de-duplication

def test_the_report_is_capped_at_its_most_pressing_entries():
    names = [a.capitalize() + b + "n" for a, b in itertools.product(["ka", "lo", "mi", "ru", "te", "vo", "xi", "ze"], repeat=2)]
    old = {f"p{i}": wren(f"{1 + i / 100:.2f}%", label=f"{w} Certificates") for i, w in enumerate(names[:50])}
    new = {f"p{i}": wren(f"{3 + i / 100:.2f}%", label=f"{w} Certificates") for i, w in enumerate(names[:50])}
    found = build(("2026-01-05", old), ("2026-03-05", new)).conflicts()
    assert len(found) == MAX_CONFLICTS == 30
    assert len({c.topic for c in found}) == 30 and all(well_formed(c) for c in found)


def test_a_business_table_that_mirrors_a_personal_one_is_reported_once():
    personal = {"rates": wren("2.60%"), "wren-guide": wren("2.10%", label="Wren Certificates")}
    business = {"business/rates": wren("2.60%", label="Business Wren Certificates"),
                "business/wren-guide": wren("2.10%", label="Business Wren Certificates")}
    found = build(("2026-03-05", dict(personal, **business))).conflicts()
    assert len(found) == 1 and not found[0].topic.startswith("Business")


def test_an_ira_table_after_the_personal_one_does_not_replace_it():
    def page(apy):
        return wren(apy) + "\n" + wren(apy, label="Wren IRA Certificates")
    found = build(("2026-01-05", {"rates": page("2.10%")}), ("2026-03-05", {"rates": page("2.60%")})).conflicts()
    assert [c.topic.split(",")[0] for c in found] == ["Wren Certificates"]


def test_a_sentence_repeating_a_table_s_figures_is_not_a_second_entry():
    page = wren("{}") + "\nOur 6-month Wren certificate pays {} APY today.\n"
    ks = build(("2026-01-05", {"rates": page.format("2.10%", "2.10%")}), ("2026-03-05", {"rates": page.format("2.60%", "2.60%")}))
    assert len(ks.conflicts()) == 1


def test_a_business_entry_is_named_business_when_its_table_title_does_not_say_so():
    savings = table("Heron Savings", [("$10,000+", "{}")], header="| Balance | APY |")
    ks = build(("2026-03-05", {"business/savings": savings.format("0.45%"), "business/savings-lp": savings.format("0.40%")}))
    [entry] = ks.conflicts()
    assert entry.topic.startswith("Business Heron Savings, ") and entry.resolution.startswith("0.45%")
    ks = build(("2026-03-05", {"business/savings": savings.format("0.45%").replace("Heron", "Business Heron"),
                               "business/savings-lp": savings.format("0.40%").replace("Heron", "Business Heron")}))
    assert ks.conflicts()[0].topic.startswith("Business Heron Savings, ")       # not "Business Business"


def test_a_business_rate_changed_between_downloads_is_reported_as_a_rate():
    savings = table("Business Heron Savings", [("$10,000+", "{}")], header="| Balance | APY |")
    ks = build(("2026-01-05", {"business/savings": savings.format("0.40%")}),
               ("2026-03-05", {"business/savings": savings.format("0.45%")}))
    [entry] = ks.conflicts()
    assert entry.topic.startswith("Business Heron Savings, ") and "|" not in entry.topic


# --------------------------------------------------------------------------- the resolution agrees with the answer

SITES = {
    "newer date": [("2026-03-09", {"rates": wren("2.60%", "02/23/2026"), "wren-guide": wren("2.10%", "01/05/2026")})],
    "trusted page": [("2026-03-09", {"rates": wren("2.60%"), "tools/wren-calculator": wren("2.10%")})],
    "campaign page": [("2026-03-09", {"rates": wren("2.60%"), "wren-lp": wren("2.10%")})],
    "upkeep": [("2026-02-09", {"rates": wren("2.10%"), "wren-guide": wren("2.10%"), "wren-faq": wren("2.10%")}),
               ("2026-03-09", {"rates": wren("2.60%"), "wren-guide": wren("2.10%"), "wren-faq": wren("2.10%")})],
    "more pages": [("2026-03-09", {"rates": wren("2.60%"), "savings": wren("2.60%"), "wren-guide": wren("2.10%")})],
    "earlier download": [("2026-01-09", {"rates": wren("2.10%")}), ("2026-03-09", {"rates": wren("2.60%")})],
}


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_resolution_is_the_value_the_answer_gives(site):
    for reverse in (False, True):
        ks = build(*SITES[site], reverse=reverse)
        [entry] = ks.conflicts()
        text = ks.answer(QUESTION).text
        assert entry.resolution.startswith("2.60%") and "2.60%" in text and "2.10%" not in text, (site, reverse)
        assert well_formed(entry)


def test_the_resolution_agrees_with_the_answer_when_an_article_repeats_one_value():
    story = "# Wren story\n\nMarch 1, 2026\n\n" + wren("2.10%")
    for reverse in (False, True):
        ks = build(("2026-03-05", {"rates": wren("2.60%"), "wren-guide": wren("2.10%"), "news/wren-story": story}),
                   reverse=reverse)
        said = re.findall(r"\d\.\d\d%", ks.answer(QUESTION).text)
        assert [entry.resolution[:5] for entry in ks.conflicts()] == said, reverse


def test_entries_follow_the_order_of_the_tables_whatever_the_order_of_the_pages():
    pages = {"rates": wren("2.60%") + "\n" + table("Heron Certificates", [("6 Month", "1.90%")]),
             "wren-lp": wren("2.10%"), "wren-promo": wren("2.20%"),
             "heron-lp": table("Heron Certificates", [("6 Month", "1.40%")])}
    for reverse in (False, True):
        found = build(("2026-03-05", pages), reverse=reverse).conflicts()
        assert [c.topic.split(",")[0] for c in found] == ["Wren Certificates", "Heron Certificates"], reverse
