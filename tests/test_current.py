"""Each rule that decides which copy of a published value is current, pinned by a test of its own.

When pages print the same row (same product, term and tier) with different values,
RowIndex.better in solution/kb/lookup.py compares two copies by a fixed list of rules, and the
first rule that tells them apart decides. The tests are numbered 1 to 9 as those comparisons
are, in their order (the ninth settles a tie that nothing the site says does). Each builds a small made-up site in which every rule before the one under test
ties, and the first rule after it that tells the copies apart, if any, would pick the other copy:
taking that one rule out of RowIndex.better changes the answer. Each site is built twice, its pages
listed in both orders, so that the order of the pages cannot stand in for the rule. The tests after
them pin what feeds the rules: a page demoted for its out-of-date tables, a date carried back from a
later download, and copies whose rules go round in a circle.

No downloads, no network and no language model are needed.
"""
import itertools

import pytest

from helpers import build, table
from kb.compose import reason

NEW, OLD = "3.65%", "2.85%"                 # the value that must win, and the one that must lose
QUESTION = "What is the APY on a 6-month Kestrel certificate?"


def both_orders(*downloads):
    """The same site twice, its pages listed as written and then backwards."""
    return [build(*downloads), build(*downloads, reverse=True)]


def kestrel(rows, effective=None):
    """The Kestrel Certificates table: (term, APY) rows, and the date printed under it, if any."""
    return table("Kestrel Certificates", rows, effective)


def six_month(apy, effective=None):
    """The same table with its 6-month row only."""
    return kestrel([("6 Month", apy)], effective)


def guide(intro, apy):
    """A guide page: an introduction of its own above the 6-month table."""
    return f"# Kestrel guide\n\n{intro}\n\n" + six_month(apy)


def answered(ks, question=QUESTION, as_of=None):
    """The answer, checked to give the value that must win and never the one that must lose."""
    answer = ks.answer(question, as_of)
    assert NEW in answer.text and OLD not in answer.text, answer.text
    return answer


# --------------------------------------------------------------------------- the rules, in their order


@pytest.mark.parametrize("term", ["6", "12"])          # a two-digit term is still a term
def test_rule_1_an_ended_promotion_loses(term):
    # Two product pages state the rate in a line of text, neither dated, in one download: only the
    # "offer ended" on one of the lines tells the copies apart.
    ended = f"{term}-month Kestrel certificate APY: 2.85% (offer ended January 31, 2026)"
    pages = {"specials": f"# Kestrel specials\n\n{ended}\n",
             "kestrel-rates": f"# Rates\n\n{term}-month Kestrel certificate APY: 3.65%\n"}
    for ks in both_orders(("2026-03-09", pages)):
        why = "the other value is a promotion that ended 2026-01-31."
        assert answered(ks, f"What is the APY on a {term}-month Kestrel certificate?").conflict_note.endswith(
            "Chosen because " + why)
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why


@pytest.mark.parametrize("where", ["heading", "sentence"])
def test_rule_1_an_offer_ended_above_a_table_ends_its_rows(where):
    # The same undated table on two product pages in one download; on the specials page an end date
    # sits in the heading above it, or in the sentence right above it. Every later rule ties.
    if where == "heading":
        specials = "# Winter special (offer ended January 31, 2026)\n\n" + kestrel([("6 Month", OLD)])
    else:
        specials = ("# Winter special\n\nThis offer ended January 31, 2026.\n\nKestrel Certificates\n\n"
                    f"| Term | APY |\n| --- | --- |\n| 6 Month | {OLD} |\n")
    pages = {"specials": specials, "kestrel-rates": six_month(NEW)}
    for ks in both_orders(("2026-03-09", pages)):
        assert answered(ks).conflict_note.endswith("Chosen because the other value is a promotion that ended "
                                                   "2026-01-31.")


def test_an_offer_that_ends_later_has_not_ended_yet():
    # "Ends March 31" in a download of March 9: the offer still runs, and its table stays in play.
    specials = "# Winter special (offer ends March 31, 2026)\n\n" + kestrel([("6 Month", "2.95%")], "03/02/2026")
    pages = {"specials": specials, "rates": six_month(OLD, "02/02/2026")}
    for ks in both_orders(("2026-03-09", pages)):
        answer = ks.answer(QUESTION)
        assert "2.95%" in answer.text and OLD not in answer.text, answer.text


@pytest.mark.parametrize("trusted, other, winner, loser", [
    ("kestrel-rates", "rates.pdf", "product page", "published document"),
    ("files/kestrel-rate-sheet.pdf", "kestrel-promo", "published document", "campaign page"),
    ("kestrel-promo", "news/kestrel", "campaign page", "article"),
    ("kestrel-rates", "calculator", "product page", "calculator page"),
])
def test_rule_2_the_more_trusted_kind_of_page_wins(trusted, other, winner, loser):
    # The same undated table on two kinds of page in one download, no promotion (rule 1 ties): the kind
    # of page is all that differs. The less trusted page has the shorter address, which rule 9 prefers.
    pages = {trusted: six_month(NEW), other: six_month(OLD)}
    for ks in both_orders(("2026-03-09", pages)):
        note = answered(ks).conflict_note
        assert note.endswith(f"Chosen because a {winner} outranks {'an' if loser == 'article' else 'a'} {loser}."), note


def test_rule_3_the_newer_effective_date_wins():
    # Two product pages print the table under different dates. Their 12-month rows agree, so the page with
    # the older date is not demoted as left behind (rule 2 ties), and the dates decide the 6-month row.
    pages = {"kestrel-guide": kestrel([("6 Month", NEW), ("12 Month", "3.15%")], "02/23/2026"),
             "rates": kestrel([("6 Month", OLD), ("12 Month", "3.15%")], "01/05/2026")}
    for ks in both_orders(("2026-03-09", pages)):
        why = "its rates are effective 2026-02-23, newer than the other copy's 2026-01-05."
        assert answered(ks).conflict_note.endswith("Chosen because " + why)
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why
    # A copy beaten on date counts for no page under rule 5, which would then pick the newer copy too. With
    # an undated copy of the old value on a third page, each value has one page: only the dates decide.
    pages["kestrel-faq"] = kestrel([("6 Month", OLD), ("12 Month", "3.15%")])
    for ks in both_orders(("2026-03-09", pages)):
        answered(ks)
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why


def test_rule_4_a_copy_its_page_keeps_updating_beats_copies_that_never_change():
    # Three product pages show 2.85% undated in the first download; in the second only the rates page has
    # changed it. No two copies have dates to compare (rules 1-3 tie), and rule 5 would pick the 2.85% that
    # two pages still show.
    first = {"rates": six_month(OLD), "kestrel-guide": six_month(OLD), "kestrel-faq": six_month(OLD)}
    second = dict(first, rates=six_month(NEW))
    for ks in both_orders(("2026-02-09", first), ("2026-03-09", second)):
        why = "its page keeps updating this copy; the other copy never changes."
        assert answered(ks).conflict_note.endswith("Chosen because " + why)
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why


def test_rule_5_the_value_shown_on_more_pages_wins():
    # Three product pages in one download, none dated: two show 3.65%, one 2.85%. Every other rule ties.
    pages = {"rates": six_month(OLD), "kestrel-guide": six_month(NEW), "kestrel-savings": six_month(NEW)}
    for ks in both_orders(("2026-03-09", pages)):
        assert answered(ks).conflict_note.endswith("Chosen because more pages show this value.")
        [found] = ks.conflicts()
        assert found.reason == "Current because more pages show this value."


def test_rule_6_a_copy_with_an_effective_date_beats_one_without():
    # Only the rates page prints a date; neither table ever changes and each value is on one page (rules 1-5
    # tie). The guide's introduction changed in the second download, so rule 7 would pick its copy.
    rates = six_month(NEW, "02/16/2026")
    downloads = [("2026-02-09", {"rates": rates, "kestrel-guide": guide("Compare the Kestrel terms.", OLD)}),
                 ("2026-03-09", {"rates": rates, "kestrel-guide": guide("Compare the Kestrel terms below.", OLD)})]
    for ks in both_orders(*downloads):
        answer = answered(ks)
        assert answer.text.endswith("(rates effective February 16, 2026).")
        why = "it carries an effective date (2026-02-16) and the other copy does not."
        assert answer.conflict_note.endswith("Chosen because " + why)
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why


def test_rule_7_the_more_recently_downloaded_page_wins():
    # Neither table is dated or ever changes, and each value is on one page (rules 1-6 tie). The guide's
    # introduction changed in the second download, so its copy is the more recently downloaded text.
    rates = six_month(OLD)
    downloads = [("2026-02-09", {"rates": rates, "kestrel-guide": guide("Compare the Kestrel terms.", NEW)}),
                 ("2026-03-09", {"rates": rates, "kestrel-guide": guide("Compare the Kestrel terms below.", NEW)})]
    for ks in both_orders(*downloads):
        assert answered(ks).conflict_note.endswith("Chosen because it is on the more recently downloaded page.")
        [found] = ks.conflicts()
        assert found.reason == "Current because it is on the more recently downloaded page."


def test_rule_8_a_table_beats_a_sentence():
    # Two product pages in one download, neither dated: one prints the rate in a table, the other in a
    # sentence. Every earlier rule ties.
    rates = ("## Rates\n\n| Certificate | Term | APY |\n| --- | --- | --- |\n"
             "| Kestrel certificate | 6 months | 3.65% |\n")
    sentence = "# Kestrel guide\n\nThe 6-month Kestrel certificate rate is 2.85% APY.\n"
    pages = {"rates": sentence, "kestrel-guide": rates}
    for ks in both_orders(("2026-03-09", pages)):
        why = "it is printed in a table and the other copy in a sentence."
        assert answered(ks).conflict_note.endswith("Chosen because " + why)
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why


@pytest.mark.parametrize("pages, why", [
    ({"kestrel": kestrel([("6 Month", NEW)]), "kestrel-plus": kestrel([("6 Month", OLD)])}, "its page has the shorter address"),
    ({"kestrel-a": kestrel([("6 Month", NEW)]), "kestrel-b": kestrel([("6 Month", OLD)])},
     "its page's address comes first in alphabetical order"),
])
def test_rule_9_with_nothing_else_the_page_nearer_the_top_of_the_site_wins(pages, why):
    # Two product pages in one download, both tables, neither dated: every earlier rule ties.
    why = f"nothing else tells the copies apart, and {why}."
    loser = next(path for path, md in pages.items() if OLD in md)
    for ks in both_orders(("2026-03-09", pages)):
        note = answered(ks).conflict_note
        assert note.endswith("Chosen because " + why) and f"{OLD} on https://cu.test/{loser}" in note, note
        [found] = ks.conflicts()
        assert found.reason == "Current because " + why


# --------------------------------------------------------------------------- what feeds the rules


def test_a_page_whose_out_of_date_rows_outnumber_its_current_ones_loses_to_a_document():
    # The guide's table is dated before the rates page's, so its rows that differ are out of date and those
    # that agree are current. Only the guide and an undated rate sheet print the 24-month row.
    rates = kestrel([("6 Month", "3.15%"), ("12 Month", "3.35%"), ("18 Month", "3.45%")], "02/23/2026")
    sheet = "Kestrel rate sheet\n\n" + kestrel([("24 Month", NEW)])

    def site(six, twelve, eighteen):
        return {"rates": rates, "files/kestrel-rate-sheet.pdf": sheet,
                "kestrel-guide": kestrel([("6 Month", six), ("12 Month", twelve), ("18 Month", eighteen),
                                        ("24 Month", OLD)], "01/05/2026")}
    question = "What is the APY on a 24-month Kestrel certificate?"
    # Three rows out of date, one current: the guide is trusted no more than a campaign page.
    for ks in both_orders(("2026-03-09", site("2.35%", "2.65%", "1.65%"))):
        assert answered(ks, question).conflict_note.endswith(
            "Chosen because the other copy is on a page that was left behind: its rate tables are older than the "
            "same tables elsewhere.")
    # One row out of date, three current: the guide is still a product page and outranks the document.
    for ks in both_orders(("2026-03-09", site("2.35%", "3.35%", "3.45%"))):
        answer = ks.answer(question)
        assert OLD in answer.text and NEW not in answer.text, answer.text
        assert answer.conflict_note.endswith("Chosen because a product page outranks a published document.")


def test_a_date_printed_in_a_later_download_dates_the_earlier_copy():
    # The rates page prints its table's date only in the second download. As of the first, that date makes its
    # copy newer than the guide's (rule 3); without it, the guide's dated copy would win (rule 6).
    rows = [("6 Month", NEW), ("12 Month", "3.15%")]
    old = kestrel([("6 Month", OLD), ("12 Month", "3.15%")], "01/05/2026")
    downloads = [("2026-03-09", {"rates": kestrel(rows), "kestrel-guide": old}),
                 ("2026-04-06", {"rates": kestrel(rows, "02/23/2026"), "kestrel-guide": old})]
    for ks in both_orders(*downloads):
        answer = answered(ks, as_of="2026-03-16")
        assert answer.text.endswith("(rates effective February 23, 2026).")
        why = "its rates are effective 2026-02-23, newer than the other copy's 2026-01-05."
        assert answer.conflict_note.endswith("Chosen because " + why)



def test_a_date_line_far_under_its_table_still_dates_it():
    # The rates page prints its date after a dozen blank lines and a footnote; the guide right under its
    # table. Both are the date of their own table, so the newer date wins (rule 3). Read only up to the
    # next heading, the rates date is not taken for the table that follows it.
    gap = "\n" * 12
    rates = (six_month(NEW) + gap + "1 APY = Annual Percentage Yield.\n\nRates effective as of 02/23/2026.\n\n"
             "## Other rates\n\nKestrel IRA Certificates\n\n| Term | APY |\n| --- | --- |\n| 12 Month | 3.05% |\n")
    pages = {"rates": rates, "kestrel-guide": six_month(OLD, "01/05/2026")}
    for ks in both_orders(("2026-03-09", pages)):
        answer = answered(ks)
        assert answer.text.endswith("(rates effective February 23, 2026).")
        ira = ks.answer("What is the APY on a 12-month Kestrel IRA certificate?")
        assert "3.05%" in ira.text and "effective" not in ira.text, ira.text


def test_a_date_worked_out_from_a_change_does_not_lose_to_a_newer_printed_date():
    # The rates page changed its undated table between the downloads, so its new copy came after
    # 2026-02-09: a lower bound, not a date anyone printed. The guide prints 02/16/2026 on the old value.
    # The change may have come after 02/16 too, so the dates do not tell, and the page that keeps its
    # table up to date wins (rule 4). The guide is not marked as left behind either.
    downloads = [("2026-02-09", {"rates": six_month(OLD)}),
                 ("2026-03-09", {"rates": six_month(NEW), "kestrel-guide": six_month(OLD, "02/16/2026")})]
    for ks in both_orders(*downloads):
        answer = answered(ks)
        assert "effective" not in answer.text, answer.text
        assert answer.conflict_note.endswith("Chosen because its page keeps updating this copy; the other copy "
                                             "never changes.")
        assert ks.engine.store.latest.stale == set()


def test_a_date_worked_out_from_a_change_beats_an_older_printed_date():
    # The same, but the guide's date is older than the download before the change: the change came after it,
    # so the guide's copy is out of date, and the guide is a page left behind.
    downloads = [("2026-02-09", {"rates": six_month(OLD)}),
                 ("2026-03-09", {"rates": six_month(NEW), "kestrel-guide": six_month(OLD, "01/05/2026")})]
    for ks in both_orders(*downloads):
        assert answered(ks).conflict_note.endswith("Chosen because the other copy is on a page that was left behind: "
                                                   "its rate tables are older than the same tables elsewhere.")
        assert ks.engine.store.latest.stale == {"cu.test/kestrel-guide"}


@pytest.mark.parametrize("third", ["the same page", "only its introduction edited"])
def test_a_change_date_is_kept_when_the_page_is_downloaded_again_with_the_same_table(third):
    # The site of the test above, downloaded a third time with the same tables. Whether the rates page is
    # byte for byte the same or only its introduction was edited, its table still dates from the change.
    rates = "# Rates\n\nOur rates.\n\n" + six_month(NEW)
    if third != "the same page":
        rates_again = rates.replace("Our rates.", "Our current rates.")
    else:
        rates_again = rates
    guide = six_month(OLD, "02/16/2026")
    downloads = [("2026-02-09", {"rates": "# Rates\n\nOur rates.\n\n" + six_month(OLD)}),
                 ("2026-03-09", {"rates": rates, "kestrel-guide": guide}),
                 ("2026-04-06", {"rates": rates_again, "kestrel-guide": guide})]
    for ks in both_orders(*downloads):
        answered(ks)
        [row] = [r for r in ks.engine.store.latest.pages["cu.test/rates"].tables if r.kind == "table"]
        assert (row.effective.isoformat(), row.dated_by) == ("2026-02-09", "change")


def test_an_undated_copy_does_not_count_the_page_of_an_old_edition_against_the_newest_date():
    # One download, three product pages. A special dated 02/23 says 2.85%; a guide dated 01/05 and an
    # undated rates page say 3.65%. Their 12-month rows agree, so no page is left behind (rule 2 ties).
    # The guide's copy lost on date (rule 3), so it does not count for 3.65% under rule 5: one page
    # each, and the copy with a printed date wins (rule 6). 2.85% must win.
    def two_terms(six, effective=None):
        return kestrel([("6 Month", six), ("12 Month", "3.15%")], effective)
    pages = {"kestrel-special": two_terms(OLD, "02/23/2026"), "kestrel-guide": two_terms(NEW, "01/05/2026"),
             "rates": two_terms(NEW)}
    for ks in both_orders(("2026-03-09", pages)):
        answer = ks.answer(QUESTION)
        assert OLD in answer.text and NEW not in answer.text, answer.text
        index = ks.engine.store.latest.cache["rows"]
        [win, rates] = [r for r in index.rows if r.quals.term == 6 and r.page.key in ("cu.test/kestrel-special",
                                                                                      "cu.test/rates")]
        if win.page.key == "cu.test/rates":
            win, rates = rates, win
        assert reason(index, win, rates) == "it carries an effective date (2026-02-23) and the other copy does not"
        [found] = [f for f in ks.conflicts() if "6 Month" in f.topic]
        assert found.resolution.startswith(OLD), found.resolution


def test_the_answer_and_the_conflict_report_name_the_same_winner():
    # One download: the rates page says 2.85%, the savings page and an article 3.65%, none dated. The
    # article's copy counts for 3.65% under rule 5 in the answer, so it must in the report too.
    pages = {"rates": six_month(OLD), "savings": six_month(NEW), "news/kestrel-rates-explained": six_month(NEW)}
    for ks in both_orders(("2026-03-09", pages)):
        assert answered(ks).conflict_note.endswith("Chosen because more pages show this value.")
        [found] = ks.conflicts()
        assert found.resolution.startswith(NEW) and found.reason == "Current because more pages show this value."


def test_a_date_printed_later_dates_an_earlier_copy_whatever_was_inferred_before():
    # May 19 shows the new rate undated, so it is first dated to the download before (May 12, a lower
    # bound). June 9 prints the same table with its real date; that date must replace the guess.
    plain = table("Certificates", [("6 Month", "3.50%")])
    ks = build(("2026-05-12", {"rates": table("Certificates", [("6 Month", "2.95%")])}), ("2026-05-19", {"rates": plain}),
               ("2026-06-09", {"rates": table("Certificates", [("6 Month", "3.50%")], "05/15/2026")}))
    answer = ks.answer("What was the APY on a 6-month certificate?", as_of="2026-05-20").text
    assert "3.50%" in answer and "(rates effective May 15, 2026)" in answer


def test_the_current_copy_does_not_depend_on_the_order_of_the_pages():
    # Pairwise the rules go round: the frozen copy dated 04/22 beats the kept-up one dated 04/06, which
    # beats the kept-up undated one, which beats the frozen one. The download order must not decide.
    frozen = table("Certificates", [("6 Month", "3.00%")], "04/22/2026")
    downloads = [("2026-05-12", {"special-rates": frozen,
                                 "equity": table("Certificates", [("6 Month", "3.50%")], "04/06/2026"),
                                 "rates": table("Certificates", [("6 Month", "3.50%")])}),
                 ("2026-05-19", {"special-rates": frozen,
                                 "equity": table("Certificates", [("6 Month", "3.75%")], "05/15/2026"),
                                 "rates": table("Certificates", [("6 Month", "3.75%")])})]
    answers = set()
    for order in itertools.permutations(["special-rates", "equity", "rates"]):
        ks = build(*[(day, {p: pages[p] for p in order}) for day, pages in downloads])
        answers.add(ks.answer("What was the APY on a 6-month certificate?", "2026-05-14").text)
    assert len(answers) == 1, answers
    assert "3.50%" in answers.pop()
