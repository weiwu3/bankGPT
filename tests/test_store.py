"""The history of the site: one view per download, pages that vanish and return, and what only the
history of a page can tell (a fuller title, a date printed later, a table left behind), and the words
learned from a site.

Made-up sites; no downloads, no model.
"""
import pytest

import solution
from helpers import CERTIFICATES, REFUSAL, build, path_of, snapshot, table, wren
from kb import text
from kb.engine import Engine
from kb.store import Store

QUESTION = "What is the APY on a 6-month Wren certificate?"


def test_nothing_ingested_is_no_view_and_an_honest_refusal():
    assert Store().view_at(None) is None and Store().view_at("2026-03-01") is None
    reply = Engine(reader=None).answer(QUESTION)
    assert (reply.text, reply.confident, reply.path, reply.pages) == (REFUSAL, False, "declined", [])


def test_a_page_that_did_not_change_is_one_version_seen_in_each_download():
    ks = build(("2026-01-05", {"rates": wren("2.10%")}), ("2026-02-05", {"rates": wren("2.10%")}),
               ("2026-03-05", {"rates": wren("2.60%")}))
    first, second = ks.engine.store.versions["cu.test/rates"]
    assert (first.first_seen, first.last_seen, second.first_seen, second.last_seen) == (0, 1, 2, 2)
    assert [v.pages["cu.test/rates"] for v in ks.engine.store.views] == [first, first, second]


def test_a_vanished_page_is_kept_as_gone_and_is_live_again_when_it_returns():
    fees = "# Fees\n\nLost card replacement fee: $6.75\n"
    ks = build(("2026-01-05", {"rates": wren("2.10%"), "fees": fees}), ("2026-02-05", {"rates": wren("2.10%")}),
               ("2026-03-05", {"rates": wren("2.10%"), "fees": fees}))
    views = ks.engine.store.views
    assert ["cu.test/fees" in v.gone for v in views] == [False, True, False]
    then = ks.answer("How much is a lost card replacement?", as_of="2026-02-10").text
    now = ks.answer("How much is a lost card replacement?").text
    assert "$6.75" in then and "$6.75" in now
    assert "no longer on the website" not in then          # an as-of answer says which download it is from instead
    assert "no longer on the website" not in now


def test_a_page_gone_from_the_newest_download_is_answered_and_says_so():
    ks = build(("2026-01-05", {"fees": "# Fees\n\nLost card replacement fee: $6.75\n", "rates": wren("2.10%")}),
               ("2026-03-05", {"rates": wren("2.10%")}))
    answer = ks.answer("How much is a lost card replacement?")
    assert "$6.75" in answer.text and answer.text.endswith(
        "(This page is no longer on the website; this is the last version that was published.)")


@pytest.mark.parametrize("live, winner", [("files/wren-rate-sheet.pdf", "2.10%"), ("wren-specials", "2.60%")])
def test_a_vanished_product_page_outranks_a_live_document_but_not_a_live_product_page(live, winner):
    ks = build(("2026-01-05", {"old-rates": wren("2.10%"), live: wren("2.60%")}), ("2026-03-05", {live: wren("2.60%")}))
    text = ks.answer(QUESTION).text
    [loser] = {"2.10%", "2.60%"} - {winner}
    assert winner in text and loser not in text


def test_an_empty_download_leaves_every_page_gone_and_the_answers_say_so():
    ks = build(("2026-01-05", {"rates": wren("2.10%")}), ("2026-03-05", {}))
    answer = ks.answer(QUESTION)
    assert "2.10%" in answer.text and "no longer on the website" in answer.text
    assert [s.snapshot_id for s in answer.sources] == ["02_2026-03-05"]


def test_each_download_is_its_own_view_whatever_came_after():
    downloads = [("2026-01-05", {"rates": wren("2.10%")}), ("2026-02-05", {"rates": wren("2.35%")}),
                 ("2026-03-05", {"rates": wren("2.60%")}), ("2026-04-05", {"other": "# Other\n\nNothing here.\n"}),
                 ("2026-05-05", {"rates": wren("2.85%")})]
    ks = build(*downloads)
    for as_of, want in [("2026-01-04", "2.10%"), ("2026-01-05", "2.10%"), ("2026-02-04", "2.10%"), ("2026-02-05", "2.35%"),
                        ("2026-03-31", "2.60%"), ("2026-04-05", "2.60%"), ("2026-05-04", "2.60%"), ("2026-05-05", "2.85%"),
                        ("2027-01-01", "2.85%")]:
        text = ks.answer(QUESTION, as_of).text
        assert want in text and [v for v in ("2.10%", "2.35%", "2.60%", "2.85%") if v in text] == [want], as_of


# --------------------------------------------------------------------------- what only the history tells

def test_an_early_short_title_takes_the_fuller_title_the_same_table_has_later():
    # Early downloads title the table "Classic"; later ones "Classic Wren Certificates".
    ks = build(("2026-01-05", {"rates": wren("2.10%", label="Classic")}),
               ("2026-03-05", {"rates": wren("2.10%", label="Classic Wren Certificates") + "\nUpdated page.\n"}))
    then = ks.answer("What was the APY on a 6-month Classic Wren certificate?", as_of="2026-01-10")
    assert "2.10%" in then.text and path_of(ks, "What was the APY on a 6-month Classic Wren certificate?", "2026-01-10") == "rows"


def test_an_undated_table_that_changed_is_dated_no_later_than_the_download_before_the_change():
    ks = build(("2026-01-05", {"rates": wren("2.10%")}), ("2026-03-05", {"rates": wren("2.60%")}))
    [old], [new] = (v.tables for v in ks.engine.store.versions["cu.test/rates"])
    assert (old.effective, old.dated_by) == (None, "page")
    assert (new.effective.isoformat(), new.dated_by, new.maintained) == ("2026-01-05", "change", True)
    assert "effective" not in ks.answer(QUESTION).text          # a lower bound is not printed as a date


def test_a_frozen_copy_s_printed_date_does_not_beat_a_page_that_changed_its_table():
    # The rates page changed its undated table between January 5 and March 5; the guide still prints
    # February 1 over the old value. The change date (January 5) is only a lower bound.
    ks = build(("2026-01-05", {"rates": wren("2.10%"), "wren-guide": wren("2.10%", "02/01/2026")}),
               ("2026-03-05", {"rates": wren("2.60%"), "wren-guide": wren("2.10%", "02/01/2026")}))
    text = ks.answer(QUESTION).text
    assert "2.60%" in text and "2.10%" not in text


def test_a_page_whose_dated_tables_lag_another_page_is_trusted_less_than_a_document():
    # The guide prints the Kestrel table under an older date with other values (two rows behind), and the
    # IRA table exactly as the rates page does. The copies that agree in full are no evidence that the
    # guide is kept up, so its two rows behind make it a page left behind: the rate sheet beats it.
    rates = (table("Kestrel Certificates", [("6 Month", "3.15%"), ("12 Month", "3.35%")], "02/23/2026")
             + "\n" + table("Kestrel IRA Certificates", [("6 Month", "3.05%"), ("12 Month", "3.25%"),
                                                         ("18 Month", "3.40%")], "03/01/2026"))
    guide = ("# Kestrel guide\n\n" + table("Kestrel Certificates", [("6 Month", "2.35%"), ("12 Month", "2.65%")], "01/05/2026")
             + "\n" + table("Kestrel IRA Certificates", [("6 Month", "3.05%"), ("12 Month", "3.25%"),
                                                         ("18 Month", "3.40%")], "03/01/2026")
             + "\n" + table("Kestrel Jumbo Certificates", [("24 Month", "2.85%")]))
    sheet = "Kestrel rate sheet\n\n" + table("Kestrel Jumbo Certificates", [("24 Month", "3.65%")])
    ks = build(("2026-03-09", {"rates": rates, "kestrel-guide": guide, "files/kestrel-rate-sheet.pdf": sheet}))
    assert ks.engine.store.latest.stale == {"cu.test/kestrel-guide"}
    answer = ks.answer("What is the APY on a 24-month Kestrel Jumbo certificate?").text
    assert "3.65%" in answer and "2.85%" not in answer


# --------------------------------------------------------------------------- the words learned from a site


def test_a_coined_name_made_of_ordinary_words_is_learned_from_the_site():
    before = list(text._LEARNED)
    try:
        text.learn_names([f"Page {n}. Members received a Give More dividend this year." for n in range(6)])
        assert "givemore" in text.tokens("how much give more did members get")
    finally:
        text._LEARNED[:] = before


def test_a_second_site_does_not_inherit_the_first_ones_vocabulary():
    # A site's own name says nothing about which product is meant, so it is learned and ignored.
    # The words are kept for the process: a site at certificates.example read first must not make
    # "certificate" a word that names nothing on the next site.
    names, learned = set(text.SITE_NAMES), list(text._LEARNED)
    try:
        other = solution.build()
        other.ingest(snapshot(1, "2026-03-05", {"rates": CERTIFICATES}, site="https://www.certificates.example"))
        ks = build(("2026-03-05", {"rates": CERTIFICATES}))
        assert "3.50%" in ks.answer("What is the APY on a 6-month certificate?").text
    finally:
        text.SITE_NAMES.clear()
        text.SITE_NAMES.update(names)
        text._LEARNED[:] = learned


def test_an_empty_first_download_does_not_keep_the_site_s_own_words_from_being_learned():
    names, learned = set(text.SITE_NAMES), list(text._LEARNED)
    try:
        ks = build()
        ks.ingest(snapshot(1, "2026-01-05", {}, site="https://www.wrenfield.test"))
        pages = {f"news/{n}": "# News\n\nMembers received a Give More dividend this year.\n" for n in range(5)}
        ks.ingest(snapshot(2, "2026-02-05", pages, site="https://www.wrenfield.test"))
        assert "wrenfield" in text.SITE_NAMES and "givemore" in text.tokens("the give more dividend")
    finally:
        text.SITE_NAMES.clear()
        text.SITE_NAMES.update(names)
        text._LEARNED[:] = learned
