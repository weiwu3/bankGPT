"""What kind of page a page is, read from its URL and its first lines, and how it is named.

The kind decides trust when pages disagree (see test_current.py for the rules themselves). Table-driven
over made-up URLs. No downloads, no model.
"""
from datetime import date

import pytest

from helpers import build, table
from kb import pages as P


@pytest.mark.parametrize("url, top, kind", [
    ("https://cu.test/rates", "", P.OFFICIAL),
    ("https://cu.test/checking/finch-checking", "# Finch Checking", P.OFFICIAL),
    ("https://cu.test/blog", "", P.OFFICIAL),                                   # a blog's front page is no article
    ("https://cu.test/files/fees.pdf", "", P.DOCUMENT),
    ("https://cu.test/docs/Fees.PDF", "", P.DOCUMENT),
    ("https://cu.test/hubfs/rate-sheet", "", P.DOCUMENT),
    ("https://cu.test/data/rates.csv", "", P.DOCUMENT),
    ("https://cu.test/forms/application.docx", "", P.DOCUMENT),
    ("https://cu.test/files/calculator-guide.pdf", "", P.DOCUMENT),           # a file before a calculator
    ("https://cu.test/tools/loan-calculator", "", P.CALCULATOR),
    ("https://cu.test/Calculators/Savings", "", P.CALCULATOR),
    ("https://cu.test/tag/savings", "", P.INDEX),
    ("https://cu.test/author/jo/", "", P.INDEX),
    ("https://cu.test/topics/loans/", "", P.INDEX),
    ("https://cu.test/blog?category=news", "", P.INDEX),
    ("https://cu.test/blog/new-branch", "", P.ARTICLE),
    ("https://cu.test/newsroom/new-branch", "", P.ARTICLE),
    ("https://cu.test/press-releases/new-branch", "", P.ARTICLE),
    ("https://cu.test/resources/saving-guide", "", P.ARTICLE),
    ("https://cu.test/stories/a-member", "", P.ARTICLE),
    ("https://cu.test/2025/4/a-member", "", P.ARTICLE),                        # a dated path
    ("https://cu.test/about/our-story", "1 min read\n\n# Our story", P.ARTICLE),   # a byline at the top
    ("https://cu.test/summer-lp", "", P.LANDING),
    ("https://cu.test/auto-lp/", "", P.LANDING),
    ("https://cu.test/promo/certificates", "", P.LANDING),
    ("https://cu.test/special-offer", "", P.LANDING),
    ("https://cu.test/landing/spring", "", P.LANDING),
    ("https://cu.test/giveaway", "", P.LANDING),
    ("https://cu.test/promotional-rates", "", P.LANDING),
])
def test_the_kind_of_page_is_read_from_its_url_and_top_lines(url, top, kind):
    assert P.classify(url, top) == kind


def test_a_reading_time_far_down_the_page_does_not_make_it_an_article():
    assert P.classify("https://cu.test/about", "x" * 500 + " 3 min read") == P.OFFICIAL


@pytest.mark.parametrize("url, key", [
    ("https://www.cu.test/rates/", "cu.test/rates"), ("http://cu.test/rates", "cu.test/rates"),
    ("cu.test/rates", "cu.test/rates"), ("https://cu.test/rates#fine-print", "cu.test/rates"),
    ("https://CU.test/rates", "cu.test/rates"), ("https://cu.test/rates?term=12", "cu.test/rates?term=12"),
])
def test_the_same_page_has_one_key_however_its_url_is_written(url, key):
    assert P.url_key(url) == key


@pytest.mark.parametrize("title, markdown, name", [
    ("Auto Loans | Finch Credit Union", "# Something else", "Auto Loans"),     # the page's own title wins
    ("Auto Loans - Finch Credit Union", "", "Auto Loans"),
    ("Auto Loans – Finch", "", "Auto Loans"),                              # an en dash, cleaned first
    ("", "Intro line\n\n# Finch Auto Loans\n\n## Rates", "Finch Auto Loans"),  # no title: the first top heading
    ("", "No heading at all", ""),
])
def test_a_page_is_named_by_its_title_without_the_site_suffix(title, markdown, name):
    assert P.page_title(title, markdown) == name


@pytest.mark.parametrize("url, markdown, day", [
    ("https://cu.test/news/x", "1 min read\n\nMarch 3, 2025\n\n# Opening", date(2025, 3, 3)),
    ("https://cu.test/news/2025/04/09/x", "# Opening", date(2025, 4, 9)),        # the URL when the text has none
    ("https://cu.test/news/2025/02/30/x", "# Opening", None),                    # no such day
    ("https://cu.test/news/x", "x" * 800 + "March 3, 2025", None),               # too far down to be the date line
])
def test_an_article_is_dated_by_its_first_date_near_the_top(url, markdown, day):
    assert P.published_on(url, markdown) == day


@pytest.mark.parametrize("kind, published, label", [
    (P.OFFICIAL, None, "product page"), (P.DOCUMENT, None, "published document"), (P.LANDING, None, "campaign page"),
    (P.ARTICLE, None, "article"), (P.ARTICLE, date(2026, 3, 1), "article published 2026-03-01"),
    (P.CALCULATOR, None, "calculator page"), (P.INDEX, None, "index page"),
])
def test_each_kind_of_page_has_a_name_for_the_reasons_given(kind, published, label):
    page = P.PageVersion(url="u", key="k", title="t", markdown="", fetched="", first_seen=0, last_seen=0, kind=kind,
                         published=published)
    assert page.label() == label


def test_a_dated_article_is_named_with_its_date_where_its_value_lost():
    story = "1 min read\n\nMarch 1, 2026\n\n# Wren news\n\n" + table("Wren Certificates", [("6 Month", "2.10%")])
    ks = build(("2026-03-05", {"rates": table("Wren Certificates", [("6 Month", "2.60%")]), "news/wren-story": story}))
    answer = ks.answer("What is the APY on a 6-month Wren certificate?")
    assert "2.60%" in answer.text and "(article published 2026-03-01)" in answer.conflict_note


def test_the_reason_uses_an_before_article():
    ks = build(("2026-03-05", {"rates": table("Wren Certificates", [("6 Month", "2.60%")]),
                               "news/wren-story": table("Wren Certificates", [("6 Month", "2.10%")])}))
    assert ks.answer("What is the APY on a 6-month Wren certificate?").conflict_note.endswith(
        "Chosen because a product page outranks an article.")


def test_an_undated_article_is_read_without_error():
    ks = build(("2026-03-05", {"news/wren-story": "# Wren story\n\nMembers opened Wren certificates at the fair.\n",
                               "rates": table("Wren Certificates", [("6 Month", "2.60%")])}))
    assert ks.engine.store.latest.pages["cu.test/news/wren-story"].published is None
    assert "2.60%" in ks.answer("What is the APY on a 6-month Wren certificate?").text
    assert "at the fair" in ks.answer("Where did members open Wren certificates?").text      # searched, so ranked
