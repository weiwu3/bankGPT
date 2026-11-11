"""How the date a question is asked as of is read, checked on a small made-up site.

The contract asks for an ISO date. A date written another way must still pick the download in
force on that day: a date that is not read is answered from the newest download, which for a past
date quotes a later value than the one asked for. No website snapshots and no language model are
needed.
"""
from datetime import date, datetime

import pytest

from helpers import build

QUESTION = "What is the APY on a 6-month certificate?"

# Three downloads, each with its own 6-month APY. The last two are a day apart, on the last day of
# 2025 and the first of 2026, so a date read even one day off picks the wrong download.
APY = {"2025-06-02": "2.95%", "2025-12-31": "3.20%", "2026-01-01": "3.50%"}
FIRST, YEAR_END, NEWEST = APY


@pytest.fixture(scope="module")
def ks():
    page = "## Rates\n\nCertificates\n\n| Term | APY |\n| --- | --- |\n| 6 Month | {} |\n"
    return build(*[(day, {"rates": page.format(apy)}) for day, apy in APY.items()])


def answered_from(ks, as_of):
    """The downloads whose APY the answer quotes: exactly one when the right view was used."""
    text = ks.answer(QUESTION, as_of).text
    return [day for day, apy in APY.items() if apy in text]


@pytest.mark.parametrize("as_of, download", [
    ("2025-12-31", YEAR_END),                    # the day of a download counts
    ("2025-12-30", FIRST),
    ("2026-01-01", NEWEST),
    ("2025-12-31T23:59:59Z", YEAR_END),          # a time or a time zone after the day is ignored
    ("2025-12-31T23:59:59-05:00", YEAR_END),
    ("2025-12-31 08:00:00", YEAR_END),
    ("2024-03-15", FIRST),                       # before the first download: the first
    ("2027-03-15", NEWEST),                      # after the last: the last
])
def test_a_full_iso_date_picks_the_download_in_force_that_day(ks, as_of, download):
    assert answered_from(ks, as_of) == [download]


@pytest.mark.parametrize("as_of, download", [
    (date(2025, 12, 31), YEAR_END),
    (datetime(2025, 12, 30, 23, 59), FIRST),
    ("2025-12", YEAR_END),                       # a month alone is its last day
    ("2025-11", FIRST),
    ("2025", YEAR_END),                          # a year alone is December 31
    ("2024", FIRST),
    ("2025/12/31", YEAR_END),
    ("2025.12.30", FIRST),
    ("12/31/2025", YEAR_END),                    # month first, as US sites print dates
    ("12/30/2025", FIRST),
    ("12/2025", YEAR_END),
    ("11/2025", FIRST),
    ("December 2025", YEAR_END),
    ("dec 2025", YEAR_END),
    ("NOV 2025", FIRST),
    ("Sept. 2025", FIRST),
    ("December 31, 2025", YEAR_END),
    ("Dec 30 2025", FIRST),
    ("31 December 2025", YEAR_END),
    ("30 dec 2025", FIRST),
    ("  as of December 2025 ", YEAR_END),
    ("As of 2025-12-31", YEAR_END),
    ("AS OF 12/30/2025", FIRST),
])
def test_a_date_written_another_way_picks_the_download_in_force_that_day(ks, as_of, download):
    assert answered_from(ks, as_of) == [download]


UNREADABLE = [
    None, "",                                    # no date at all
    "garbage", "Smarch 2025", "end of 2025",
    "2025-02-30", "February 30, 2025", "2025-13", "13/2025",      # impossible dates
    "20251231", 20251231, 2025, 2025.12,         # a number is not a date
]


@pytest.mark.parametrize("as_of", UNREADABLE + [["2025-12-31"], {"day": "2025-12-31"}])    # unhashable too
def test_a_date_that_cannot_be_read_answers_from_the_newest_download(ks, as_of):
    assert answered_from(ks, as_of) == [NEWEST]


@pytest.mark.parametrize("as_of", UNREADABLE + [[], ["2025-12-31"], {"day": "2025-12-31"}, b"2025-12-31", object()])
def test_choosing_the_download_never_raises(ks, as_of):
    assert ks.engine.store.view_at(as_of).taken_on.isoformat() == NEWEST
