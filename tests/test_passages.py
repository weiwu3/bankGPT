"""How pages are cut into passages and how passages are searched, checked on small made-up sites.

No website snapshots and no language model are needed: each test builds a few pages in memory
and feeds them to the solution exactly as the scorer does.
"""
import pytest

from helpers import build


def passages_of(ks, as_of=None):
    return ks.engine.store.view_at(as_of).cache["passages"]


# --------------------------------------------------------------------------- a heading that states a fact

GRANTS = """# Maker grants

Our makers fund helps local builders finish a project of their own.

#### Application period: March 3 - April 14, 2027

#### Status: Accepting proposals

## Applying

Fill in the online form and attach a sketch and a budget for your project.

### Who can apply

### Ask the desk

Write to the grants desk and a coordinator will reply within a week.

### Step 3: Send the form by April 14

#### Next steps

## Grants awarded in 2026

### Spring round

Nine builders shared the spring round, from a cedar canoe to a weather station.
"""


def test_a_heading_that_states_a_fact_is_a_passage_found_by_search():
    ks = build(("2026-09-01", {"grants": GRANTS}))
    index = passages_of(ks)
    found = [p for p in index.passages if p.text == "Application period: March 3 - April 14, 2027"]
    assert len(found) == 1 and found[0].heading == "Maker grants"      # labelled by the heading above it
    best = index.search("When is the application period for maker grants?")[0][1]
    assert best is found[0]
    assert "April 14, 2027" in ks.answer("When is the application period for maker grants?").text


def test_only_a_heading_that_states_something_and_is_nowhere_else_becomes_a_passage():
    index = passages_of(build(("2026-09-01", {"grants": GRANTS})))
    texts = {p.text for p in index.passages}
    assert "Status: Accepting proposals" in texts                    # a label and its value
    assert "Step 3: Send the form by April 14" in texts              # only an empty sub-heading under it
    assert "Who can apply" not in texts and "Next steps" not in texts    # bare titles state nothing
    assert "Grants awarded in 2026" not in texts                     # already the label of the section under it
    assert any(p.heading == "Grants awarded in 2026 - Spring round" for p in index.passages)


def test_the_passages_of_a_page_stay_in_page_order():
    index = passages_of(build(("2026-09-01", {"grants": GRANTS})))
    page = index.view.pages["cu.test/grants"]
    assert [p.order for p in page.passages] == list(range(len(page.passages)))
    assert [p.text[:12] for p in page.passages] == ["Our makers f", "Application ", "Status: Acce", "Fill in the ",
                                                    "Write to the", "Step 3: Send", "Nine builder"]


def test_a_teaser_headline_with_nothing_under_it_is_still_left_out():
    grants = ("1 min read\n\nMay 2, 2026\n\n# Foundation awards $37,500 to 12 nonprofits\n\n"
              "The foundation gave $37,500 in grants to 12 nonprofits in its spring round.\n")
    savers = ("1 min read\n\nJanuary 8, 2026\n\n# Savers club tops $2.1 million\n\n"
              "The winter savers club put away $2.1 million this season, its best year so far.\n")
    renovation = ("1 min read\n\nJune 9, 2026\n\n# Branch reopens after renovation\n\n"
                  "The renovated branch reopened this week after a year of work on the building.\n\n"
                  "## Recent news\n\n### Foundation awards $37,500 to 12 nonprofits\n\n"
                  "### Savers club tops $2.1 million\n")
    ks = build(("2026-06-10", {"newsroom/grants": grants, "newsroom/savers": savers, "newsroom/renovation": renovation}))
    index = passages_of(ks)
    page = index.view.pages["cu.test/newsroom/renovation"]
    teasers = [p for p in page.passages if "$" in p.text]
    assert len(teasers) == 2                                         # cut as passages of their own ...
    assert all(id(p) in index.teasers for p in teasers)              # ... and recognised as teasers
    assert not any(p.page is page and "$" in p.text for p in index.passages)


LIBRARY = """# Tool library

Members borrow sanders, sewing machines and pressure washers from the tool library for a week at a time.

## Recent news

### [Tool library lends its 4,000th sander](https://cu.test/newsroom/tool-library-sanders)

### **[Repair cafe mends 85 lamps in one evening](https://cu.test/newsroom/repair-cafe)**

### [![](https://cu.test/img/1.png) Library wins 3 county awards](https://cu.test/newsroom/county-awards)

## Ask the desk

### Call 555-0142, ext. 31, or [write to the library desk](mailto:library@cu.test)

### [Call 555-0142, ext. 32 for the workshop bench](tel:5550142)
"""


def test_a_headline_that_is_only_a_link_is_left_out_even_when_its_article_is_not_in_the_download():
    index = passages_of(build(("2026-06-10", {"tools": LIBRARY})))
    texts = {p.text for p in index.passages}
    assert not index.teasers                          # no article in the download: the teaser rule cannot see them
    assert "Tool library lends its 4,000th sander" not in texts
    assert "Repair cafe mends 85 lamps in one evening" not in texts          # a link in bold is still only a link
    assert "Library wins 3 county awards" not in texts                       # and so is one with a picture in it
    assert "Call 555-0142, ext. 31, or write to the library desk" in texts   # a fact that has a link in it
    assert "Call 555-0142, ext. 32 for the workshop bench" in texts          # a number to call is not another page


MARKET = """# Night market

Every summer Friday the night market fills the square by the river with stalls and music.

#### Stall applications: Closed

#### Pitch fee: Waived for members

#### Evening 2: Which stalls are new?

#### Stall or table: Does it matter?

#### Choosing the pitches: How we decide

#### Selling at night: A season of summer evenings by the river

## Volunteers

Volunteers set out the stalls at dusk and stack them away after the last song.
"""


def test_a_label_with_a_short_answer_states_something_and_a_title_with_a_colon_does_not():
    texts = {p.text for p in passages_of(build(("2026-09-01", {"market": MARKET}))).passages}
    assert "Stall applications: Closed" in texts                     # a label and its value
    assert "Pitch fee: Waived for members" in texts
    assert "Evening 2: Which stalls are new?" in texts               # a figure is enough, as before
    assert "Stall or table: Does it matter?" not in texts            # a question
    assert "Choosing the pitches: How we decide" not in texts        # a question without its mark
    assert "Selling at night: A season of summer evenings by the river" not in texts     # the title goes on


def test_a_teaser_of_another_article_is_not_evidence():
    old = "1 min read\n\nMay 2, 2024\n\n# Foundation awards grants to nonprofits\n\nThe foundation gave grants to 17 local nonprofits this year, the first year of the program.\n"
    new = ("1 min read\n\nJune 9, 2026\n\n# Branch renovation is complete\n\nThe renovated branch reopened this week after a year of work on the building.\n\n"
           "## Recent news\n\n### Foundation awards grants to nonprofits\n\nThe foundation gave grants to 17 local nonprofits this year, the first... Keep reading\n")
    ks = build(("2026-06-10", {"newsroom/grants-2024": old, "newsroom/renovation": new}))
    index = ks.engine.store.latest.cache["passages"]
    holders = {p.page.key.split("/")[-1] for p in index.passages if "17 local nonprofits" in p.text}
    assert holders == {"grants-2024"}                  # only the article itself, under its own date


def test_the_same_words_with_other_numbers_on_two_current_pages_are_two_facts():
    # One statement repeated with newer numbers keeps only its newest copy; two desks' extensions,
    # printed in the same words on two pages trusted and dated alike, are both kept.
    desk = ("Questions about this service? Our team answers calls on weekdays from eight in the morning to five in the "
            "evening and on Saturday mornings; call 555-0100, ext. {}, and ask for the desk that handles it, or leave a "
            "message with your name and number and we will call you back the same business day.")
    ks = build(("2026-03-05", {"auto-loans": "# Auto loans\n\nFinance a car.\n\n## Talk to us\n\n" + desk.format("522") + "\n",
                               "business/payroll": "# Payroll\n\nPay your staff on time.\n\n## Talk to us\n\n"
                                                   + desk.format("411") + "\n"}))
    index = ks.engine.store.latest.cache["passages"]
    assert sorted(p.page.key for p in index.passages if "ext." in p.text) == ["cu.test/auto-loans", "cu.test/business/payroll"]
    assert "ext. 411" in ks.answer("What extension do I call about business payroll?").text


def test_fine_print_travels_with_the_sentence_that_points_to_it():
    page = ("# Youth savings\n\nOpen a youth savings account for your child and we add $25 to it1.\n\n## More\n\nOther text here for the page.\n\n"
            "_1 One bonus per child, paid into the account within thirty (30) days._\n")
    ks = build(("2026-03-05", {"youth": page}))
    index = ks.engine.store.latest.cache["passages"]
    passage = next(p for p in index.passages if "we add $25 to it" in p.text)
    assert any("within thirty (30) days" in note for note in index.footnotes(passage))


# --------------------------------------------------------------------------- "last year" means a year


def volunteering(year, hours, published):
    return (f"1 min read\n\n{published}\n\n# Our staff keeps giving back\n\n"
            f"In {year}, our staff gave {hours} volunteer hours to local causes.\n")


# Each article gives the same figure for another year. By their words alone they tie, and the newest wins.
YEAR_1 = ("2026-12-01", {"newsroom/staff-giving": volunteering(2025, "5,200", "January 20, 2026"),
                         "newsroom/staff-giving-update": volunteering(2026, "3,900", "November 15, 2026")})
YEAR_2 = ("2027-12-01", {**YEAR_1[1], "newsroom/staff-giving-recap": volunteering(2027, "2,800", "November 20, 2027")})
QUESTION = "How many volunteer hours did our staff give last year?"
FIGURES = ("5,200", "3,900", "2,800")


def test_a_relative_year_is_searched_as_the_year_it_means_in_the_download_in_force():
    ks = build(YEAR_1, YEAR_2)
    then, now = passages_of(ks, "2026-12-15"), passages_of(ks)
    assert "2025" in then.query_tokens(QUESTION) and "2026" not in then.query_tokens(QUESTION)
    assert "2026" in now.query_tokens(QUESTION) and "2025" not in now.query_tokens(QUESTION)
    assert {"last", "year"} <= set(now.query_tokens(QUESTION))      # words are added, none taken away
    assert "2027" in now.query_tokens("How many volunteer hours this year?")
    assert "2028" in now.query_tokens("What will the program look like next year?")
    assert "2026" in now.query_tokens("How many volunteer hours last  year?")    # however the words are spaced
    assert "2026" in now.query_tokens("How many volunteer hours last\nyear?")
    for said in ("in the past year", "in the previous year", "a year ago", "in the prior yr"):      # as the reader reads them
        assert "2026" in now.query_tokens(f"How many volunteer hours {said}?"), said
    assert "5,200" in then.search(QUESTION)[0][1].text
    assert "3,900" in now.search(QUESTION)[0][1].text


@pytest.mark.parametrize("as_of, figure", [("2026-12-15", "5,200"), (None, "3,900")])
def test_a_last_year_answer_changes_with_the_download_in_force(as_of, figure):
    answer = build(YEAR_1, YEAR_2).answer(QUESTION, as_of).text
    assert [f for f in FIGURES if f in answer] == [figure]



@pytest.mark.parametrize("text, found", [("Lobby: 9 a.m. to 5:30 p.m.", ["9 a.m", "5:30 p.m"]), ("Open until noon", ["noon"]),
                                         ("Written by Jane Roe | Mar 3, 2024 10:41:07 AM", [])])
def test_a_clock_time_is_an_opening_time_not_an_article_s_time_to_the_second(text, found):
    from kb.passages import CLOCK_TIME
    assert [m.group(0) for m in CLOCK_TIME.finditer(text)] == found


# --------------------------------------------------------------------------- the member's words for the site's

WORDS_SITE = ("2026-06-10", {
    "locations": "# Locations\n\n## Fargo\n\nBirch ATM, 1 Main St, Fargo, ND, 58102\n\n## Birch\n\n"
                 "North Birch ATM, 4 Elm Rd, Birch, SD, 57701\n\n## Pine\n\nNorth Pine ATM, 9 Oak Rd, Lead, SD, 57754\n",
    "press": "# Newsroom\n\n## Media\n\nMedia contact: Jo Smith, ext. 12.\n\n## Questions\n\n"
             "Call us with questions about your account: 555-010-2000.\n",
    "about": "# About us\n\n## Our team\n\nOur 250 employees serve the valley.\n\n## Visiting\n\n"
             "People love to visit our lobby, and many work nearby.\n",
    "join": "# Join\n\n## Who\n\nAnyone who lives in Birch County is eligible to join.\n\n## How\n\n"
            "How can you open an account? Bring an ID to qualify for the bonus.\n",
    "fees": "# Fee schedule\n\n| Service | Fee |\n| --- | --- |\n| Official check | $7.50 |\n| Money order | $2.00 |\n"
            "| Domestic wire | $20.00 |\n| International wire | $40.00 |\n",
    "rates": "# Rates\n\nHome Equity Line of Credit\n\n| Plan | APR |\n| --- | --- |\n| Variable | 7.25% |\n\n"
             "Home Loans\n\n| Term | APR |\n| --- | --- |\n| 30 Year | 6.10% |\n\n"
             "Money Market\n\n| Balance | APY |\n| --- | --- |\n| $0 - $9,999 | 0.40% |\n| $10,000+ | 1.10% |\n",
})


@pytest.mark.parametrize("question, found", [
    ("Is there an ATM in North Dakota?", "Fargo, ND"),
    ("Who do I call if I'm a journalist?", "Media contact"),
    ("How many people work there?", "250 employees"),
    ("How can I qualify to join?", "eligible to join"),
])
def test_the_member_s_word_finds_the_site_s_word(question, found):
    assert found in passages_of(build(WORDS_SITE)).search(question)[0][1].text


@pytest.mark.parametrize("question, want", [
    ("How much is a cashier's check?", "Fee schedule, Official check: $7.50."),
    ("What is the home equity line rate?", "Home Equity Line of Credit, Variable: 7.25% APR."),
    ("What does it cost to wire money overseas?", "Fee schedule, International wire: $40.00."),
    ("I have 15k for an MM account, what rate would I get?", "Money Market, $10,000+: 1.10% APY."),
])
def test_the_member_s_word_finds_the_site_s_row(question, want):
    assert build(WORDS_SITE).answer(question).text == want


@pytest.mark.parametrize("question", [
    "Can someone join my account as a joint owner?", "Who can join my checking account?", "Can I join the call?",
    "Do I qualify for membership rewards?", "Will the bank check my credit for a car loan?",
    "Does the bank check my income?", "Is the press of business slowing loans?", "Who may join the credit union board?",
])
def test_a_word_used_in_another_sense_is_not_folded(question):
    from kb.text import canon
    assert not any(w in canon(question) for w in ("eligible to join", "official check", "media"))


def test_a_state_the_site_names_everywhere_brings_no_postal_code():
    # Where many pages name the state, its code would be on every address too and only dilute the
    # question's other words: "in North Dakota" is searched as the words it is.
    assert "nd" in passages_of(build(WORDS_SITE)).query_tokens("Is there an ATM in North Dakota?")
    pages = {f"news/{n}": f"# News {n}\n\nA North Dakota story, number {n}.\n" for n in range(120)}
    index = passages_of(build(("2026-06-10", dict(WORDS_SITE[1], **pages))))
    assert "nd" not in index.query_tokens("Is there an ATM in North Dakota?")
