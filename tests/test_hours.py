"""The hours rule on small made-up sites: which branch a question names, and how the answer names it.

No website snapshots and no language model are needed.
"""
import re

import pytest

import helpers


def first_heading(markdown):
    return re.search(r"^# (.+)$", markdown, re.M).group(1)


def build(pages, day="2026-03-05"):
    """pages: {path: markdown}, one download. A page's title is its first heading."""
    return helpers.build((day, pages), title=first_heading)


def branch(town, state, where, friday, below=""):
    """A branch announcement: a dateline with the state (none when state is None), a sentence saying
    where it is, its schedule, and what the page prints below the schedule."""
    dateline = f"{town.upper()}, {state}-" if state else ""
    return (f"March 3, 2025\n\n# Credit union opens {town} branch\n\n{dateline}Our {town} office "
            f"opens today. {where}\n\n**Lobby**\nMonday-Thursday: 9:15 a.m.-4:45 p.m.\nFriday: 9:15 a.m.-{friday}\n\n"
            "**Drive-up**\nMonday-Friday: 7:45 a.m.-6:15 p.m.\n" + below)


def two_branches(state):
    return build({"news/elmwood": branch("Elmwood", state, "It is located at 12 Elm Street.", "7:15 p.m."),
                  "news/oakdale": branch("Oakdale", state, "It is located at 40 Oak Street.", "3:45 p.m.")})


@pytest.mark.parametrize("on_page, asked", [
    ("Ill.", "IL"), ("Ill.", "il"), ("Ill.", "Illinois"), ("IL", "Ill."), ("Illinois", "IL"),
    ("Kan.", "Kansas"), ("Kansas", "KS"), ("Kans.", "Kan."),        # tokenised as "kan" and "kansa"
    ("Ore.", "OR"), ("OR", "Oregon"),                                # "or" is a stop word: no state reaches the rule
    ("N.Y.", "New York"), ("New York", "NY"), ("W.Va.", "West Virginia"),
])
def test_a_state_written_another_way_still_reaches_the_branch(on_page, asked):
    reply = two_branches(on_page).engine.answer(f"What time does the lobby close on Fridays at the Elmwood, {asked} branch?")
    assert reply.path == "hours"
    assert "7:15 p.m." in reply.text and "3:45" not in reply.text


@pytest.mark.parametrize("asked", ["Pinehurst", "Pinehurst, IL", "Pinehurst, Ill.", "Pinehurst, Illinois"])
def test_a_town_with_no_schedule_gets_none_from_this_rule(asked):
    reply = two_branches("Ill.").engine.answer(f"What are the lobby hours at the {asked} branch?")
    assert reply.path != "hours" and not reply.confident


def test_a_page_that_names_another_state_is_not_the_branch_asked_about():
    # A state's name is also a street's: "Vermont Avenue" must not reach every page with an "avenue".
    ks = build({"news/elmwood": branch("Elmwood", "Ore.", "It is located at 40 Vermont Avenue.", "7:15 p.m."),
                "news/oakdale": branch("Oakdale", "Ore.", "It is located at 12 Elm Avenue.", "3:45 p.m.")})
    text = ks.answer("What time does the lobby close on Fridays at the Vermont Avenue branch?").text
    assert "7:15 p.m." in text and "3:45" not in text
    assert ks.engine.answer("What are the lobby hours at the Kansas Avenue branch?").path != "hours"
    # Two towns of one name: the state picks the branch when the pages name their states.
    ks = build({"news/elmwood-or": branch("Elmwood", "Ore.", "It is located at 12 Elm Street.", "7:15 p.m."),
                "news/elmwood-ks": branch("Elmwood", "Kan.", "It is located at 40 Oak Street.", "3:45 p.m.")})
    text = ks.answer("What time does the lobby close on Fridays at the Elmwood, KS branch?").text
    assert "3:45 p.m." in text and "7:15 p.m." not in text


@pytest.mark.parametrize("asked", ["Oregon", "Ore."])
def test_a_question_that_names_only_a_state_gets_the_branches_in_it(asked):
    ks = build({"news/elmwood": branch("Elmwood", "Ore.", "It is located at 12 Elm Street.", "7:15 p.m."),
                "news/oakdale": branch("Oakdale", "Kan.", "It is located at 40 Oak Street.", "3:45 p.m.")})
    text = ks.answer(f"What are the lobby hours of your {asked} branches?").text
    assert "7:15 p.m." in text and "3:45" not in text


@pytest.mark.parametrize("state, street, other", [
    (None, "Ohio Street", "Elm Street"), (None, "Ohio St.", "Elm St."), (None, "New York Avenue", "Elm Avenue"),
    ("Ore.", "Oregon Ave.", "Elm Ave."),       # the pages' own state, written another way
    ("Ore.", "Oregon Ct.", "Elm Ct."),         # "Ct." is the court, not Connecticut
])
def test_a_street_named_after_a_state_is_a_place_not_a_state(state, street, other):
    # Read as a state, "Ohio" would drop out of the question and leave "street", which every page has.
    ks = build({"news/elmwood": branch("Elmwood", state, f"It is located at 40 {street} in Elmwood.", "7:15 p.m."),
                "news/oakdale": branch("Oakdale", state, f"It is located at 12 {other} in Oakdale.", "3:45 p.m.")})
    reply = ks.engine.answer(f"What time does the lobby close on Fridays at the {street} branch?")
    assert reply.path == "hours"
    assert "7:15 p.m." in reply.text and "3:45" not in reply.text


@pytest.mark.parametrize("state", [None, "Kan."])
@pytest.mark.parametrize("asked", ["the Elmwood, Kansas, Elm Street branch", "the Elm Street branch in Elmwood, Kansas"])
def test_a_state_beside_a_street_is_still_only_a_state(state, asked):
    # "Kansas" is not followed by a street word here: it stays the state, which the page need not
    # print, or may print another way.
    ks = build({"news/elmwood": branch("Elmwood", state, "It is located at 12 Elm Street.", "7:15 p.m."),
                "news/oakdale": branch("Oakdale", state, "It is located at 40 Oak Street.", "3:45 p.m.")})
    reply = ks.engine.answer(f"What time does the lobby close on Fridays at {asked}?")
    assert reply.path == "hours"
    assert "7:15 p.m." in reply.text and "3:45" not in reply.text


@pytest.mark.parametrize("where, label", [
    ("It is located at 1200 E. Maple Road at the corner of Maple Road and Fifth Street.", "1200 E. Maple Road"),
    ("It is located at 45 N. Birch St. in Elmwood.", "45 N. Birch St."),
    ("It is located at 9 Lake Blvd. The lobby has two teller windows.", "9 Lake Blvd."),
    ("It is located at 300 Pine Road, with a coin counter.", "300 Pine Road"),
    ("It is located at 5 Hill Street at the intersection of Hill Street and Oak Avenue.", "5 Hill Street"),
    ("It is located at 77 Cedar Lane. Parking is behind the building.", "77 Cedar Lane"),
    ("Calls are answered by staff located at our main office. The office is located at 610 W. Birch Ave. in the mall.",
     "610 W. Birch Ave."),                                                           # the first "located at" is no address
    ("It is located at 610 W. Birch Ave., and it has a night drop.", "610 W. Birch Ave."),
    ("It is located at 100 Mt. Vernon Road in Elmwood.", "100 Mt. Vernon Road"),
    ("It is located at 2 County Hwy. 30, next to the library.", "2 County Hwy. 30"),
    ("It is located at 500 Oak Ave. SE in Elmwood.", "500 Oak Ave. SE"),
    ("It is located at 300 Birch-Hollow Road in Elmwood.", "300 Birch-Hollow Road"),
    ("It is located at 12 Elm Road\u2014across from the library.", "12 Elm Road"),       # a dash glued to the next word
    ("It is located at the corner of Pine Road and Hill Street.", "Credit union opens Elmwood branch"),    # no address
    ("See also: our Oakdale office, located at 9 Lake...", "Credit union opens Elmwood branch"),      # a cut-off teaser
    ("See also: our Oakdale office, located at 9 N. Lake Road...", "Credit union opens Elmwood branch"),   # cut after an initial
])
def test_the_branch_is_named_by_its_whole_street_address(where, label):
    ks = build({"news/elmwood": branch("Elmwood", "Ore.", where, "7:15 p.m.")})
    text = ks.answer("What are the lobby hours at the Elmwood branch?").text
    named = label if label.startswith("Credit") else f"the branch at {label}"
    assert text.startswith(f"For {named}, as announced 2025-03-03: lobby hours: ")


@pytest.mark.parametrize("where, label", [
    ("It is located at 12 Elm Street.", "the branch at 12 Elm Street"),
    ("It is located at the corner of Pine Road and Hill Street.", "Credit union opens Elmwood branch"),  # no address
])
def test_an_address_below_the_hours_is_another_branch(where, label):
    # Below its hours a page points to another branch. Its address is not this schedule's, even
    # when the page's own address cannot be read: then the page is named by its title.
    below = "\nSee also: our Oakdale office, located at 40 Oak Street, opens next month.\n"
    ks = build({"news/elmwood": branch("Elmwood", "Ore.", where, "7:15 p.m.", below=below)})
    text = ks.answer("What are the lobby hours at the Elmwood branch?").text
    assert text.startswith(f"For {label}, as announced 2025-03-03: lobby hours: ")


@pytest.mark.parametrize("asked, want", [
    ("Are you open tomorrow?", "Lobby hours vary by branch"),
    ("What time do you open today?", "Lobby hours vary by branch"),
    ("How late is the drive-up open tonight?", "Drive-up hours: Monday-Friday: 7:45 a.m.-6:15 p.m."),
])
def test_a_word_for_when_names_no_branch(asked, want):
    # "Today", "tomorrow" and "tonight" say when, not where: the question still names no branch.
    reply = two_branches("Ore.").engine.answer(asked)
    assert reply.path == "hours" and want in reply.text


def test_a_word_for_when_beside_a_branch_still_gets_that_branch():
    reply = two_branches("Ore.").engine.answer("What time does the Elmwood lobby close tonight?")
    assert reply.path == "hours"
    assert "7:15 p.m." in reply.text and "3:45" not in reply.text


def test_a_highway_number_tells_branches_apart():
    # "Highway 52" is not the branch by Highway 88: after "highway" a number is the road's name. With
    # "tomorrow" and "morning" read as when, "highway" alone was left and reached the Highway 88 branch.
    ks = build({"news/elmwood": branch("Elmwood", "Ore.", "It is located at 12 Elm Street, by Highway 88.", "7:15 p.m."),
                "news/oakdale": branch("Oakdale", "Ore.", "It is located at 40 Oak Street.", "3:45 p.m.")})
    assert ks.engine.answer("Is the Highway 52 branch open tomorrow morning?").path != "hours"
    reply = ks.engine.answer("Is the Highway 88 branch open tomorrow morning?")
    assert reply.path == "hours" and "7:15 p.m." in reply.text and "3:45" not in reply.text


# --------------------------------------------------------------------------- without a model, named places

LOCATIONS = ("# Find a location\n\nStop by and see us at any of our branches.\n\n"
             "### Credit union branch\n\n4 Birch Way, Birchwood, KS, 66001\n\n"
             "### Credit union branch\n\n77 Linden Hill Dr., Elmwood, KS, 66002\n")


def with_locations():
    """Two announcements with schedules, a list of locations that prints two more branches without hours, and a
    contact page whose hours are when messages are answered."""
    return build({"news/elmwood": branch("Elmwood", "Kan.", "It is located at 12 Elm Street.", "7:15 p.m."),
                  "news/oakdale": branch("Oakdale", "Kan.", "It is located at 40 Oak Street.", "3:45 p.m."),
                  "locations": LOCATIONS,
                  "contact": "# Contact us\n\nWe answer messages Monday-Friday, 8 a.m.-4 p.m. Call 555-010-0199.\n"})


@pytest.mark.parametrize("question", ["What are the Birchwood branch hours?", "When does the Birchwood branch open on Saturday?",
                                      "birchwood branch hours", "Birchwood lobby hours?", "is the birchwood branch open today?",
                                      "What time does the Linden Hill Drive branch close?",
                                      "is the linden hill drive branch open this afternoon?",
                                      "How late is the Birchwood branch open today?", "Birchwood open on saturday?",
                                      "birchwood hours on sunday?"])
def test_a_branch_with_no_published_hours_is_declined_not_given_another_branchs(question):
    # The site names the branch, in a list of locations, but prints hours only for Elmwood and Oakdale.
    reply = with_locations().engine.answer(question)
    assert not reply.confident and not re.search(r"\d [ap]\.m\.", reply.text), reply.text


@pytest.mark.parametrize("question", ["Where is the Linden Hill Drive branch?", "where is the linden hill drive branch located?"])
def test_a_street_written_out_finds_the_abbreviation_on_the_site(question):
    # The site writes "Dr."; the question "Drive".
    reply = with_locations().engine.answer(question)
    assert reply.confident and "77 Linden Hill Dr." in reply.text, reply.text


def test_a_street_written_out_reaches_the_branch_whose_page_abbreviates_it():
    ks = build({"news/elmwood": branch("Elmwood", "Kan.", "It is located at 9 Cedar Hill Blvd.", "7:15 p.m."),
                "news/oakdale": branch("Oakdale", "Kan.", "It is located at 40 Oak Street.", "3:45 p.m.")})
    reply = ks.engine.answer("What time does the Cedar Hill Boulevard branch close on Friday?")
    assert reply.path == "hours" and "7:15 p.m." in reply.text and "3:45" not in reply.text, reply.text


@pytest.mark.parametrize("question", ["When does the Birchwood branch close?", "Is the Birchwood branch still open?"])
def test_a_question_about_closing_for_good_is_not_read_as_one_about_hours(question):
    closing = ("March 3, 2025\n\n# Credit union to close Birchwood branch\n\nBIRCHWOOD, Kan.-The credit union will "
               "close its Birchwood branch. This location will close March 31, 2026, and members can use the Elmwood "
               "branch.\n")
    ks = build({"news/elmwood": branch("Elmwood", "Kan.", "It is located at 12 Elm Street.", "7:15 p.m."),
                "news/birchwood-closing": closing})
    reply = ks.engine.answer(question)
    assert reply.confident and "March 31, 2026" in reply.text, reply.text


# --------------------------------------------------------------------------- a branch named in a news story


BRANCH = ("1 min read\n\nMarch 3, 2025\n\n# Credit union opens {town} branch\n\n{TOWN}, Kan.-The new branch at {street} in {town} "
          "is open.\n\nOpening times are:\n\n**Lobby**\nMonday-Thursday: 9:15 a.m.-4:45 p.m.\nFriday: 9:15 a.m.-{friday}\n\n"
          "**Drive-up**\nMonday-Friday: 7:45 a.m.-6:15 p.m.\n")


def test_hours_of_a_named_branch_come_from_the_page_about_that_branch():
    ks = helpers.build(("2026-03-05", {
        "newsroom/elm": BRANCH.format(town="Elmwood", TOWN="ELMWOOD", street="12 Elm Street", friday="6 p.m."),
        "newsroom/oak": BRANCH.format(town="Oakdale", TOWN="OAKDALE", street="40 Oak Street", friday="5:30 p.m.")}))
    elm = ks.answer("What time does the lobby close on Fridays at the Elmwood branch?").text
    oak = ks.answer("What time does the lobby close on Fridays at the Oakdale branch?").text
    assert "6 p.m." in elm and "5:30" not in elm
    assert "5:30 p.m." in oak
    assert "varies" in ks.answer("What are your lobby hours?").text.lower() or "vary" in ks.answer("What are your lobby hours?").text.lower()
    assert not ks.answer("What are the lobby hours at the Pinehurst branch?").confident


def test_a_branch_named_without_capitals_is_still_a_name():
    # The site names a Birchwood Avenue branch only in passing and never a Zorbleton Road branch: neither
    # question is answered with the Elmwood hours, however it is written. Each wording is asked on its
    # own system, as answers are kept by the question in small letters.
    elm = BRANCH.format(town="Elmwood", TOWN="ELMWOOD", street="12 Elm Street", friday="6 p.m.")
    pages = {"newsroom/elm": elm,
             "newsroom/staff": "1 min read\n\nMarch 9, 2025\n\n# Staff news\n\nAna moved to the Birchwood Avenue branch.\n",
             "newsroom/elm-year": "1 min read\n\nMay 1, 2025\n\n# A year in Elmwood\n\nThe Elmwood branch turned one.\n",
             "newsroom/elm-art": "1 min read\n\nJune 2, 2025\n\n# Art in Elmwood\n\nPainters show their work in Elmwood.\n"}
    for question in ("what are the lobby hours at the birchwood avenue branch?",
                     "What are the lobby hours at the birchwood avenue branch?",
                     "WHAT ARE THE LOBBY HOURS AT THE BIRCHWOOD AVENUE BRANCH?",
                     "what are the hours at the zorbleton road branch?"):
        assert not helpers.build(("2026-03-05", pages)).answer(question).confident, question
    assert helpers.build(("2026-03-05", pages)).answer("what is the elmwod branch address?").confident   # a slip, not a name


def test_a_name_in_capitals_is_not_read_as_a_word_one_letter_away():
    # The site uses "harvest" but has no Harvesta branch: the question is declined, not answered with the
    # Elmwood hours. Only a name written in small letters is spell-checked ("elmwod" above).
    elm = BRANCH.format(town="Elmwood", TOWN="ELMWOOD", street="12 Elm Street", friday="6 p.m.")
    pages = {"newsroom/elm": elm}
    for n in range(3):
        pages[f"newsroom/fair-{n}"] = f"1 min read\n\nMay {n + 1}, 2025\n\n# Fair {n}\n\nMembers met at the harvest fair.\n"
    assert not helpers.build(("2026-03-05", pages)).answer("What are the lobby hours at the Harvesta branch?").confident


def test_hours_volunteered_are_not_opening_hours():
    # Opening-hours questions need a passage with a clock time; hours volunteered are a count.
    pages = {"newsroom/elm": BRANCH.format(town="Elmwood", TOWN="ELMWOOD", street="12 Elm Street", friday="6 p.m."),
             "community": "# Community\n\nIn 2025 our staff volunteered 1,234 hours at food banks and schools.\n"}
    for question in ("total hours staff volunteered in 2025?", "How many hours did staff volunteer in 2025?"):
        assert "1,234" in helpers.build(("2026-03-05", pages)).answer(question).text, question
