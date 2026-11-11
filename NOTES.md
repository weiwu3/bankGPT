# Notes

The solution is the `solution/` folder (`solution.py` with `build()`; the work is in
`solution/kb/`). It uses a paid API and needs one key: **`ANTHROPIC_API_KEY`**.

## How it works

The old assistant combined the downloads and gave out-of-date answers. I keep each download as
its own dated copy of the site and split the work in two:

- **Rules keep the books.** Rate and fee tables are read into rows, one per product, term and
  tier, and branch hours into schedules. A question that a row or a schedule answers gets an
  exact answer in under 0.15 s, the same every time: 20 of the 40 practice questions. Two more
  are declined by rule.
- **A language model reads the prose.** Every other question goes to one model call. The model
  sees only what the rules prepared: the current rows nearest the question, and passages from
  the download in force with old versions of a statement removed, product pages first and
  articles newest first, each tagged with its kind of page. It declines when the evidence does
  not say.

A question with `as_of` runs through the same steps on the last download on or before that date,
so a later value cannot appear in the answer. (A date before the first download gets the first
one.)

## How I decide what is current

Two copies compete only when they are the same row: same product, term, balance or loan-amount
tier, model years, personal or business. A different tier is a different fact, not a
contradiction. When two copies disagree, the first rule that separates them decides
(`RowIndex.better` in `kb/lookup.py`):

1. A value whose line, or the heading over its table, says its offer has ended loses.
2. The more trusted kind of page wins: a product or rates page, then a published document such
   as a fee schedule, then a campaign page, then an article or a calculator. A page with more
   out-of-date rows than current ones is trusted no more than a campaign page, and a page that
   has gone from the site a little less than a live page of its kind.
3. The newer "rates effective as of" date wins. A table with no date line takes the date the
   same table shows in a later download of its page. A table I only know changed between two
   downloads is dated to the earlier one, as a lower bound.
4. A copy its page keeps updating beats one that never changes.
5. The value shown on more pages wins, not counting copies a newer date has already beaten.
6. A copy with a printed date beats one without.
7. The copy whose page changed most recently wins.
8. A table row beats a sentence.
9. Last, the page with the shorter address wins, so the order of the files never decides.

Whole tables compete too (`resolve` in the same file). A page that still prints a product's
tiers under an older scheme, such as last year's model years, has no row that matches a current
one. Then the first four rules pick the newer edition of the table, which speaks alone, and a
calculator's table is dropped when a product page has one. This is why the rates page's 5.39%
beats the 5.34% that a calculator and a forgotten product page still show.

On these five downloads rules 2 and 3 do the work: of the rows whose copies show different
figures, rule 2 settles 31, rule 3 settles 12 and rule 4 one. Most of rule 2's cases are a
product page demoted for its out-of-date rows. No row here says its offer has ended, so rule 1
settles none, and neither do rules 5 to 9: they make sure two copies always have one winner,
whatever order the pages arrive in. `tests/test_current.py` has one test per rule on a small
made-up site, and each fails when its rule is removed.

For prose, long passages that read the same once their digits are masked are versions of one
statement, and only the copy on the most trusted, newest page is kept; teasers of an article on
other pages are dropped.

**"I don't know."** A rate asked for a kind of product that no table lists and no passage prices
(a construction loan) is declined by rule, and the refusal says what is missing. Otherwise the
model declines when the evidence has nothing on the thing asked. A refusal never states a number
that is not in the question. With the model, a question about a past year's count that the site
does not give is answered with the two latest years' figures, and the answer says that year has
none.

**Contradictions.** `conflicts()` reports four kinds: rates and fees that differ between copies
of a row or changed between downloads; figures the site restates with different numbers, such as
the member count; a closed branch that a branch list still offers; and amounts edited between two
downloads of a page. For rates and fees the winner is chosen by the rules above; for the other
kinds the newest statement stands. It reports 25 on these downloads.

## The model, the key and the cost

The solution calls `claude-sonnet-5-5` through the `anthropic` package: one call for each
question that reaches the model (18 of the 40 practice questions), about 8,700 tokens in and
170 out per call, and nothing while loading. The practice run costs $0.34.

| Model (`BANKGPT_MODEL`), one run each | Practice accuracy | My 313 questions | Answers with a forbidden value | Seconds per call, median / 95% | Cost per 1,000 model answers |
|---|---|---|---|---|---|
| `claude-sonnet-5-5` (default) | 0.969 | 0.962 | 2 | 1.7 / 2.6 | $19.81 |
| `claude-haiku-4-5` | 0.946 | 0.942 | 1 | 1.7 / 2.9 | $7.33 |
| `claude-haiku-5-5` | 0.946 | 0.945 | 4 | 1.1 / 1.6 | $0.99 |

(Sonnet's call times are from its 195 calls on my questions. The Haiku rows are one recorded
pass each over the same 353 questions.)

Sonnet is the most accurate, so it is the default. Haiku 5.5 costs a twentieth as much and its
calls are fast enough for full speed marks, but it is about two points less accurate, and by the
scorer's weights that leaves Sonnet about one point ahead. The forbidden-value column is noise at
this size: another pass of Sonnet over the same questions scored 0.963 with none. All my own
measuring cost about $94 of API calls.

Without a key the solution runs on the rules alone: table and hours questions are unaffected,
the rest get the best-matching sentences quoted from the site or a refusal, and practice
accuracy falls from 0.969 to 0.752. When a call fails, a busy or slow model hands the question
to the next one (`claude-haiku-4-5`, then `claude-haiku-5-5`) and an unreachable API is asked
again; after three questions in a row with no reply the next five skip the model, and a rejected
key turns it off. No answer took more than 42 s in my tests with a failing API; a question that
gets no reply is answered without the model.

Results, a comparison with plain search, limits and how to check all of this are in the
[README](README.md). I built this with Claude Code; the invitation encourages AI-assisted
development.
