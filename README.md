# A knowledge system that knows what is true now

My submission for interface.ai's knowledge assignment: answer credit-union members' questions
from five dated downloads of the credit union's website, with the current value, the right tier,
a source, and an honest "I don't know".

- The solution is the [`solution/`](solution/) folder: `solution.py` with `build()`.
- [`NOTES.md`](NOTES.md) says how it works, how "current" is decided, which API key it needs and
  what that costs.

## Run it

Python 3.9 or newer. The website downloads are not in this repository: put the kit's
`snapshots_zip/` folder at the top level, and the scorer unpacks it on its first run.

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
python harness/run_eval.py --submission solution --eval eval/public_eval.jsonl
```

With the key, the 40 practice questions make 18 model calls and cost about $0.34. Without it the
solution answers from its rules alone, and practice accuracy is 0.752 instead of 0.969.
`BANKGPT_MODEL` picks another model; `BANKGPT_LLM=off` forces rules only.

## Results

The organisers' scorer with the default model, `claude-sonnet-5-5`:

| | Accuracy | Consistency | Speed (p95) | Final |
|---|---|---|---|---|
| 40 practice questions | 0.969 | 1.00 | 0.95 (2.9 s) | 0.96 |
| The same without a key | 0.752 | 1.00 | 1.00 (0.07 s) | 0.75 |
| 313 questions of my own | 0.962 | 0.99 | 0.97 (2.5 s) | 0.95 |

"Final" leaves out the conflicts part: the kit has no list of this site's contradictions. With
my own list as the answer key (see Limits), the conflicts part scores 0.54 and the scorer's final
for the 313 is 0.89. One practice question lists no value to match, so 0.969 is the most the
scorer can give on that set. Graded afterwards with the scorer's own judge (the rule its `--judge`
option applies, run once on the saved answers), the same answers score 0.986. Loading the five
downloads takes about 37 s, and a whole run without the model peaks at about 0.45 GB of memory.
The tests pass on Python 3.9 to 3.12; the runs through the API were on 3.11 with `anthropic` 2.8
(on 3.9 pip installs an older `anthropic`, 0.125, which the tests pass with).

**What each part contributes.** I compare four systems on the same 353 questions (the 40 practice
questions and my 313). The numbers are accuracy as the scorer counts it. The two search baselines
ran earlier with the same model; the final tree's baseline code sends the same 353 messages.

| How my solution answers the question | Search + model, all downloads combined | Search + model, download in force | My rules, no model | Rules + model (the solution) |
|---|---|---|---|---|
| From a table row, by rule (111 questions) | 0.564 | 0.813 | 0.969 | 0.969 |
| From a published schedule, by rule (18) | 0.277 | 0.517 | 1.000 | 1.000 |
| Everything else (224) | 0.597 | 0.767 | 0.500 | 0.956 |
| Of all these, with an `as_of` date (40) | 0.393 | 0.871 | 0.886 | 1.000 |
| **All 353** | **0.568** | **0.768** | **0.684** | **0.963** |
| Answers with a value the key forbids | 29 | 5 | 8 | 2 |

"Search + model" is plain keyword search handing its 20 best passages to the model, with the
instruction my reader also has never to mention a superseded value. The first column stands in
for the system the assignment describes: it searches all downloads as one and ignores `as_of`.
Searching only the download in force lifts it from 0.568 to 0.768, and on dated questions from
0.393 to 0.871. The rows group the questions by the route my solution takes without a model; on
the first two the model is never called, so the rules and the solution are the same thing. The
rules add exact answers on tables and hours, and the model, shown evidence the rules prepared,
adds the prose the rules cannot read.

**My 313 questions** were written by AI agents that worked from the downloads and were told not
to open my code: four topic sets (branches, deposits, loans, news), two rewordings of every
practice question, "as of" variants, everyday phrasings with typos, and questions from randomly
chosen pages. I developed against one half of them (144 questions) and held the other half (169)
back at first: the solution scores 0.981 on the first and 0.946 on the second. The second half
includes 50 rewordings of practice questions, and later error reviews also looked at its misses,
so neither number is a blind test.

**A blind check.** Near the end three more agents wrote 180 questions in members' plain wording
(rates, things the site does not say, everything else), again from the downloads alone. I ran one
half first: 0.81. Its misses had nine general causes, such as a member's word that is not the
site's word, an ordinary adjective read as a kind of product, and a caveat in the model's answer
read as a refusal. I fixed six of them, then ran the other half once: 0.917 over 90 questions.
Fifteen of its refusal keys also list a source, which a refusal cannot cite, so 0.959 is the most
that half can score. It is one run, on the commit before the last fixes; I take 0.92 as my
estimate for plainly worded questions like these that the solution has not seen. It lost points
through the solution on four questions: "overseas" for international, "MM" for money market, an
amount written without "$", and a table matched through one word of a page title. I fixed the
first three afterwards, so that half is no longer unseen; the fourth is in the limits below.

The question sets and my list of contradictions are not in this repository: they restate the
credit union's site, which the assignment asks not to redistribute. I can share them.
`dev/run.py` takes any file in the practice file's format. For the same reason the commit history
here was rebuilt from the final tree: my working history had the site's text in early tests.

## Limits

- **The model's reading.** Free-text answers depend on it. It sometimes declines a question whose
  answer is on the site in other words. A second run words most model answers differently; in my
  last comparison it changed the score of 3 of 353 questions.
- **Two pattern checks on the model's reply.** An answer that says the site does not state
  something becomes a refusal. There are two exceptions: a yes-or-no question, and an answer that
  first gives a figure and whose gap is a caveat about something the question did not ask (unless
  the rate tables lack the rate asked for). A reply that declines but gives the latest years'
  figures for a past year that has none becomes an answer. I measured both on 639 recorded
  replies from the three models; a caveat in words I have not seen can still go the wrong way.
- **The quote check is not a check on the answer.** A quote the model gives that is not in the
  evidence is dropped, but the answer goes out either way.
- **Words.** Search is by words, with a short list that maps a member's word to the site's
  ("overseas" to international). Each of my last two sets of questions found words the list
  lacked.
- **A table can answer on a half match.** A question whose own nouns are all general ("a loan
  to help rebuild my credit") can match a rate table through one word of a page title and get
  that table's rates. Two fixes each lost one or two measured questions, so it stays.
- **Rows are picked by the words of their names.** Asked for "the fee for a wire transfer", the
  answer lists the two fee lines that have "transfer" in their names and leaves out the domestic
  one, whose line does not; "your wire fees" lists all three. The model does not see answers made
  from rows, so it cannot add the missing line.
- **Contradictions** cover rates, fees, restated figures, closed branches and edited amounts.
  Against a list my question-writing agents made on their own, the report finds 17 of 36 and
  names the listed winner for all 17. Of the 19 it misses, 10 are changed links, dates, names or
  phone numbers, which it does not look for; the other 9 are figures or terms it passes over,
  such as an introductory rate.
- **Branch hours** are on the site only in seven branch announcements from 2021 to 2024. An
  answer for a named branch gives the announcement's date and cites it. A question that names no
  branch gets every distinct schedule, without dates. Nothing newer exists to confirm them.
- **Page kinds** are read from the address and the top of the page, and an address that matches
  no pattern counts as a product page. A site organised differently would need that list changed.
- **Without a model** (no key, or a call that got no reply) table and hours answers are
  unchanged, but the other questions get the best-matching sentences quoted from the site or a
  refusal: 85 of the 213 that would go to the model are fully right. A quoted answer is marked
  confident even when its passage is off topic.
- **Sources.** An answer cites up to ten pages: every page that prints the winning value and,
  for model answers, the passages it named and the best-ranked pages. The scorer does not
  penalise extra sources; a product would cite fewer.
- **Repeats.** The same question asked twice gets the cached answer, so the scorer's repeat
  check always passes. Across runs a few model answers change.
- **Dates.** An `as_of` date before the first download is answered from the first download, and
  a value that cannot be read as a date from the newest. Downloads must arrive oldest first, as
  the scorer sends them: out of order, the last one added counts as current, with no warning.
- **Scale.** One site per process: the site's learned names are process-wide, and each new
  download rebuilds the indexes of every earlier one.
- **Not built:** a rate that depends on a credit score given in a footnote; a date given only in
  the question's words (the practice questions carry `as_of` for those); a list put together
  from the links of a hub page.
- **A crash is answered "I don't know".** A question that breaks the code gets the fixed refusal
  and its traceback goes to stderr, so the scorer would count it as a refusal. None did in my
  runs.

## What is where

```
solution/
  solution.py        build(): the object with ingest(), answer(), conflicts()
  kb/
    engine.py        the router: rows, hours, "not published", the reader, else a quote
    store.py         every version of every page; one view of the site per download
    pages.py         what kind of page this is (product page, article, calculator ...)
    tables.py        rate tables and labelled values -> rows, one per tier
    lookup.py        match a question to rows; decide which copy of a row is current
    compose.py       word an answer from rows
    hours.py         branch hours as schedules
    passages.py      ranked text passages, links, footnotes, the contact paragraph
    reader.py        what the model is shown, and the checks on its reply
    transport.py     the API calls, their failures, and what they cost
    quote.py         the route with no model: quote the site or decline
    reply.py         what every route returns; the one place a refusal is worded
    conflicts.py     the contradiction report
    text.py          tokens, synonyms, terms, amounts, dates
tests/               the rules, the reader, the API failure handling and the dev tools, on made-up
                     sites (seven tests use the real downloads when they are present)
dev/                 development tools
snapshots/           where the scorer unpacks the five downloads (not in the repository)
harness/, eval/, starter/    the assignment kit, unchanged
```

## Check it

```bash
pip install -r requirements-dev.txt
python -m pytest tests                              # 1,228 tests; 1,221 need no downloads, none needs a key or the internet
python dev/run.py eval/public_eval.jsonl --fail     # every missed practice question, with the answer given
python dev/conflicts.py                             # the contradiction report
python dev/outage.py                                # the scorer through 22 kinds of API failure and a control run, on a local fake API (about 28 minutes)
python dev/find.py "17 Month" --snap 05             # search the downloads
```

With the key set, `dev/run.py` calls the model (about $0.34 for the practice file);
`BANKGPT_LLM=off` runs it on the rules alone. `dev/baseline.py` is the plain search-and-ask system
of the comparison above. Setting `BANKGPT_SPEND_LOG=results/spend.jsonl` records what each run's
model calls cost, and `python dev/spend.py` adds the file up.
