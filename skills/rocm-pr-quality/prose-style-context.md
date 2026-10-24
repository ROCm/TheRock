# Prose and Style Context

## Purpose

Use this when writing pull-request descriptions, review comments, handoffs, design notes, and engineering summaries for any change in a ROCm repository, whatever the language or component.

A busy engineer should understand the change without reading the code or earlier pull requests. Accuracy is required. Repository knowledge is not assumed, but neither is ignorance.

## Voice

Write like a mass-market magazine covering a technical subject: clear, brisk, and a little human.

- Lead with what changed and why a reader would care.
- Use short, active sentences with a clear subject.
- Prefer the specific to the abstract: a number, a name, a before and after.
- Treat the reader as smart and curious. Explain a term in a clause, and only when the point depends on it.
- Vary sentence length. Use a connective (because, so, but) only when the link between two facts is not obvious. Read the paragraph aloud, and revise if every sentence has the same shape.
- Skip slogans, sales language, and in-jokes. A concrete comparison is welcome when it is shorter than the abstract explanation.

## Reader

A smart engineer from a neighboring team. They are comfortable with Python or C++ and new to this component. They may not know GPU terms, ROCm conventions, or project acronyms.

Gloss a term only if the change makes no sense without it. Words the repository's regular reviewers use every day (for example, "submodule" or "artifact" in TheRock) need no gloss. When unsure, gloss in a clause, not a sentence.

## Priority order

1. Correctness
1. Clarity, which includes brevity: a sentence the reader does not need makes the text less clear
1. Necessary context
1. Elegance

Cut words, not explanation. Do not replace an explanation with jargon.

## Length

Scale to the change. These are ceilings, not targets.

- **Trivial or docs-only:** one to three sentences per section. "N/A" is fine for a section that does not apply.
- **Typical:** 100 to 200 words in total.
- **Large, risky, or an unusual decision:** more, where each extra sentence carries evidence, risk, or a decision the reviewer must make.

Aim for sentences averaging 15 to 20 words, and rewrite any over 30. Keep most paragraphs to two through four sentences.

Length is a signal, not a pass/fail test. Never cut test evidence to hit a number. When a draft runs over, cut in this order: repetition, development history, process boilerplate, then disclaimers about things that did not happen.

## Sentence-level gate

Apply every question to every sentence. If any answer is no, rewrite it, or delete it when question 3 fails.

1. Can the reader tell what object, function, file, or process the sentence is about?
1. If a name or acronym is new and the point depends on it, is it defined, in a clause where possible?
1. Would the reader miss this sentence if it were deleted?
1. Does it make sense without an earlier pull request?
1. Does it carry one main idea with a clear subject and verb?
1. Is it concrete about an input, output, behavior, failure, or consequence?
1. Does the reader know why the fact matters?
1. Could a simpler verb replace an abstract noun phrase?
1. Does it avoid slogans and sales language?
1. Does it describe the current net change, not discarded development history?
1. Does the diff, a test result, or an authoritative source support every factual claim?
1. Could the reader paraphrase it correctly after one reading?

## Explain concepts before identifiers

Introduce the idea first, then give the code name.

Bad:

> `clusterEnabled` distinguishes the disabled unit cluster from changes to either axis.

Good:

> A graphics processor may run neighboring groups of work together as a cluster. `clusterEnabled` accepts `[x, y]`, the cluster size in two directions. It returns `False` for `[1, 1]` and `True` when either value is greater than 1.

Use backticks for literal identifiers, commands, values, and paths. An identifier does not explain itself.

## Terminology

- Define an acronym on first use when the reader may not know it: "Windows Subsystem for Linux (WSL)."
- Treat code-specific names such as `PAP`, `TDM`, `StreamK`, and `F32X` as labels, then say what behavior they control.
- Explain a domain term with a short concrete phrase: "tile, or block of matrix work," "kernel, or function that runs on the GPU."
- Define a technique the first time the change depends on it, in one plain sentence. Mutation testing makes temporary source changes and checks whether tests catch them. A template is C++ code the compiler stamps out once per type. An RAII type releases a resource when it goes out of scope.
- For an outside reader, prefer "saved expected result" (optionally "snapshot") to "golden"; "source-code version" to "revision"; "automated GitHub checks" to "CI"; "tests that record current behavior" to "characterization tests"; and "the set of functions and types other code calls" to "API." When the binary interface (ABI) can change, name what compatibility is at stake.

## Warning words

Each word below tends to stand in for a fact. If you use one, make sure the sentence still states the input, output, behavior, or measurement it replaces. Usually the fix is to delete the word or swap in the fact. Do not fix one by adding a sentence.

- **Review shorthand:** surface, contract, pin, wiring, provenance, fail-closed, golden, survivor, harden, durable, robust, comprehensive, surgical, idiomatic, canonical, first-class, source of truth, end-to-end, battle-tested.
- **Vague improvement (say what changed and by how much):** clean up, cleaner, simplify, streamline, optimize, improve, enhance, refactor (as the whole explanation), modernize, tweak, fix up, properly or correctly (as the only description of new behavior), gracefully.
- **Hand-waving (delete, or give the fact):** just, simply, merely, obviously, clearly, of course, basically, essentially, various, several, a number of (give the count), etc., and so on (name the rest or stop), as needed, if necessary (state the condition), should (say what it does, or what is unverified).
- **Claims that need evidence:** scalable, performant, fast, efficient (give the measurement), lightweight, production-ready, thread-safe (name what is now safe under which access), backward-compatible (name the interface and the consumers).
- **Marketing:** powerful, seamless, best-in-class, game-changing, blazing-fast, rock-solid, elegant, cutting-edge.

## Concrete language

State observable behavior instead of broad intent.

Bad:

> This hardens LibraryIO serialization contracts.

Good:

> The tests verify that an omitted output format selects YAML and that an unknown format prints the current error.

Bad:

> This refactors the allocator for robustness.

Good:

> `TensorBuffer` now frees its device memory in a destructor instead of a manual `free()` call, so an early return no longer leaks the buffer.

Bad:

> Improved kernel performance.

Good:

> The change tiles the inner loop by 64 elements, which reduced runtime on the `sgemm` benchmark from 1.8 ms to 1.2 ms on gfx942.

## Cut what the reader does not need

Concrete does not mean long. These pairs say the same thing in fewer words.

Bad (preamble before the news):

> hipBLASLt is a library that multiplies matrices on AMD GPUs. It chooses a kernel for each problem size. This change affects how it chooses a kernel when a matrix has an odd dimension.

Good:

> hipBLASLt picked a slow kernel for matrices with an odd dimension. It now picks the faster one.

Bad (explains what the reviewers already know):

> This change bumps the `rocm-libraries` submodule, a Git feature that pins another repository at a specific commit, to `e4f5a6b`.

Good:

> Bumps `rocm-libraries` to `e4f5a6b`.

Bad (a disclaimer about something that is not at stake, on a docs-only change):

> No tests were run on Windows or macOS. The GPU jobs did not run and are not required checks.

Good:

> Docs only. `pre-commit` passes.

## Pull-request structure

Keep the headings the repository's PR template and automated policy checks require. Each section has one job. Where a repository names a section differently, match the job to the closest heading.

### Issue or tracker line

Give the exact issue key or link, in the form the repository's policy check accepts. Keep it in the body when the title leaves it out.

### Motivation

Open with the problem and its consequence, in one to three sentences. Add what the component does only if an outsider could not follow without it. Do not begin with the implementation or a list of files.

### Technical Details

Describe the current net diff. Use bullets only for parallel facts a reviewer can check independently.

- Name exact inputs and expected outputs when useful.
- Give a small example when it says more than a label.
- Say whether production code, configuration, saved expected results, or device behavior changes.
- Link the direct dependency, if there is one.

Leave out files removed during an earlier decomposition that are absent from the current diff.

### Test Plan

Say what to run and what each group checks. Use exact file or module names when they help the reader reproduce it.

### Test Result

Keep completed evidence apart from pending automation.

- Give exact pass, skip, and saved-result counts.
- Do not call an environment failure a test failure.
- Write "Automated GitHub checks are pending" when that is the state.
- Do not claim a run without evidence.

Name what did not run only where the reader would otherwise assume it did, such as a platform or device the change affects. Do not list non-events.

### Submission Checklist

Keep the repository-required checklist item.

### Risk level

Give the level and the reason in a sentence, and name what can change.

Bad:

> Low risk.

Good:

> Low (1/5): test-only changes; product and device behavior are unchanged.

## Links

Link when an explanation the reader needs would bloat the body. Prefer repository documentation to external summaries, and use descriptive link text, not a raw URL. A link adds depth. It does not replace the purpose, behavior, evidence, and risk, which stay in the body. Do not link to stale branches, private paths, temporary files, or unverified documents.

## Formatting

- Use only the headings the repository template requires.
- Use bullets for parallel facts, not for ordinary prose. Avoid nested lists and bold used as organization.
- Put code identifiers in backticks. Use numerals for exact inputs, counts, versions, and percentages.

## Evidence and scope

These rules keep a claim from overreaching. They do not ask for a list of everything that did not happen.

- Describe what the pull request contains now.
- Separate changes to tests, build files, and configuration from changes to production behavior.
- Separate current behavior from behavior judged to be correct. Passing tests that record existing behavior do not prove it is correct.
- Do not treat one metric (line coverage, mutation score, a benchmark delta) as a quality claim alone. Say what was measured, on what input, on what hardware or platform.
- Do not hide pending checks behind "all tests pass."
- Do not claim device coverage for host-only tests, or carry a result from one GPU architecture to another.
- When the change can affect the functions and types other code calls, or the binary interface, say so and name the consumers.

## Drafting workflow

1. Write the lead first: what changed, and why a reader would care.
1. List the technical nouns and acronyms. Gloss only the ones the point depends on.
1. Replace each broad claim with an input, output, example, or measurement.
1. Run the sentence-level gate, one sentence at a time.
1. Cut pass: with the draft correct, remove about a fifth. Take repetition, history, preamble, and needless disclaimers first.
1. Read the result as a stand-alone entry point, not as one layer of a known stack.
1. Check every number and test claim against recorded evidence.
1. Check the required headings, the issue key, the dependency, and the length for the size of the change.

## Final acceptance test

Before delivering prose, a reviewer should be able to answer these without opening the diff:

1. What does this pull request change, and why?
1. What was wrong or missing before?
1. What evidence has passed, and what is pending?
1. What production or device behavior can it affect?

If an outsider could not follow the first answer without knowing what the component does, add that in one sentence. If any answer is missing, the prose is not ready.
