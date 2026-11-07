# DocPilot: UI Report (November 2026)

**Branch:** `ui/polish-2026-11`, built on `fix/verified-issues-2026-11`. Not merged.

**Scope:** the web UI (`ui/index.html`, served at `/`) and the About text in `gradio_app.py`.

**Environment:** a cloud container with no GPU, no Hugging Face access and no Groq key. The real QA model could not load, so the screenshots use a **clearly labelled stand-in engine**. Its answers are not model output, and **no new metrics are reported here.**

---

## 1. Flaws found in the old page

I ran the old page and opened it in headless Chromium at 1280 px and 390 px. These are the problems I found.

1. **It broke completely without the network.** All styling came from the Tailwind CDN, and fonts came from Google Fonts. With those blocked, the page rendered as unstyled text. Every tab was shown at once, and two SVG icons filled the whole screen. Screenshots of that were not kept; you can reproduce it by loading the `main` branch page offline.
2. **It claimed more than the code does.** It showed F1 71.1 and 67.8, "INT8 P95 latency = 120 ms", "Model size 120 MB", "10+ pattern classes" and "14 entity types detected & redacted", with no caveats.
3. **There was no way to send an API key.** With `REQUIRE_AUTH=1`, every request failed with no explanation. Since Fix 7, this also applies to `/security/*`.
4. **It didn't show the server's state.** The retrieval mode wasn't visible. Nothing explained that the QA model, dense retrieval or Groq was unavailable. A 503 or 401 error showed as a generic failure.
5. **Low contrast.** Most secondary text was zinc-600 on near-black (#07070f), which is well under WCAG AA for body text.
6. **Keyboard and screen-reader support.** The tabs were plain buttons with no ARIA tab roles and no arrow-key support. Focus was barely visible.
7. **Confusing labels.** "confidence" was really a raw logit sum. "Grounded 100%" read like a correctness check. "SAFE" was shown when no pattern matched, which overstates what a regex check can know. One example question (WebRTC) wasn't covered by the demo corpus.
8. **Missing states.** Long requests had no loading state on the button, an empty question gave no message, and there was no clear message for a blocked question.

## 2. What changed

- **`ui/index.html` was rewritten** as plain HTML, CSS and JavaScript, with **no external requests**. A test enforces this.
- **Header status, read from `/health`:**
  - QA model loaded or not loaded
  - retrieval mode (`bm25`, `hybrid`, `hybrid+rerank`, with plain labels)
  - LLM synthesis on or off
  - passage count
- **Plain notices** for each unavailable piece: no model (with the command to build it), empty corpus, dense retrieval off, Groq off.
- **Answer card:**
  - answer type (extracted span or LLM)
  - grounding percentage, with a tooltip saying it is word overlap
  - latency for retrieval, QA, synthesis and total
  - QA score, labelled as a logit sum
  - retrieval mode and sources
  - the passage text used, collapsible
- **Error and empty states:**
  - **Loading:** the button is disabled, with a spinner and `aria-busy`.
  - **Empty question:** a prompt to type one.
  - **No confident answer:** a suggestion to rephrase.
  - **Blocked question (400):** shows the pattern class and the matched text.
  - **401 / 403:** says to add a key in Settings, or that the role isn't allowed.
  - **503:** "model not loaded".
  - **Network error:** "Can't reach the server".
- **Security checks tab:**
  - A banner says the checks are **pattern-based (regular expressions)** and names what they miss.
  - "No pattern matched" replaces "SAFE", with a note that this doesn't mean the text is safe.
  - Examples now include a benign look-alike.
- **Audit log tab:** shows stats, recent queries (with previews labelled as redacted) and security events, with empty states and an explanation of what "flagged" means.
- **Settings tab:** an `X-API-Key` field. The key is kept in memory only, unless "remember" is ticked.
- **About tab:** describes the pipeline as the code works today and states what is and isn't measured. It follows the `FIX_REPORT.md` wording rules: no F1/EM figures and no GPU numbers, and the latency figure is described as per-pass, not per-request.
- **Accessibility:** ARIA tabs (Left/Right/Home/End keys), Enter to ask, Shift+Enter for a new line, visible focus rings, light and dark themes from `prefers-color-scheme`, reduced-motion support, and no horizontal scroll at 390 px.
- **Copy:** plain wording, no em dashes. A test enforces this.
- **Bug fixed:** the Redact handler read `e.currentTarget` after an `await`. At that point it is `null`, so the button stayed stuck in its busy state and no result appeared.
- **API change (additive only):** `/health` now has `qa_engine_label`. It is `null` for the real engine, and only a stand-in sets it, so the UI can label it. No other API changes.
- **`gradio_app.py`:** the About text was rewritten to the same honest wording. It said "enterprise-grade" and "10+ pattern classes", and claimed PII was stripped on every path. The window title and the "SAFE" message lost their em dashes. **Gradio isn't installed here, so this app was not run;** I only checked that it parses.

## 3. What was tested

| Check | Result |
|---|---|
| `python -m pytest -q` | **42 passed** (the 38 earlier tests plus 4 new UI tests in `tests/test_ui.py`) |
| `python -m tests.smoke_test` | 20 / 20 |
| `ruff check . --select E,F,W --ignore E501` | All checks passed |
| Headless Chromium, 1280 px and 390 px | Ask flow, blocked question, security checks, PII redaction and audit log all ran. Width with no horizontal scroll: 1280 / 390. ArrowRight moves focus to the next tab. The only console errors are the expected HTTP 400s from the blocked-question test. |
| "Model not loaded" state | The red notice is shown, and asking gives a clear 503 message (`docs/ui/no-model-desktop.png`). |

`tests/test_ui.py` checks four things:
- no CDN or external `src`/`href`
- no em dashes and no stale claims
- every endpoint and retrieval mode is wired up
- `/health` returns `qa_engine_label`, and `/` serves the page

**Screenshots in `docs/ui/`** (all taken with the stand-in engine, which is labelled in the header and in a notice):

| | |
|---|---|
| `ask-desktop.png`, `ask-phone.png` | Answer with metadata |
| `blocked-desktop.png`, `blocked-phone.png` | Blocked question |
| `security-desktop.png`, `security-phone.png` | Injection check and PII redaction |
| `audit-desktop.png`, `audit-phone.png` | Audit log |
| `no-model-desktop.png` | Server with no QA model |

## 4. Not tested

- **The real QA model, dense retrieval, the reranker and Groq.** I did not see how the UI looks with LLM answers (`groq-8b`, `groq-70b`, `cached`) or with `hybrid+rerank` mode. The code paths and labels are there, but haven't been seen live.
- **`REQUIRE_AUTH=1` in the browser.** The 401/403 messages are tested only by reading the code. The API-level auth tests are in `tests/test_auth.py`.
- **The Gradio app.** It was not run, because gradio isn't installed here.
- **Real browsers and screen readers.** Only headless Chromium was used: no Safari or Firefox, no real phone, and no screen-reader pass.
- **Formal contrast audit.** No automated contrast or axe audit was run; the axe tool failed to connect in this session. The colours were chosen for AA contrast, but that wasn't measured.

## 5. Running it locally

With the real model (after `make demo-corpus train export quantize`), open http://localhost:8000/:

```bash
make serve
```

To look at the UI without the real model, use the stand-in engine. It is clearly labelled, and its answers are not model output:

```bash
make demo-corpus
```

```bash
python docs/ui/standin_server.py
```

To see the "model not loaded" state:

```bash
NO_MODEL=1 python docs/ui/standin_server.py
```

With auth on, enter `my-admin-key` in the Settings tab:

```bash
REQUIRE_AUTH=1 API_KEYS="my-admin-key:admin" make serve
```

Tests and lint:

```bash
python -m pytest -q && python -m tests.smoke_test && ruff check . --select E,F,W --ignore E501
```

## 6. Interview notes

- **Why drop Tailwind?** The page broke completely when the CDN was unreachable. That's a real failure mode on locked-down networks. About 150 lines of plain CSS covers what this page needs, with no build step.
- **Why show "not loaded" and "BM25 only" so prominently?** So nobody mistakes degraded mode for full mode. The UI reports what `/health` measured, not what the README promises.
- **Why "No pattern matched" instead of "SAFE"?** A regex check can only say it didn't find a known phrasing. On fresh rewordings it caught 2 of 10.
