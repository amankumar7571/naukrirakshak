# NaukriRakshak — AI Fake-Job-Offer Detector

**Catalyst – Solo AI Hackathon, Makers Need More (MnM), 15 Nov 2026**
Built solo by Aman Kumar, B.Tech CSE (AI & ML), KIET '27.

Paste any job message, posting or offer letter — WhatsApp HR message, email,
portal listing — and NaukriRakshak tells you in seconds whether it is a scam,
and shows you exactly why.

## The problem

Every day, Indian freshers lose ₹5,000–₹50,000 to fake job offers:
"refundable security deposits", WhatsApp-only HR contacts, offer letters
with no interview. First-generation job seekers have no way to tell a real
recruiter from a fraud. This tool is for them.

## How it works (two stages, real AI — not a chatbot wrapper)

1. **Trained classifier (stage 1, runs in the browser).** TF-IDF (word 1–2
   grams) + logistic regression, trained in scikit-learn on 117 labelled
   representative examples. The full exported vocabulary (2,063 terms,
   coefficients + idf) runs client-side with exact TF-IDF math — sublinear
   term frequency, L2 normalisation — so the browser score matches the
   trained model to within 0.01. No server, no API key, works offline.
2. **Evidence extraction (stage 2).** The exact suspicious phrases are
   highlighted in the pasted text with a plain-language reason for each
   ("Real employers never charge a fee to hire you"). Gemini is the
   production explainer; the demo ships an offline explainer so the demo
   can never die on a quota error.

Risk bands: **0–39% Looks genuine** · **40–64% Uncertain — verify these
3 things** (check the company, never pay a fee, confirm the HR email domain)
· **65–100% Likely scam**.

## Measured results (honest numbers)

Held-out test, stratified 80/20 split of the 117 labelled examples
(93 train / 24 test):

| Metric | Value |
|---|---|
| Held-out accuracy | 24 / 24 correct (1.000) |
| Precision (scam) | 1.000 |
| Recall (scam) | 1.000 |
| F1 (scam) | 1.000 |

**Honest caveat:** 24 test examples is a small set, and they are
representative examples of common patterns — real-world accuracy will be
lower. That is exactly why every verdict shows its evidence and an
"uncertain" band instead of pretending to be sure, and why reported
messages retrain the model (human-in-the-loop).

**Judge review (14 Nov 2026):** an adversarial review found the first
build's browser scorer was a rough approximation (it showed 99% for a
message the trained model scores 69%). The scorer was rebuilt with exact
TF-IDF math and the training set was hardened with 33 adversarial examples
(genuine small companies hiring over WhatsApp, explicit "no fee" postings,
legitimate work-from-home and urgent hiring, subtle scams, Hinglish). The
numbers above are the true sklearn measurements, and the browser scorer
reproduces them within 0.01 (verified in `train.py`, see `test_vectors.json`).

## Features

- One-click demo samples: scam WhatsApp message, genuine posting, borderline
  consultancy message
- "Money saved" counter — when a scam mentions an amount (₹N), the app
  shows "You just avoided losing ₹N" and keeps a session total
- "Report this message" button — reported messages feed model retraining
- "How it works" section with the real pipeline and tech stack
- Mobile-first, works fully offline, no sign-in

## Try it

Open `naukrirakshak.html` in any browser — no install, no server.
Or use the live demo link in the submission.

## Reproduce the model

```bash
python train.py   # trains TF-IDF + logistic regression, prints metrics,
                  # exports model.json (weights) and metrics.json
```

Requires Python 3.12 + scikit-learn 1.8.

## Repo structure

- `naukrirakshak.html` — the full working app (single file)
- `train.py` — training script + labelled dataset (representative examples)
- `model.json` — exported classifier weights used by the app
- `metrics.json` — held-out test metrics

## Roadmap

Bigger labelled dataset from user reports · MCA/LinkedIn company
verification layer · WhatsApp forward-to-check bot · Hindi interface ·
browser extension.

## Author

Aman Kumar — B.Tech CSE (AI & ML), KIET Ghaziabad, Class of 2027.
GitHub: https://github.com/amankumar7571 ·
LinkedIn: https://www.linkedin.com/in/amankumar7571/

## Current files for the GitHub repo (15 Nov 2026)
The app.html and pitch-deck.html in this folder are the PRE-panel-fix exports. The current versions (Hinglish toggle, green band renamed "No scam signs found", plain-language checks, 1930/cybercrime.gov.in line, Devanagari note, honest roadmap) are the two exported files: ~/workspace/your_files/naukrirakshak/naukrirakshak.html and ~/workspace/your_files/naukrirakshak-pitch-deck/naukrirakshak-pitch-deck.html — upload those as app.html and pitch-deck.html instead of the copies in this folder. Everything else in this folder (train.py, model.json, metrics.json, test_vectors.json) is current.
