# Feedback Pulse

A lightweight tool that turns a product's raw customer reviews into a quick, honest read on sentiment — built with a pretrained Hugging Face Transformers model, no training required.

## Why

Companies collect hundreds of reviews per product, but star ratings alone are a blunt signal: they don't say whether sentiment is improving or getting worse, and nobody has time to read every review by hand to find out. This tool automates that read — given a product, it tells you the real sentiment split from the review text itself, checks it against the star ratings, and flags whether things are trending up or down over time, so a business can catch a problem (or a win) early instead of discovering it in a quarterly report.

## What it does

Given a product ID from the Amazon Fine Food Reviews dataset, the tool runs that product's reviews through a pretrained sentiment classifier and prints a report:

- The % of reviews classified positive vs. negative
- How that compares to the product's star ratings
- Whether sentiment is trending up or down, comparing the earliest half of reviews to the most recent half

## How it works

- **Dataset:** [Amazon Fine Food Reviews](https://huggingface.co/datasets/PJ2005/amazon-fine-food-reviews) (~568K reviews, single domain — food/grocery — chosen over a larger mixed-category set so the output stays specific rather than generic).
- **Model:** `distilbert-base-uncased-finetuned-sst-2-english`, used as-is with no fine-tuning. No Google Colab access and no local GPU meant CPU fine-tuning would have burned days the schedule didn't have — so the project leans into an applied-AI framing instead: take an existing model and build a genuinely useful tool around it, rather than train one from scratch.
- **Trend calculation:** reviews are sorted chronologically and split into an earlier half and a later half by *count*, not by calendar month — review volume for a single product is often extremely bursty (most reviews arriving in a short window after launch), so calendar-based buckets can be misleadingly small. A count-based split stays reliable regardless of how a product's reviews are distributed in time.
- **Minimum data check:** a product needs at least 10 reviews for the tool to generate a report — below that, both the overall stats and the trend split become unreliable, so the tool refuses and says why instead of printing a misleading number.

## Setup

```bash
git clone https://github.com/jandaghi14/1-small_project_feedback-classifier
python -m venv venv
venv\Scripts\activate        # Windows (use `source venv/bin/activate` on Mac/Linux)
pip install -r requirements.txt
```

Download the dataset from [Hugging Face](https://huggingface.co/datasets/PJ2005/amazon-fine-food-reviews) and place it at `data/reviews.csv`.

## Usage

```bash
python analyze.py --product B007JFMH8M
```

Example output:
1- For a product with more than 10 reviews:
Loading weights: 100%|██████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 294.44it/s]
The rating percentage is: rating_sentiment
POSITIVE    0.921053
NEGATIVE    0.017544
NEUTRAL     0.061404
Name: proportion, dtype: float64
The predicted percentage is: predicted_label
POSITIVE    0.916667
NEGATIVE    0.083333
Name: proportion, dtype: float64
The first half of the data percentage is: predicted_label
POSITIVE    0.953947
NEGATIVE    0.046053
Name: proportion, dtype: float64
The second half of the data percentage is:predicted_label
POSITIVE    0.879386
NEGATIVE    0.120614
Name: proportion, dtype: float64

2- For a product ID that has too few reviews, the tool prints a clear error instead of crashing:
Loading weights: 100%|██████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 367.01it/s]
Only 1 reviews found for product B005OTVL8C — need at least 10 to generate a reliable report.

3- For a product ID that doesn't exist , the tool prints a clear error instead of crashing:
Loading weights: 100%|██████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 169.51it/s]
No review found for product ID: IncorrectProductId



## Limitations

- **No neutral category in predictions:** the classifier only outputs positive/negative, so mixed or lukewarm reviews get forced into one bucket. The star-rating comparison (which does have a neutral tier) is included partly to surface this gap.
- **Struggles with informal slang:** phrases like "the bomb" (meaning great) or "falls apart in your mouth" (meaning pleasantly soft) can read as literally negative to a model trained mostly on more formal text, and get misclassified as a result.
- **Trend needs enough data to mean anything:** the tool guards against products with too few reviews, but even above that threshold, a trend from a handful of reviews should be read with caution.