import pandas as pd
from transformers import pipeline
import argparse


parser = argparse.ArgumentParser(description="Analyze sentiment for a product's review.")
parser.add_argument("--product", required=True, help="Amazon ProductId to analyze")
args = parser.parse_args()
product = args.product

df = pd.read_csv("data/reviews.csv")
df = df.drop_duplicates(subset=['UserId' , 'Text']).reset_index(drop=True)

classifier = pipeline("sentiment-analysis" , model ="distilbert-base-uncased-finetuned-sst-2-english")

def score_to_sentiment(score):
    if score >= 4:
        return "POSITIVE"
    elif score <=2:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

def generate_report(df, product_id):
    # 1. filter df to just this product's reviews
    product_df = df[df['ProductId'] == product_id].reset_index(drop = True)
    # 1.5 Validate input
    if len(product_df) == 0:
        raise ValueError(f"No review found for product ID: {product_id}")
    elif len(product_df) <10:
        raise ValueError(f"Only {len(product_df)} reviews found for product {product_id} — need at least 10 to generate a reliable report.")
    # 2. run the classifier on the Text column (remember truncation=True),
    #    attach predicted_label as a new column
    
    
    prediction = classifier(
        product_df['Text'].tolist(),
        truncation = True
    )
    product_df['predicted_label'] = [pre['label'] for pre in prediction]

    # 3. compute % positive/negative from predicted_label (value_counts normalize=True)
    percentage = product_df['predicted_label'].value_counts(normalize = True).reindex(['POSITIVE' , 'NEGATIVE'] , fill_value = 0)

    # 4. convert Score to rating_sentiment (your score_to_sentiment function),
    #    compute its % breakdown too
    
    product_df['rating_sentiment'] = product_df['Score'].apply(score_to_sentiment)
    rating_breakdown = product_df['rating_sentiment'].value_counts(normalize = True).reindex(['POSITIVE' , 'NEGATIVE', 'NEUTRAL'] , fill_value = 0)

    # 5. compute trend: convert Time to date, sort chronologically,
    #    split into first half / second half, compute % positive in each
    product_df['date'] = pd.to_datetime(product_df['Time'] , unit = 's')
    product_df = product_df.sort_values('date')
    num_rows = product_df.shape[0]//2
    first_half = product_df[:num_rows]
    second_half = product_df[num_rows:]
    first_half_positive = first_half['predicted_label'].value_counts(normalize = True).reindex(['POSITIVE' , 'NEGATIVE'] , fill_value = 0)
    second_half_positive = second_half['predicted_label'].value_counts(normalize = True).reindex(['POSITIVE' , 'NEGATIVE'] , fill_value = 0)
    
    mismatches = product_df[
    ((product_df['predicted_label'] == "POSITIVE") & (product_df['Score']<=2)) |
    ((product_df['predicted_label'] == "NEGATIVE") & (product_df['Score']>=4))
    
]
    
    
    # 6. print a readable report using all of the above
    print(f"The rating percentage is: {rating_breakdown}\nThe predicted percentage is: {percentage}\nThe first half of the data percentage is: {first_half_positive}\nThe second half of the data percentage is:{second_half_positive}")
    
    # 7. return a dict with the key numbers (model %s, rating %s, trend %s)
    return {
        'data_rating' : rating_breakdown,
        'whole_predicted' : percentage,
        'first_half_predicted' :first_half_positive ,
        'second_half_predicted':second_half_positive,
        'mismatches' : mismatches,
    }

if __name__ == "__main__":
    try:
        generate_report(df, product)
    except ValueError as e:
        print(e)
