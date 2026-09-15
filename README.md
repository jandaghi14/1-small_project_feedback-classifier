A lightweight customer-feedback analysis tool built on Hugging Face Transformers. 
Given a product ID from the Amazon Fine Food Reviews dataset, it runs that product's reviews through a pretrained sentiment classifier 
(no training required) and prints a business-facing report: 
the positive/negative split, 
how that compares to the raw star rating, 
and whether sentiment is trending up or down over time. 
Built to demonstrate applying an existing NLP model to a real business question rather than training one from scratch.