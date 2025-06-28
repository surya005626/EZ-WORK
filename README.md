# EZ-WORK
Building a Smart Assistant for Research Summarization
                    # Import necessary libraries
from transformers import pipeline
import torch

# Initialize the summarization model
def initialize_model():
    # Use a pre-trained model for summarization
    model_name = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return summarizer

# Summarize text
def summarize_text(summarizer, text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Main function
def main():
    summarizer = initialize_model()

    # Example text
    text = """
    The city of Paris is the capital and most populous city of France. It is situated on the Seine River in the north-central part of the country. 
    Paris has a rich history dating back to the 3rd century BC, when it was a Celtic settlement. The city has been the hub of French culture, 
    art, and politics for centuries. It is known for its stunning architecture, art museums, fashion, and cuisine. 
    The Eiffel Tower, built in the late 19th century, is one of the most iconic landmarks in the world and a symbol of Paris.
    """

    # Summarize the text
    summary = summarize_text(summarizer, text)

    # Print the original text and the summary
    print("Original Text:")
    print(text)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()
