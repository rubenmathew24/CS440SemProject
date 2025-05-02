from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import argparse
import config
import utils

DEVICE = config.DEVICE
MAX_LENGTH = config.MAX_LENGTH
TOP_K = config.TOP_K

IS_BINARY = None
SIZE = ""
LEARNING_TYPE = ""

def predict_review(model, tokenizer, text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        if not IS_BINARY:
            probs = torch.softmax(logits, dim=1)
            topk_values, topk_indices = torch.topk(probs, TOP_K, dim=1)

    if not IS_BINARY:
        top_k_classes = topk_indices.squeeze().tolist()
        top_k_probabilities = topk_values.squeeze().tolist()
        return predicted_class, top_k_classes, top_k_probabilities
    
    return predicted_class


def test_loop(id2label):
    label_type = 'binary' if IS_BINARY else 'multi'
    model_path = f'models/{LEARNING_TYPE}/{SIZE}/{label_type}_{SIZE}'

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.to(DEVICE)

    while True:
        user_input = input("Enter a review (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        if IS_BINARY:
            label = predict_review(model, tokenizer, user_input)
            print(f"Predicted label: {id2label[label]}")
        else:
            label, top_k_classes, top_k_probabilities = predict_review(model, tokenizer, user_input)
            print(f"Predicted label: {id2label[label]}")

            print(f"Top-{TOP_K} Results:")
            
            for i in range(TOP_K):
                print(f"\t{i+1}. {100*top_k_probabilities[i]:.4f}% {id2label[top_k_classes[i]]}")


def main():
    global IS_BINARY, SIZE, LEARNING_TYPE
    parser = argparse.ArgumentParser(description="Process type and size arguments.")

    all_sizes = ["small", "medium", "large"]
    all_types = ["binary", "multi"]
    all_learnings = ['baseline', 'AL1', 'AL2']

    parser.add_argument('--type', choices=all_types, required=True, help='Specify type')
    parser.add_argument('--size', choices=all_sizes, required=True, help='Specify size')
    parser.add_argument('--learning', choices=all_learnings, required=True, help='Specify learning type')
    args = parser.parse_args()

    # Setup Globals
    IS_BINARY = True if args.type == 'binary' else False
    SIZE = args.size
    LEARNING_TYPE = args.learning

    # Get id2label
    if IS_BINARY:
        id2label = {
            0: "Negative",
            1: "Positive"
        }
    else:
        categories = utils.get_all_categories()
        id2label = {i:category for i,category in enumerate(sorted(categories))}


    # Run Test loop
    test_loop(id2label)


if __name__ == "__main__":
    main()
