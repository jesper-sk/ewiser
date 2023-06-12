import argparse
import json
from pathlib import Path

from tqdm import tqdm


def interpret_token(token):
    return {
        "text": token.text,
        "lemma": token.lemma_,
        "pos": token.pos_,
        "offset": token._.offset,
    }


def run(args):
    import nltk
    import spacy
    from ewiser.spacy.disambiguate import Disambiguator

    nltk.download("wordnet")
    classifier = Disambiguator(
        args.checkpoint, "en", batch_size=args.batch_size, save_wsd_details=False
    ).eval()
    classifier.to(args.device)

    nlp = spacy.load(args.spacy, disable=["parser", "ner"])
    classifier.enable(nlp, "WSD")

    with open(args.input, "r", encoding="utf-8") as file:
        data = file.readlines()

    output = [
        {"text": line, "tokens": [interpret_token(token) for token in nlp(line)]}
        for line in tqdm(data)
    ]

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Trained EWISER checkpoint.", required=True
    )
    parser.add_argument(
        "-d", "--device", default="cpu", help="Device to use. (cpu, cuda, cuda:0 etc.)"
    )
    parser.add_argument("-s", "--spacy", default="en_core_web_sm")
    parser.add_argument("-b", "--batch_size", default=5, type=int)
    args = parser.parse_args()
    run(args)
