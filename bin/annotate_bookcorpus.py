import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


def interpret_doc(doc):
    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "offset": token._.offset,
        }
        for token in doc
    ]


def is_valid_directory(path: Path | str, must_exist: bool = False):
    path = Path(path)
    if path.exists() and path.is_dir():
        return True
    if must_exist:
        return False
    else:
        return path.suffix == ""


def validate_and_create_dir(path: Path | str) -> Path:
    if not is_valid_directory(path):
        raise ValueError(
            f"The path f{path} does not point to a valid directory location"
        )

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    return path


def batched(iterable: Iterable, batch_size: int) -> Iterable:
    """Batch data from the `iterable` into tuples of length `batch_size`.
    The last batch may be shorter than `batch_size`.

    Parameters
    ----------
    `iterable : Iterable`
        The iterable that is to be batched.

    `batch_size : int`
        The size of every batch. Must be higher than zero.

    Returns
    -------
    `Iterable`
        The batched iterable.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    iterator = iter(iterable)
    while batch := islice(iterator, batch_size):
        yield batch


def run(args):
    import datasets
    import nltk
    import spacy
    from ewiser.spacy.disambiguate import Disambiguator

    out = validate_and_create_dir(args.output)

    nltk.download("wordnet")
    classifier = Disambiguator(
        args.checkpoint, "en", batch_size=args.batch_size, save_wsd_details=False
    ).eval()
    classifier.to(args.device)

    nlp = spacy.load(args.spacy, disable=["parser", "ner"])
    classifier.enable(nlp, "WSD")

    print("Loading dataset...")
    ds = datasets.load_from_disk(args.input).to_dict()

    print(
        f"\nBatch size of {args.batch_size}, saving every {args.save_every} iterations"
    )

    data = ds["text"][args.offset :]
    gen = tqdm(
        nlp.pipe(data, batch_size=args.batch_size), total=len(data), desc="Annotating"
    )

    for i, batch in enumerate(batched(gen, args.save_every)):
        output = [interpret_doc(doc) for doc in batch]
        with open(
            out
            / f"{args.offset + (i * args.save_every):08d}-{args.offset + ((i+1) * args.save_every) - 1:08d}.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(output, file, indent=2)

    print("Done! Shutdown...")
    import os

    os.system("shutdown -s -f -t 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="Where to load the dataset from"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="The directory to save the batches of annotations",
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Trained EWISER checkpoint.", required=True
    )
    parser.add_argument(
        "-d", "--device", default="cpu", help="Device to use. (cpu, cuda, cuda:0 etc.)"
    )
    parser.add_argument("-s", "--spacy", default="en_core_web_sm")
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument(
        "--offset", type=int, help="The start position for annotation", default=0
    )
    parser.add_argument("--save_every", type=int, default=80)
    args = parser.parse_args()
    run(args)
