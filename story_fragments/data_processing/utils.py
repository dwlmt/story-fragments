from jsonlines import jsonlines


def build_salience_override_dict(salience_override_json):
    salience_dict = {}  # Is a lookup for the salience mapping if overriding.
    if salience_override_json is not None:
        with jsonlines.open(salience_override_json) as reader:
            for obj in reader:
                title = obj["title"]

                if title not in salience_dict:
                    salience_dict[title] = {}

                for chapter in obj["chapters"]:
                    summary = chapter["summary"]
                    sentences = summary["sentences"]

                    for sentence in sentences:

                        # print(f"Sentence: {sentence}")

                        if "alignments" in sentence:
                            alignments = sentence["alignments"]
                            for alignment in alignments:
                                salience_dict[title][alignment["text"]] = {**alignment,
                                                                           **{"summary_text": sentence["text"],
                                                                              "summary_seq_num": sentence[
                                                                                  "seq_num"]}}
    return salience_dict