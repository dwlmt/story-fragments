from jsonlines import jsonlines


def build_salience_override_dict(salience_override_json, salience_score_filter: float = 0.35):
    salience_dict = {}  # Is a lookup for the salience mapping if overriding.
    if salience_override_json is not None:
        chapter_count = 0
        title_count = 0
        summary_sentence_count = 0
        alignment_count = 0
        sentence_count = 0
        with jsonlines.open(salience_override_json) as reader:
            for obj in reader:
                title = obj["title"]

                title_count += 1

                if title not in salience_dict:
                    salience_dict[title] = {}

                for chapter in obj["chapters"]:

                    chapter_count += 1

                    summary = chapter["summary"]
                    sentences = summary["sentences"]

                    for sentence in sentences:

                        summary_sentence_count += 1

                        # print(f"Sentence: {sentence}")

                        if "alignments" in sentence:
                            alignments = sentence["alignments"]
                            for alignment in alignments:
                                if alignment["similarity"] > salience_score_filter:
                                    alignment_count += 1

                                    salience_dict[title][alignment["text"]] = {**alignment,
                                                                               **{"summary_text": sentence["text"],
                                                                                  "summary_seq_num": sentence[
                                                                                      "seq_num"]}}

                    if "full_text" in chapter:
                        sentence_count += len(chapter["full_text"]["sentences"])
                print(f"Running stats : {title} - Title Count: {title_count}, Chapter Count: {chapter_count}, "
                      f"Alignment Count: {alignment_count}, Sentence Count: {sentence_count} ")
    return salience_dict
