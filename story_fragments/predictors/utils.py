import more_itertools
from blingfire import text_to_sentences


def input_to_passages(inputs, sentence_batch_size: int = 4, sentence_step_size:int = 4):

    labels = []

    #print(f"Inputs: {inputs}")
    if "sentences" in inputs and len(inputs["sentences"]) > 0:
        sentences = inputs["sentences"]
    elif "text" in inputs and len(inputs["text"]) > 0:
        sentences = text_to_sentences(inputs["text"]).split('\n')
    elif "passages" in inputs and len(inputs["passages"]) > 0:
        passages = []
        labels = []
        for p in inputs["passages"]:
            if isinstance(p, dict):
                passages.append(p["text"])

                if "label" in p:
                    labels.append(p["label"])
            else:
                passages.append(p)

        sentences = None
    else:
        raise ValueError("Input text or sentences must be provided.")
  
    if sentences is not None:
        sentences = list(more_itertools.windowed(sentences, n=sentence_batch_size, fillvalue=" ",
                                                step=sentence_step_size))
        passages = [{"id": f"{i}", "seq_num": i, "text": " ".join(s), "prompt": True} for i, s in enumerate(sentences)]
    else:
        passages = [{"id": f"{i}", "seq_num": i, "text": s, "prompt": True} for i, s in enumerate(passages)]

    if labels is not None and len(labels) > 0:
        if len(labels) < len(passages):
            labels += [" "] * (len(labels) - len(passages))
        for p, l in zip(passages,labels):
            p["label"] = l
            
    return passages