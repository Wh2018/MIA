"""Light text normalization for similarity scoring (ORM helper)."""


def str_delete_format(text: str) -> str:
    """Drop the leading 3 tokens and filler words of each ';'-separated segment.

    Original code: removes the "The seeker XXX" prefix so that the cosine
    similarity is computed on the content-bearing tail of each statement.
    """
    segments = text.split(";")
    out = []
    for seg in segments:
        words = seg.strip().split()
        pruned = words[3:]
        while pruned and pruned[0] in ("the", "to", "that"):
            pruned.pop(0)
        out.append(" ".join(pruned))
    return ";".join(out)
