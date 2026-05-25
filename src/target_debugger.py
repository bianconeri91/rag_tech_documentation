def debug_target(chunks, target, section=None):
    for idx, chunk in enumerate(chunks):
        text = chunk["chunk_text"]

        if section and chunk["section"] != section:
            continue

        if target.lower() in text.lower():
            print("=" * 80)
            print(idx)
            print(chunk["section"])
            print(text)