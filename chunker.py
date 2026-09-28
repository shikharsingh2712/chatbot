def chunk_text(text, chunk_size=1000, chunk_overlap=250):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        piece = text[start:end]
        chunks.append(piece)
        start = start + chunk_size - chunk_overlap

    return chunks


if __name__ == "__main__":
    from pypdf import PdfReader

    reader = PdfReader("sample.pdf")
    full_text = ""
    for page in reader.pages:
        full_text = full_text + page.extract_text()

    result = chunk_text(full_text)
    print("Total characters:", len(full_text))
    print("Number of chunks:", len(result))
    print("--- First chunk ---")
    print(result[0])