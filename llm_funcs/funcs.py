import torch


def read_and_split_text(filename):
    """
    Function to read a text file and return paragraphs one by one.
    """

    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    # Split the text into paragraphs (simple split by newline characters)
    paragraphs = text.split("\n")

    # Filter out any empty paragraphs or undesired entries
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]
    return paragraphs


def encode_contexts(text_list, context_tokenizer, context_encoder, max_len):
    """
    Function to tokenize and create contextual embeddings of passages to be used in RAG.
    """

    # Encode a list of texts into embeddings
    embeddings = []
    for text in text_list:
        inputs = context_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=max_len
        )
        outputs = context_encoder(**inputs)
        embeddings.append(outputs.pooler_output)
    return torch.cat(embeddings).detach().numpy()


def search_relevant_contexts(
    question, question_tokenizer, question_encoder, index, k=5
):
    """
    Searches for the most relevant contexts to a given question, after teokenizing and embeeding it.

    Returns:
    tuple: Distances and indices of the top k relevant contexts.
    """
    # Tokenize the question
    question_inputs = question_tokenizer(question, return_tensors="pt")

    # Encode the question to get the embedding
    question_embedding = (
        question_encoder(**question_inputs).pooler_output.detach().numpy()
    )

    # Search the index to retrieve top k relevant contexts
    D, I = index.search(question_embedding, k)

    return D, I


def generate_answer(
    tokenizer,
    model,
    question,
    contexts,
    max_len,
    min_len,
    length_penalty,
    num_beans,
):
    """
    Generates an answer based on a given question and contexts.
    """

    # Concatenate the retrieved contexts to form the input to GPT2
    input_text = question + " " + " ".join(contexts)
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=1024, truncation=True
    )

    # Generate output using GPT2
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_len,
        min_length=min_len,
        length_penalty=length_penalty,
        num_beams=num_beans,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
