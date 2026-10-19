"""Extra questions I did NOT tune the pipeline on.

HELDOUT_DATASET: answerable, written from passages in the WTTC report.
UNANSWERABLE: questions the report cannot answer. The system should refuse them.
Anchors are short strings that sit inside a single chunk, so gold chunks are easy to find.
"""

HELDOUT_DATASET = [
    {
        "id": "h01_deep_blue",
        "question": "When did IBM Deep Blue beat chess grandmaster Gary Kasparov?",
        "expected_answer": "In 1997.",
        "anchors": ["Kasparov"],
        "required_facts": ["1997"],
    },
    {
        "id": "h02_godfathers",
        "question": "Who are the researchers nicknamed the 'Godfathers of AI'?",
        "expected_answer": "Yann LeCun, Geoffrey Hinton and Yoshua Bengio.",
        "anchors": ["Godfathers of"],
        "required_facts": ["LeCun", "Hinton", "Bengio"],
    },
    {
        "id": "h03_turing_paper",
        "question": "Which 1950 paper by Alan Turing introduced the Turing Test?",
        "expected_answer": "'Computing Machinery and Intelligence'.",
        "anchors": ["Computing Machinery and Intelligence"],
        "required_facts": ["Computing Machinery and Intelligence"],
    },
    {
        "id": "h04_moores_law",
        "question": "What did Gordon Moore observe about transistors on an integrated circuit?",
        "expected_answer": "The number of transistors was doubling around every two years (Moore's Law).",
        "anchors": ["Gordon Moore"],
        "required_facts": ["transistors", "two years"],
    },
    {
        "id": "h05_google_tpu",
        "question": "What is the name of the AI chip Google has developed since 2016?",
        "expected_answer": "The TPU (Tensor Processing Unit).",
        "anchors": ["Tensor Processing Unit"],
        "required_facts": ["TPU", "Tensor Processing Unit"],
    },
    {
        "id": "h06_webpages",
        "question": "How many webpages were online in 2022, and how long would a person need to study them all?",
        "expected_answer": "About 5.75 billion webpages; roughly 11,000 years at one minute per page.",
        "anchors": ["5.75 billion"],
        "required_facts": ["5.75 billion", "11,000"],
    },
    {
        "id": "h07_randstad",
        "question": "What share of workers had been offered AI training according to the Randstad survey?",
        "expected_answer": "Only 13% of workers.",
        "anchors": ["7,000 employees"],
        "required_facts": ["13%"],
    },
    {
        "id": "h08_youtube_views",
        "question": "How many views did videos about AI tools get on YouTube in 2023?",
        "expected_answer": "More than 17 billion views.",
        "anchors": ["17 billion"],
        "required_facts": ["17 billion"],
    },
    {
        "id": "h09_uk_exascale",
        "question": "How much is the UK investing in an exascale computer for AI research?",
        "expected_answer": "£900 million.",
        "anchors": ["900 million"],
        "required_facts": ["900"],
    },
    {
        "id": "h10_new_data_units",
        "question": "Which two new units of data measurement were agreed in 2022?",
        "expected_answer": "The Ronnabyte and the Quettabyte.",
        "anchors": ["Ronnabyte"],
        "required_facts": ["Ronnabyte", "Quettabyte"],
    },
]

UNANSWERABLE = [
    {"id": "u01_gpt4_params", "question": "How many parameters does GPT-4 have?"},
    {"id": "u02_bitcoin", "question": "What was the price of Bitcoin in 2023?"},
    {"id": "u03_capital", "question": "What is the capital city of Australia?"},
    {"id": "u04_llama", "question": "How many parameters does the Llama model have?"},
    {"id": "u05_openai_revenue", "question": "What was OpenAI's revenue in 2023?"},
]