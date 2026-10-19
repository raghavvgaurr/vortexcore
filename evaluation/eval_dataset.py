"""The original 10 evaluation questions, copied unchanged from the notebook."""

EVAL_DATASET = [
    {
        "id": "q01_dartmouth_ai",
        "question": 'Where and when was the term "Artificial Intelligence" first used?',
        "expected_answer": "The term Artificial Intelligence was first used at the Dartmouth Conference in 1956.",
        "anchors": ["Dartmouth Conference", "1956"],
        "required_facts": ["Dartmouth", "1956"],
    },
    {
        "id": "q02_chatgpt_100m",
        "question": "How fast did ChatGPT reach 100 million users?",
        "expected_answer": "ChatGPT reached 100 million users in about 2 months after its November 2022 launch.",
        "anchors": ["100 million users", "2 months", "November 2022"],
        "required_facts": ["100 million", "2 months", "November 2022"],
    },
    {
        "id": "q03_ai_components",
        "question": "What are the three core components of AI?",
        "expected_answer": "Algorithms, data, and compute/computing power.",
        "anchors": ["algorithms", "data", "compute"],
        "required_facts": ["algorithms", "data", "compute"],
    },
    {
        "id": "q04_tokens_words",
        "question": "What is the rough tokens-to-words rule?",
        "expected_answer": "One token is about 0.75 words, so 100 tokens is about 75 words.",
        "anchors": ["0.75 words", "100 tokens", "75 words"],
        "required_facts": ["0.75", "100 tokens", "75 words"],
    },
    {
        "id": "q05_transformers",
        "question": "Which paper introduced Transformers, and in what year?",
        "expected_answer": 'The paper "Attention Is All You Need" introduced Transformers in 2017.',
        "anchors": ["Attention Is All You Need", "2017"],
        "required_facts": ["Attention Is All You Need", "2017"],
    },
    {
        "id": "q06_christies_ai_art",
        "question": "How much did Christie's sell the first AI artwork for?",
        "expected_answer": "$432,500 (Edmond de Belamy, 2018).",
        "anchors": ["$432,500", "432,500", "Edmond de Belamy"],
        "required_facts": ["432,500"],
    },
    {
        "id": "q07_cisco_devices",
        "question": "How many networked devices does Cisco estimate, and how many are IoT?",
        "expected_answer": "Cisco estimates 29.3 billion networked devices, of which 14.7 billion are IoT.",
        "anchors": ["29.3 billion", "14.7 billion", "IoT"],
        "required_facts": ["29.3 billion", "14.7 billion", "IoT"],
    },
    {
        "id": "q08_tsmc_samsung",
        "question": "What market share do TSMC and Samsung have?",
        "expected_answer": "TSMC has about 55% and Samsung about 15%, about 70% combined.",
        "anchors": ["TSMC", "55%", "Samsung", "15%"],
        "required_facts": ["TSMC", "55", "Samsung", "15"],
    },
    {
        "id": "q09_swedish_breast_cancer",
        "question": "What did the Swedish breast cancer screening study find?",
        "expected_answer": "AI-supported screening was as good as two radiologists, cut workload by 50%, and led to 20% more cancers being detected early.",
        "anchors": ["as good as two radiologists", "50%", "20% more cancers"],
        "required_facts": ["two radiologists", "50%", "20%"],
    },
    {
        "id": "q10_itu_offline",
        "question": "What do the ITU figures say about the offline population?",
        "expected_answer": "34% had never connected (2.6 billion people) in 2022; in least-developed countries, 64% are offline.",
        "anchors": ["34%", "2.6 billion", "64%", "least developed countries"],
        "required_facts": ["34%", "2.6 billion", "64%"],
    },
]