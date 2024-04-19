# Large Language Models (LLMs)

This repository is dedicated to the exploration and use of Large Language Models (LLMs) such as GPT (Generative Pre-trained Transformer), BERT (Bidirectional Encoder Representations from Transformers), and others. These models have significantly impacted natural language processing by providing state-of-the-art results in tasks like text completion, translation, and question answering.

## Prerequisites

- Python 3.6 or higher
- Libraries: transformers, torch

## Installation

To work with large language models, you need to install the Hugging Face Transformers library, which provides a straightforward API to use these models:

```bash
pip install transformers torch
```

## Example - Using GPT-2 for Text Generation

This example demonstrates how to use the GPT-2 model from the Transformers library to generate text.

### `gpt2_text_generation.py`

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    encoded_input = tokenizer.encode(prompt, return_tensors='pt')
    output_sequences = model.generate(
        encoded_input,
        max_length=max_length,
        temperature=1.0,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.2
    )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(generated_text)

# Example usage
prompt = "The future of AI in natural language processing is"
generate_text(prompt)

```

## Contributing

We welcome contributions that improve the repository's examples, documentation, or introduce new functionalities related to LLMs. Please fork this repository, commit your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
