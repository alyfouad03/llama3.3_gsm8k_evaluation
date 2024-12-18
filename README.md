## LLAMA Evaluation on GSM8K
This repo evaluates the accuracy of the LLAMA LLM answering math questions provided by the [GSM8K](https://huggingface.co/datasets/openai/gsm8k) dataset found on Hugging Face.

The dataset provides the school math problems and their answers and is split into a training and testing subset.

### Requirements

- Python 3.9 or later
- The packages listed in 'requirements.txt' -- can be installed using: 'pip install -r requirements.txt'
- Ollama server, can be downloaded here: https://ollama.com/
- Before running the code you need to pull 'llama3.3' using: 'ollama pull llama3.3'
- Ensure the Ollama server is running before executing the script: 'ollama serve'

### How it works
1. Loads the GSM8K dataset
2. Samples a subset of questions (default: 20)
3. Queries the Llama model to answer the questions
4. Compares the model's answers to correct dataset answers
5. Calculates the model's accuracy

### Usage
- Specify subset size you want to test
- Then run the code using: 'python main.py' or the run button in your IDE

Disclaimer: Depending on your device's capabilities the code may take a while to finish execution!

