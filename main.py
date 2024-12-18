import random

from datasets import load_dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


def llama(question):
    try:
        template = """Question: {question}
        Answer: Think step by step to solve the problem. After solving the problem check your answers and try to proof them.
        At the end Only return '####' + the numeric solution, no currencies, no units, and no words, only the result number."""
        prompt_template = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model="llama3.3")
        chain = prompt_template | model
        response = chain.invoke({"question": question})
        return response.strip()

    except Exception as e:
        print(f"Error calling LLAMA: {e}")
        return None


def extract_answer(answer):
    try:
        return answer.split("####")[-1].strip()
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return None


def evaluate_llama_accuracy():
    try:
        dataset = load_dataset("gsm8k", 'main', split="test")
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        return
    counter = 0

    total_questions = len(dataset)

    # you can add your preferred sample size here
    sample_size = 20
    samples = random.sample(range(total_questions), sample_size)
    correct_predictions = 0

    print(f"Total Questions in the dataset: {total_questions}")
    print(f"Using subset of {sample_size} questions.")

    for i in samples:
        counter += 1
        instance = dataset[i]
        question = instance["question"]
        correct_answer = extract_answer(instance["answer"])

        if not correct_answer:
            print(f"Skipping question {i}, answer doesn't follow template.")
            continue

        llm_result = llama(question)

        if not llm_result:
            print(f"Skipping question {i}, no response from LLAMA.")
            continue
        llm_result_stripped = extract_answer(llm_result)

        try:
            llm_result_stripped = int(llm_result_stripped)
            correct_answer = int(correct_answer.strip())
            if llm_result_stripped == correct_answer:
                correct_predictions += 1
        except Exception:
            print(f"Skipping question {i}, couldn't extract numerical result from answer")
            continue

        print(f"Question {counter}/{sample_size}")
        print(f"Question: {question}")
        print(f"Correct Answer: {correct_answer}")
        print(f"LLAMA's Answer: {llm_result_stripped}")
        print(f"{'Correct' if llm_result_stripped == correct_answer else 'Incorrect'}")
        print(f"total correct answers: {correct_predictions}")
        print(f"total wrong answers: {counter - correct_predictions}")
        print("=" * 50)

    accuracy = (correct_predictions / sample_size) * 100
    print(f"Final Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_llama_accuracy()
