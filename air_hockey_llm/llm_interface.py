from openai import OpenAI


class LLMInterface:
    def __init__(self, llm_model, port, output_dir, max_attempts=20):
        self.output_dir = f"{output_dir}/prompts"
        self.query_index = 0
        self.llm_model = llm_model
        self.port = port
        self.max_attempts = max_attempts

        self.api_key = "EMPTY"
        self.api_base = f"http://localhost:{self.port}/api"
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def query_model(self, chat_history, process_answer, **kwargs):
        success = False

        attempt_index = 0
        while not success and attempt_index < self.max_attempts:
            attempt_index += 1
            with open(f'prompt_{self.query_index}_{self.attempt_index}.txt', 'w') as fp:
                fp.write(chat_history[0]['content'])

            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=chat_history,
                **kwargs)
            answer = completion.choices[0].message.content
            
            with open(f'response_{self.query_index}_{self.attempt}.txt', 'w') as fr:
                fr.write(answer)

            try: 
                success, output = process_answer(answer)
            except Exception as e:
                pass
        
        self.query_index += 1
        return output
