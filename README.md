# Model Card analysis and summarization tool

## Set Up

1. **Create a `.env` file** and add your API credentials to it:
   ```bash
   # Add your API keys (e.g., OpenAI, Anthropic, etc.)
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   ```

2. **Activate the virtual environment** (optional):
   ```bash
   source .venv/bin/activate
   ```

3. **Run the analyzer** to get summary results:
   ```bash
   # Analyze a model card
   python main.py llama_3.md
   
   # Or with custom options
   python main.py llama_3.md --model gpt-5.2 --temperature 0.3
   
   # List available model cards
   python main.py --list
   ```

4. **Run the LLM judge evaluation** to get automatic summary evaluation (still worth reviewing manually):
   ```bash
   # Evaluate a summary against the original model card
   python llm_judge_eval.py model_cards/llama_3.md results/llama_3_20260112_095449.json
   
   # Or with a different evaluation model
   python llm_judge_eval.py model_cards/llama_3.md results/llama_3_20260112_095449.json --model gpt-5.2
   ```

## Prompting Techniques

For prompts that are part of the tool, in the `prompts` folder and in the `llm_judge_eval.py` file, I tried to always prioritize giving the LLM as much rich and accurate context as I can. I put the document, text or image, after the system message and leave the request for last. I also include formatting examples for the output and structure the prompts using XML tags for clarity (both for my own understanding and the LLM's).

For prompts related to building the tool, I focus on encouraging brevity and simplicity and avoid having the LLM too involved in the architecture.

## Tool Evaluation

I included an automatic LLM-as-a-judge feature in the tool which prints 2 evaluation metrics: accuracy and completeness. These are good analogs for recall and precision, which are the 2 statistics I have the most experience with. I also read through the documents and the results myself to form my own opinions of the tool's overall effectiveness. LLM are very good at generating broad summaries of long technical documents, I am not sure if these summaries in particular meet the needs of the client but as far as I can tell they are accurate, complete and useful.

## Pre and Post Processing of Data

For PDF documents, I converted them to images using the pdf2image Python library and included them as image context in LLM calls. For textual data, I included them directly in my LLM calls along with appropriate context. I verify that the LLM output is in JSON format before saving it to a results file.

## Possible Improvements

Most of the improvements I wish I had time to include are based around the evaluation step. There are a lot of LLM-as-a-judge techniques and tools I haven't implemented yet that may prove very useful for this relatively simple LLM task.

I also have not evaluated different models very thoroughly. I doubt any flagship models would struggle with this task too much, but it is still worth double checking to see if there are obvious improvements with Sonnet 4.5, Gemini 3, or any of the comparatively cheaper models.

With more time and context for the user, I also could have included more than just 3 specialized summaries. I could have extracted specific entities like "date published" or "approximate number of parameters" that might be useful on their own.

The biggest potential improvement would probably be asking a subject matter expert to red-team some of the summaries. This would be to evaluate accuracy and completeness, but also to compare broader design decisions and whether or not these summaries meet the needs of the client.
