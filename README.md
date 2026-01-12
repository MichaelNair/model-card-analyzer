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
   # create virtual environment using uv
   uv venv

   # activate it
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

For prompts that are part of the tool, in the `prompt` folder and in the `llm_judge_eval.py` file, I tried to always prioritize giving the LLM as much rich and accurate context as I can in accordance with context engineering principles. I put the document, text or image, after the system message and leave the request for last. I also include formatting examples for the output and structure the prompts using XML tags for clarity (both for my own understanding and the LLM's).

For prompts related to building the tool, I focus on encouraging brevity and simplicity and avoid having the LLM too involved in the architecture. I used Cursor running Sonnet 4.5 and GPT 5.2 for code assist and scaffold.

## Tool Evaluation

I included an automatic LLM-as-a-judge feature in the tool, which prints two evaluation metrics: accuracy and completeness. These are good analogs for recall and precision, which are the two statistics I have the most experience with. I also read through the documents and the results myself to form my own opinions of the tool’s overall effectiveness. LLMs are very good at generating broad summaries of long technical documents. Based on my analysis these summaries are accurate, complete, and useful.

## Pre and Post Processing of Data

For PDF documents, I converted them to images using the pdf2image and base64 Python libraries and included them as image context in LLM calls. For textual data, I included them directly in my LLM calls along with appropriate context. I verify that the LLM output is in JSON format before saving it to a results file.

This is the JSON schema that I used for the output:
```json
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "metadata": {
      "type": "object",
      "properties": {
        "model_card_file": {
          "type": "string"
        },
        "analysis_timestamp": {
          "type": "string"
        },
        "llm_model": {
          "type": "string"
        },
        "tokens_used": {
          "type": "object",
          "properties": {
            "prompt_tokens": {
              "type": "integer"
            },
            "completion_tokens": {
              "type": "integer"
            },
            "total_tokens": {
              "type": "integer"
            }
          },
          "required": [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens"
          ]
        }
      },
      "required": [
        "model_card_file",
        "analysis_timestamp",
        "llm_model",
        "tokens_used"
      ]
    },
    "analysis_results": {
      "type": "object",
      "properties": {
        "model_name": {
          "type": "string"
        },
        "training_data": {
          "type": "string"
        },
        "model_architecture": {
          "type": "string"
        },
        "limitations": {
          "type": "string"
        }
      },
      "required": [
        "model_name",
        "training_data",
        "model_architecture",
        "limitations"
      ]
    }
  },
  "required": [
    "metadata",
    "analysis_results"
  ]
}
```

## Possible Improvements

1. Most of the improvements I wish I had time to include are based around the evaluation step. There are a lot of LLM-as-a-judge techniques and tools I haven't implemented yet that may prove very useful for this relatively simple LLM task.

2. I also have not evaluated different models very thoroughly. I doubt any flagship models would struggle with this task too much, but it is still worth double checking to see if there are obvious improvements with Sonnet 4.5, Gemini 3, or any of the comparatively cheaper models.

3. With more time and context for the user, I also could have included more than just 3 specialized summaries. I could have extracted specific entities like "date published" or "approximate number of parameters" that might be useful on their own.

4. The biggest potential improvement would probably be asking a subject matter expert to red-team some of the summaries. This would be to evaluate accuracy and completeness, but also to compare broader design decisions and whether or not these summaries meet the needs of the client.

5. A possible next step would be to convert the extractions and summaries from JSON into md or another more readable format, md, pdf, etc.
