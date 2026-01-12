"""
Model Card Analyzer

A tool to analyze machine learning model cards by extracting key information
using LLM-powered analysis. Supports both PDF and Markdown model cards.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from pypdf import PdfReader
from litellm_driver import LiteLLMDriver


class ModelCardAnalyzer:
    """Main analyzer class for processing model cards."""
    
    def __init__(self, model_name: str = "gpt-5.2", temperature: float = 0.3):
        """
        Initialize the analyzer.
        
        Args:
            model_name: The LLM model to use for analysis
            temperature: Temperature setting for the LLM
        """
        self.driver = LiteLLMDriver()
        self.model_name = model_name
        self.temperature = temperature
        self.project_root = Path(__file__).parent
        
    def load_model_card(self, filename: str) -> str:
        """
        Load a model card file from the model_cards folder.
        
        Args:
            filename: Name of the file in the model_cards folder
            
        Returns:
            The text content of the model card
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        model_cards_dir = self.project_root / "model_cards"
        file_path = model_cards_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model card not found: {file_path}")
        
        # Determine file type and extract text accordingly
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._extract_pdf_text(file_path)
        elif suffix in ['.md', '.txt']:
            return self._read_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdf, .md, .txt")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                raise ValueError(f"No text could be extracted from PDF: {file_path}")
            
            return full_text
            
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read a text or markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading text file: {e}")
    
    def load_prompt(self, filename: str) -> str:
        """
        Load a prompt file from the prompt folder.
        
        Args:
            filename: Name of the file in the prompt folder
            
        Returns:
            The prompt text
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        prompt_dir = self.project_root / "prompt"
        file_path = prompt_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Error reading prompt file: {e}")
    
    def analyze_model_card(self, model_card_text: str, prompt_text: str) -> Dict[str, Any]:
        """
        Send the model card and prompt to the LLM for analysis.
        
        Args:
            model_card_text: The text content of the model card
            prompt_text: The analysis prompt
            
        Returns:
            Dictionary containing the LLM response and metadata
        """
        # Combine prompt with model card
        full_query = f"{prompt_text}\n\n<model_card>\n{model_card_text}\n</model_card>"
        
        # Query the LLM
        print(f"Analyzing model card with {self.model_name}...")
        response = self.driver.query(
            model_name=self.model_name,
            query_text=full_query,
            temperature=self.temperature,
            additional_args={"max_tokens": 4096}
        )
        
        return response
    
    def validate_json_response(self, response_content: str) -> Dict[str, Any]:
        """
        Validate that the response is valid JSON.
        
        Args:
            response_content: The response text from the LLM
            
        Returns:
            Parsed JSON as a dictionary
            
        Raises:
            ValueError: If the response is not valid JSON
        """
        try:
            # Try to parse the JSON
            parsed_json = json.loads(response_content)
            return parsed_json
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks if present
            if "```json" in response_content:
                try:
                    start = response_content.find("```json") + 7
                    end = response_content.find("```", start)
                    json_content = response_content[start:end].strip()
                    parsed_json = json.loads(json_content)
                    return parsed_json
                except:
                    pass
            
            # If still fails, raise an error
            raise ValueError(f"Response is not valid JSON: {e}\n\nResponse content:\n{response_content}")
    
    def save_results(self, 
                     results: Dict[str, Any], 
                     model_card_filename: str,
                     response_metadata: Dict[str, Any]) -> Path:
        """
        Save the analysis results to a JSON file with a unique timestamped filename.
        
        Args:
            results: The validated JSON results from the LLM
            model_card_filename: Original model card filename
            response_metadata: Metadata from the LLM response (tokens, model, etc.)
            
        Returns:
            Path to the saved results file
        """
        # Create results directory if it doesn't exist
        results_dir = self.project_root / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(model_card_filename).stem
        output_filename = f"{base_name}_{timestamp}.json"
        output_path = results_dir / output_filename
        
        # Prepare the complete output with metadata
        output_data = {
            "metadata": {
                "model_card_file": model_card_filename,
                "analysis_timestamp": datetime.now().isoformat(),
                "llm_model": response_metadata.get("model", "unknown"),
                "tokens_used": response_metadata.get("usage", {}),
            },
            "analysis_results": results
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {output_path}")
        return output_path
    
    def run(self, 
            model_card_filename: str, 
            prompt_filename: str = "extract_and_summarize.txt") -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            model_card_filename: Name of the model card file to analyze
            prompt_filename: Name of the prompt file to use (default: extract_and_summarize.txt)
            
        Returns:
            Dictionary containing the analysis results
        """
        try:
            # Step 1: Load model card
            print(f"Loading model card: {model_card_filename}")
            model_card_text = self.load_model_card(model_card_filename)
            print(f"✓ Loaded {len(model_card_text)} characters")
            
            # Step 2: Load prompt
            print(f"\nLoading prompt: {prompt_filename}")
            prompt_text = self.load_prompt(prompt_filename)
            print(f"✓ Loaded prompt")
            
            # Step 3: Analyze with LLM
            print(f"\nAnalyzing with LLM...")
            response = self.analyze_model_card(model_card_text, prompt_text)
            print(f"✓ Analysis complete ({response['usage']['total_tokens']} tokens used)")
            
            # Step 4: Validate JSON
            print(f"\nValidating JSON response...")
            validated_results = self.validate_json_response(response['content'])
            print(f"✓ Valid JSON received")
            
            # Step 5: Save results
            print(f"\nSaving results...")
            output_path = self.save_results(validated_results, model_card_filename, response)
            
            return {
                "success": True,
                "results": validated_results,
                "output_file": str(output_path),
                "metadata": response
            }
            
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze machine learning model cards using LLM-powered extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a markdown model card
  python main.py llama_3.md
  
  # Analyze a PDF model card
  python main.py nitrogen.pdf
  
  # Use a custom prompt and model
  python main.py llama_3.md --prompt custom_prompt.txt --model gpt-5.2
  
  # List available model cards
  python main.py --list
        """
    )
    
    parser.add_argument(
        "model_card",
        nargs="?",
        help="Name of the model card file in the model_cards/ folder"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        default="extract_and_summarize.txt",
        help="Name of the prompt file in the prompt/ folder (default: extract_and_summarize.txt)"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="gpt-5.2",
        help="LLM model to use for analysis (default: gpt-5.2)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.3,
        help="Temperature setting for the LLM (default: 0.3)"
    )
    
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available model cards"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        project_root = Path(__file__).parent
        model_cards_dir = project_root / "model_cards"
        if model_cards_dir.exists():
            files = sorted(model_cards_dir.glob("*"))
            print("\nAvailable model cards:")
            for f in files:
                if f.is_file():
                    print(f"  - {f.name}")
        else:
            print("No model_cards directory found")
        return
    
    # Require model_card argument if not listing
    if not args.model_card:
        parser.print_help()
        print("\nError: model_card argument is required (or use --list to see available files)")
        sys.exit(1)
    
    # Run the analyzer
    analyzer = ModelCardAnalyzer(
        model_name=args.model,
        temperature=args.temperature
    )
    
    result = analyzer.run(
        model_card_filename=args.model_card,
        prompt_filename=args.prompt
    )
    
    # Exit with appropriate code
    if result["success"]:
        print("\n✓ Analysis complete!")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
