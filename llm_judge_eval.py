"""
G-Eval: LLM-based evaluation for model card summaries.

Evaluates model card analysis results for accuracy and completeness.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from pdf2image import convert_from_path
from litellm_driver import LiteLLMDriver


class GEvaluator:
    """G-Eval evaluator for model card summaries."""
    
    def __init__(self, model_name: str = "gpt-5.2", temperature: float = 0.0):
        """
        Initialize the evaluator.
        
        Args:
            model_name: The LLM model to use for evaluation
            temperature: Temperature setting (lower for more consistent scoring)
        """
        self.driver = LiteLLMDriver()
        self.model_name = model_name
        self.temperature = temperature
        self.temp_image_paths = []  # Track temporary image files for cleanup
        
    def load_model_card(self, filepath: str) -> tuple:
        """
        Load the original model card.
        
        Returns:
            Tuple of (content, is_image, image_paths)
            - For text files: (text_content, False, None)
            - For PDFs: (None, True, [image_paths])
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Model card not found: {filepath}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            # Convert PDF to images
            images = convert_from_path(str(path))
            image_paths = []
            
            # Save each page as a temporary image
            for i, image in enumerate(images):
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_page_{i}.png', delete=False)
                image.save(temp_file.name, 'PNG')
                image_paths.append(temp_file.name)
                self.temp_image_paths.append(temp_file.name)
            
            return None, True, image_paths
        elif suffix in ['.md', '.txt']:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read(), False, None
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def cleanup_temp_files(self):
        """Clean up temporary image files."""
        for path in self.temp_image_paths:
            try:
                Path(path).unlink()
            except Exception:
                pass
        self.temp_image_paths = []
    
    def load_summary_results(self, filepath: str) -> dict:
        """Load the analysis results JSON."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get("analysis_results", data)
    
    def evaluate_accuracy(self, model_card_content, summary: dict, is_image: bool = False, image_paths: list = None) -> dict:
        """
        Evaluate the accuracy of the summary.
        
        Args:
            model_card_content: Original model card text (or None if image)
            summary: The generated summary
            is_image: Whether the model card is an image
            image_paths: List of image paths if PDF
            
        Returns:
            Dictionary with score and reasoning
        """
        if is_image and image_paths:
            # For PDFs as images - use vision model
            prompt = f"""You are an expert evaluator assessing the accuracy of a model card summary.

The original model card is provided as image(s). Please review it carefully.

Evaluation Criteria:
- Accuracy measures whether the information in the summary is factually correct and faithful to the original model card
- Information should not be fabricated, misrepresented, or distorted
- Numbers, dates, names, and technical details must match the source
- Score 1-5 where:
  1 = Major inaccuracies, most information is incorrect or fabricated
  2 = Several significant inaccuracies present
  3 = Some minor inaccuracies or misrepresentations
  4 = Mostly accurate with very minor issues
  5 = Completely accurate, all information is faithful to the source

Generated Summary:
{json.dumps(summary, indent=2)}

Please evaluate the accuracy of the summary against the model card image(s) and provide:
1. A score from 1 to 5
2. Brief reasoning for your score

Respond in JSON format:
{{
  "score": <1-5>,
  "reasoning": "<your explanation>"
}}"""
            
            # For multi-page PDFs, use the first image for now (simplification)
            # Could be enhanced to handle multiple images
            response = self.driver.query(
                model_name=self.model_name,
                query_text=prompt,
                temperature=self.temperature,
                image_path=image_paths[0],
                additional_args={"max_tokens": 500}
            )
        else:
            # For text files
            prompt = f"""You are an expert evaluator assessing the accuracy of a model card summary.

Evaluation Criteria:
- Accuracy measures whether the information in the summary is factually correct and faithful to the original model card
- Information should not be fabricated, misrepresented, or distorted
- Numbers, dates, names, and technical details must match the source
- Score 1-5 where:
  1 = Major inaccuracies, most information is incorrect or fabricated
  2 = Several significant inaccuracies present
  3 = Some minor inaccuracies or misrepresentations
  4 = Mostly accurate with very minor issues
  5 = Completely accurate, all information is faithful to the source

Original Model Card:
{model_card_content}

Generated Summary:
{json.dumps(summary, indent=2)}

Please evaluate the accuracy of the summary and provide:
1. A score from 1 to 5
2. Brief reasoning for your score

Respond in JSON format:
{{
  "score": <1-5>,
  "reasoning": "<your explanation>"
}}"""

            response = self.driver.query(
                model_name=self.model_name,
                query_text=prompt,
                temperature=self.temperature,
                additional_args={"max_tokens": 500}
            )
        
        return self._parse_eval_response(response['content'])
    
    def evaluate_completeness(self, model_card_content, summary: dict, is_image: bool = False, image_paths: list = None) -> dict:
        """
        Evaluate the completeness of the summary.
        
        Args:
            model_card_content: Original model card text (or None if image)
            summary: The generated summary
            is_image: Whether the model card is an image
            image_paths: List of image paths if PDF
            
        Returns:
            Dictionary with score and reasoning
        """
        if is_image and image_paths:
            # For PDFs as images - use vision model
            prompt = f"""You are an expert evaluator assessing the completeness of a model card summary.

The original model card is provided as image(s). Please review it carefully.

Evaluation Criteria:
- Completeness measures whether the summary captures all important information from the model card
- Key aspects should include: model purpose, architecture, training data, performance metrics, limitations, intended use, ethical considerations
- Score 1-5 where:
  1 = Severely incomplete, most critical information is missing
  2 = Many important details are omitted
  3 = Some key information is missing
  4 = Mostly complete with minor omissions
  5 = Comprehensive, captures all important information

Generated Summary:
{json.dumps(summary, indent=2)}

Please evaluate the completeness of the summary against the model card image(s) and provide:
1. A score from 1 to 5
2. Brief reasoning for your score

Respond in JSON format:
{{
  "score": <1-5>,
  "reasoning": "<your explanation>"
}}"""
            
            # For multi-page PDFs, use the first image for now (simplification)
            response = self.driver.query(
                model_name=self.model_name,
                query_text=prompt,
                temperature=self.temperature,
                image_path=image_paths[0],
                additional_args={"max_tokens": 500}
            )
        else:
            # For text files
            prompt = f"""You are an expert evaluator assessing the completeness of a model card summary.

Evaluation Criteria:
- Completeness measures whether the summary captures all important information from the model card
- Key aspects should include: model purpose, architecture, training data, performance metrics, limitations, intended use, ethical considerations
- Score 1-5 where:
  1 = Severely incomplete, most critical information is missing
  2 = Many important details are omitted
  3 = Some key information is missing
  4 = Mostly complete with minor omissions
  5 = Comprehensive, captures all important information

Original Model Card:
{model_card_content}

Generated Summary:
{json.dumps(summary, indent=2)}

Please evaluate the completeness of the summary and provide:
1. A score from 1 to 5
2. Brief reasoning for your score

Respond in JSON format:
{{
  "score": <1-5>,
  "reasoning": "<your explanation>"
}}"""

            response = self.driver.query(
                model_name=self.model_name,
                query_text=prompt,
                temperature=self.temperature,
                additional_args={"max_tokens": 500}
            )
        
        return self._parse_eval_response(response['content'])
    
    def _parse_eval_response(self, response_text: str) -> dict:
        """Parse the evaluation response JSON."""
        try:
            # Try direct JSON parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_content = response_text[start:end].strip()
                return json.loads(json_content)
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_content = response_text[start:end].strip()
                return json.loads(json_content)
            raise ValueError(f"Could not parse evaluation response: {response_text}")
    
    def evaluate(self, model_card_path: str, results_path: str) -> dict:
        """
        Run complete evaluation.
        
        Args:
            model_card_path: Path to the original model card
            results_path: Path to the analysis results JSON
            
        Returns:
            Dictionary containing evaluation scores and reasoning
        """
        try:
            print(f"Loading model card: {model_card_path}")
            model_card_content, is_image, image_paths = self.load_model_card(model_card_path)
            
            if is_image:
                print(f"✓ Loaded PDF as {len(image_paths)} image(s)")
            else:
                print(f"✓ Loaded {len(model_card_content)} characters")
            
            print(f"\nLoading summary results: {results_path}")
            summary = self.load_summary_results(results_path)
            print(f"✓ Loaded summary")
            
            print(f"\nEvaluating accuracy...")
            accuracy_eval = self.evaluate_accuracy(model_card_content, summary, is_image, image_paths)
            print(f"✓ Accuracy Score: {accuracy_eval['score']}/5")
            print(f"  Reasoning: {accuracy_eval['reasoning']}")
            
            print(f"\nEvaluating completeness...")
            completeness_eval = self.evaluate_completeness(model_card_content, summary, is_image, image_paths)
            print(f"✓ Completeness Score: {completeness_eval['score']}/5")
            print(f"  Reasoning: {completeness_eval['reasoning']}")
            
            results = {
                "accuracy": accuracy_eval,
                "completeness": completeness_eval,
                "average_score": (accuracy_eval['score'] + completeness_eval['score']) / 2
            }
            
            print(f"\n{'='*60}")
            print(f"OVERALL EVALUATION")
            print(f"{'='*60}")
            print(f"Accuracy:     {accuracy_eval['score']}/5")
            print(f"Completeness: {completeness_eval['score']}/5")
            print(f"Average:      {results['average_score']:.1f}/5")
            print(f"{'='*60}")
            
            return results
        finally:
            # Clean up temporary files
            self.cleanup_temp_files()


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="G-Eval: Evaluate model card summaries for accuracy and completeness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a summary
  python g_eval.py model_cards/llama_3.md results/llama_3_20260112_095449.json
  
  # Use a different model for evaluation
  python g_eval.py model_cards/nitrogen.pdf results/nitrogen_20260112_095559.json --model gpt-4
        """
    )
    
    parser.add_argument(
        "model_card",
        help="Path to the original model card file"
    )
    
    parser.add_argument(
        "results",
        help="Path to the analysis results JSON file"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="gpt-5.2",
        help="LLM model to use for evaluation (default: gpt-5.2)"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.0,
        help="Temperature setting for the LLM (default: 0.0 for consistent scoring)"
    )
    
    args = parser.parse_args()
    
    evaluator = None
    try:
        evaluator = GEvaluator(
            model_name=args.model,
            temperature=args.temperature
        )
        
        results = evaluator.evaluate(args.model_card, args.results)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Ensure cleanup happens even on error
        if evaluator:
            evaluator.cleanup_temp_files()


if __name__ == "__main__":
    main()
