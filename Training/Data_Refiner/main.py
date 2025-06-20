import google.generativeai as genai
import pandas as pd
import json
import time
from typing import List, Dict, Any
import re

class RomanSentenceRefiner:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-lite"):
        """
        Initialize the Roman Sentence Refiner with Gemini API
        
        Args:
            api_key (str): Your Google Gemini API key
            model_name (str): The Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.generation_config = {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1000,
        }
    
    def refine_sentences_batch(self, sentences: List[str]) -> List[str]:
        """
        Send a batch of Roman sentences to Gemini for refinement
        
        Args:
            sentences (List[str]): List of Roman sentences to refine
            
        Returns:
            List[str]: List of refined sentences
        """
        # Create the prompt for batch processing
        sentences_text = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])
        
        prompt = f"""You are an expert in Roman Urdu (Urdu language written in Latin script). Please refine the following Roman Urdu sentences according to these rules:

1. Correct any spelling errors in Roman Urdu transliteration
2. Fix grammatical mistakes while keeping the sentences in Roman Urdu
3. If a sentence is too short (less than 5 words), expand it to make it more complete and meaningful while maintaining the original meaning in Roman Urdu
4. Ensure proper Roman Urdu grammar and natural flow
5. Don't use any quotations around the sentences!(important)
6. Keep all sentences in Roman Urdu script (Latin alphabet), DO NOT translate to any other language
7. Maintain the conversational and natural tone of Urdu language
8. Return ONLY the refined Roman Urdu sentences in the same order, numbered 1-{len(sentences)}

IMPORTANT: Keep everything in Roman Urdu (Latin script). Do not translate to Arabic, English, or any other language And there shouldn't be less than 5 words in a sentence.

Here are the Roman Urdu sentences to refine:
{sentences_text}

Return the refined sentences in this exact format:
1. [refined Roman Urdu sentence atleast 5 words]
2. [refined Roman Urdu sentence atleast 5 words]
3. [refined Roman Urdu sentence atleast 5 words]
...and so on.

Do not include any explanations or additional text, just the numbered refined Roman Urdu sentences."""


        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Parse the response to extract sentences
            refined_sentences = self._parse_response(response.text, len(sentences))
            return refined_sentences
            
        except Exception as e:
            print(f"Error in batch refinement: {str(e)}")
            return sentences  # Return original sentences if error occurs
    
    def _parse_response(self, response_text: str, expected_count: int) -> List[str]:
        """
        Parse the Gemini response to extract refined sentences
        
        Args:
            response_text (str): Raw response from Gemini
            expected_count (int): Expected number of sentences
            
        Returns:
            List[str]: Parsed refined sentences
        """
        lines = response_text.strip().split('\n')
        refined_sentences = []
        
        for line in lines:
            line = line.strip()
            if line and re.match(r'^\d+\.', line):
                # Remove the number and period from the beginning
                sentence = re.sub(r'^\d+\.\s*', '', line).strip()
                if sentence:
                    refined_sentences.append(sentence)
        
        # If we didn't get the expected number of sentences, pad with empty strings
        while len(refined_sentences) < expected_count:
            refined_sentences.append("")
        
        return refined_sentences[:expected_count]
    
    def check_and_fix_sentence(self, sentence: str) -> str:
        """
        Check if a single sentence is correct and fix it if needed
        
        Args:
            sentence (str): Single Roman sentence to check
            
        Returns:
            str: Corrected sentence or original if already correct
        """
        prompt = f"""You are an expert in Roman Urdu (Urdu language written in Latin script). Please analyze this Roman Urdu sentence:
"{sentence}"
If the sentence is grammatically correct, properly spelled in Roman Urdu, and of adequate length (at least 5 words), respond with exactly: "CORRECT: {sentence}"

If the sentence needs correction (spelling errors in Roman Urdu, grammar issues, or too short), provide the corrected version with exactly: "FIXED: [corrected Roman Urdu sentence]"

IMPORTANT: 
- Keep everything in Roman Urdu (Latin script)
- Don't use any single or double quotations around the sentence!
- Do not translate to Arabic, English, or any other language
- Only fix Roman Urdu spelling and grammar issues
- Maintain the natural Urdu conversational tone

Sentence to analyze: {sentence}"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            response_text = response.text.strip()
            
            if response_text.startswith("CORRECT:"):
                return sentence  # Return original sentence
            elif response_text.startswith("FIXED:"):
                fixed_sentence = response_text.replace("FIXED:", "").strip()
                return fixed_sentence
            else:
                # If response format is unexpected, try to extract the sentence
                return response_text.strip()
                
        except Exception as e:
            print(f"Error checking sentence: {str(e)}")
            return sentence  # Return original sentence if error occurs

def process_roman_dataset(df: pd.DataFrame, sentence_column: str, api_key: str, 
                         batch_size: int = 5, delay: float = 1.0) -> List[str]:
    """
    Process an entire dataframe of Roman sentences
    
    Args:
        df (pd.DataFrame): DataFrame containing Roman sentences
        sentence_column (str): Name of the column containing sentences
        api_key (str): Google Gemini API key
        batch_size (int): Number of sentences to process in each batch
        delay (float): Delay between API calls to avoid rate limiting
        
    Returns:
        List[str]: List of all refined sentences
    """
    refiner = RomanSentenceRefiner(api_key)
    refined_sentences = []
    
    # Get all sentences from the dataframe
    sentences = df[sentence_column].dropna().tolist()
    total_sentences = len(sentences)
    
    print(f"Processing {total_sentences} sentences...")
    
    # Process in batches
    for i in range(0, total_sentences, batch_size):
        batch_sentences = sentences[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_sentences + batch_size - 1)//batch_size}")
        
        # Process the batch
        refined_batch = refiner.refine_sentences_batch(batch_sentences)
        refined_sentences.extend(refined_batch)
        
        # Add delay to avoid rate limiting
        if i + batch_size < total_sentences:
            time.sleep(delay)
    
    return refined_sentences

def validate_and_fix_sentences(sentences: List[str], api_key: str, 
                             delay: float = 0.5) -> List[str]:
    """
    Validate and fix individual sentences in a loop
    
    Args:
        sentences (List[str]): List of sentences to validate
        api_key (str): Google Gemini API key
        delay (float): Delay between API calls
        
    Returns:
        List[str]: List of validated and corrected sentences
    """
    refiner = RomanSentenceRefiner(api_key)
    final_sentences = []
    
    print(f"Validating {len(sentences)} sentences...")
    
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # Skip empty sentences
            print(f"Validating sentence {i+1}/{len(sentences)}")
            corrected_sentence = refiner.check_and_fix_sentence(sentence)
            final_sentences.append(corrected_sentence)
            
            # Add delay to avoid rate limiting
            time.sleep(delay)
        else:
            final_sentences.append(sentence)
    
    return final_sentences

# Example usage and main function
def main():
    # Configuration
    API_KEY = "AIzaSyD8qHygtmEhG84PmnxbfZJmYOTTp3YU7gs"  # Replace with your actual API key
    
    # Example: Create a sample dataframe (replace with your actual data)
    temp = {
        'normalized': [
            'ye kya hai ?',
            'wo kya hai ?',
            'idhar aa .',
            'udhar ja .',
            'ye mera ghar hai .',
            'wo mera dost hai .',
            'kya aap thak gaye hain ?',
            'mein thak gaya hoon .',
            'mujhy neend aa rahi hai .',
            'subah ho gayi hai .',
            'raat ho gayi hai .',
            'aaj garmi hai .',
            'aaj sardi hai .',
            'barish ho rahi hai .',
            'hawa chal rahi hai .',
            'kya aap ko bhook lagi hai ?',
            'kya aap ko pyas lagi hai ?',
            'mujhy pyas lagi hai .',
            'pani do .',
            'khana ready hai .',
            'chalo khana khate hain .',
            'maza aaya .'
        ]
    }
    df=pd.read_csv('./Final_normalized.csv')
    # df=pd.DataFrame(temp)
    try:
        # Step 1: Process the entire dataset in batches
        print("=== Step 1: Batch Processing ===")
        refined_sentences = process_roman_dataset(
            df=df,
            sentence_column='normalized',
            api_key=API_KEY,
            batch_size=200,
            delay=2.0
        )
        
        print(f"\nRefined sentences: {len(refined_sentences)}")
        for i, sentence in enumerate(refined_sentences):
            print(f"{i+1}. {sentence}")
        
        # # Step 2: Validate and fix individual sentences
        # print("\n=== Step 2: Individual Validation ===")
        # final_sentences = validate_and_fix_sentences(
        #     sentences=refined_sentences,
        #     api_key=API_KEY,
        #     delay=2.05
        # )
        final_sentences=refined_sentences
        print(f"\nFinal sentences: {len(final_sentences)}")
        for i, sentence in enumerate(final_sentences):
            print(f"{i+1}. {sentence}")
        
        # Save to file
        output_df = pd.DataFrame({'refined_roman_sentences': final_sentences})
        output_df.to_csv('refined_roman_dataset.csv', index=False)
        print("\nDataset saved to 'refined_roman_dataset.csv'")
        
        return final_sentences
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        return []

if __name__ == "__main__":
    # Run the main function
    final_dataset = main()
    
    # Print summary
    print(f"\nProcess completed. Generated {len(final_dataset)} refined sentences.")