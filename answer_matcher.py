"""
Simple fuzzy answer matching utility for Python backend
Uses difflib for fuzzy string matching as an alternative to Fuse.js
"""

import difflib
import re
from typing import Dict, Any, List, Optional


class AnswerMatcher:
    def __init__(self, threshold: float = 0.7):
        """
        Initialize the answer matcher
        
        Args:
            threshold: Similarity threshold (0.0 to 1.0, higher = more strict)
        """
        self.threshold = threshold
        
    def clean_answer(self, answer: str) -> str:
        """Clean and normalize answer text"""
        if not answer:
            return ""
        
        # Convert to string and strip
        answer = str(answer).strip()
        
        # Remove extra spaces
        answer = re.sub(r'\s+', ' ', answer)
        
        # Remove common punctuation at the end
        answer = re.sub(r'[.,!?;:]+$', '', answer)
        
        # Remove quotes
        answer = answer.replace('"', '').replace("'", '')
        
        return answer
    
    def check_answer(self, user_answer: str, correct_answer: str, 
                    custom_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Check if user answer matches correct answer with fuzzy matching
        
        Args:
            user_answer: The user's provided answer
            correct_answer: The correct answer to match against
            custom_threshold: Custom threshold for this specific match
            
        Returns:
            Dictionary containing match result and details
        """
        if not user_answer or not correct_answer:
            return {
                'is_match': False,
                'score': 0.0,
                'confidence': 'invalid',
                'method': 'validation'
            }
        
        # Clean inputs
        clean_user = self.clean_answer(user_answer)
        clean_correct = self.clean_answer(correct_answer)
        
        # Exact match check
        if clean_user == clean_correct:
            return {
                'is_match': True,
                'score': 1.0,
                'confidence': 'exact',
                'method': 'exact_match'
            }
        
        # Case-insensitive exact match
        if clean_user.lower() == clean_correct.lower():
            return {
                'is_match': True,
                'score': 1.0,
                'confidence': 'exact_case_insensitive',
                'method': 'case_insensitive'
            }
        
        # Fuzzy matching using difflib
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        similarity = difflib.SequenceMatcher(None, clean_user.lower(), clean_correct.lower()).ratio()
        
        if similarity >= threshold:
            confidence = self._get_confidence_level(similarity)
            return {
                'is_match': True,
                'score': similarity,
                'confidence': confidence,
                'method': 'fuzzy_match'
            }
        
        # Try alternative matching strategies
        alt_result = self._try_alternative_matches(clean_user, clean_correct)
        if alt_result['is_match']:
            return alt_result
        
        # No match found
        return {
            'is_match': False,
            'score': similarity,
            'confidence': 'no_match',
            'method': 'no_match'
        }
    
    def _try_alternative_matches(self, user_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Try alternative matching strategies"""
        user_lower = user_answer.lower()
        correct_lower = correct_answer.lower()
        
        # Substring matching
        if user_lower in correct_lower or correct_lower in user_lower:
            return {
                'is_match': True,
                'score': 0.8,
                'confidence': 'partial_match',
                'method': 'substring_match'
            }
        
        # Common variations (numbers)
        if self._check_common_variations(user_lower, correct_lower):
            return {
                'is_match': True,
                'score': 0.9,
                'confidence': 'variation_match',
                'method': 'common_variation'
            }
        
        # Word order check
        if self._check_word_order(user_lower, correct_lower):
            return {
                'is_match': True,
                'score': 0.85,
                'confidence': 'word_order_match',
                'method': 'word_order'
            }
        
        return {'is_match': False}
    
    def _check_common_variations(self, user_answer: str, correct_answer: str) -> bool:
        """Check for common variations like number words vs digits"""
        number_variations = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'zero': '0'
        }
        
        user_converted = user_answer
        correct_converted = correct_answer
        
        for word, num in number_variations.items():
            user_converted = re.sub(r'\b' + word + r'\b', num, user_converted)
            correct_converted = re.sub(r'\b' + word + r'\b', num, correct_converted)
        
        return user_converted == correct_converted
    
    def _check_word_order(self, user_answer: str, correct_answer: str) -> bool:
        """Check if words are in different order but same content"""
        user_words = sorted(user_answer.split())
        correct_words = sorted(correct_answer.split())
        
        return user_words == correct_words
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on similarity score"""
        if score >= 0.95:
            return 'high'
        elif score >= 0.85:
            return 'medium'
        elif score >= 0.7:
            return 'low'
        else:
            return 'very_low'
    
    def check_multiple_answers(self, user_answer: str, 
                             possible_answers: List[str]) -> Dict[str, Any]:
        """Check user answer against multiple possible correct answers"""
        best_match = {'is_match': False, 'score': 0.0}
        
        for answer in possible_answers:
            result = self.check_answer(user_answer, answer)
            if result['is_match'] and result['score'] > best_match['score']:
                best_match = {**result, 'matched_answer': answer}
        
        return best_match


# Create singleton instance
answer_matcher = AnswerMatcher()


def check_answer_fuzzy(user_answer: str, correct_answer: str, 
                      threshold: float = 0.7) -> Dict[str, Any]:
    """
    Convenience function for checking answers with fuzzy matching
    
    Args:
        user_answer: User's answer
        correct_answer: Correct answer
        threshold: Similarity threshold (0.0-1.0)
        
    Returns:
        Match result dictionary
    """
    matcher = AnswerMatcher(threshold)
    return matcher.check_answer(user_answer, correct_answer)


# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("paris", "Paris"),  # Case difference
        ("paaris", "Paris"),  # Typo
        ("new york", "New York"),  # Case and spacing
        ("3", "three"),  # Number variation
        ("york new", "new york"),  # Word order
        ("completely wrong", "Paris"),  # No match
    ]
    
    matcher = AnswerMatcher()
    
    for user, correct in test_cases:
        result = matcher.check_answer(user, correct)
        print(f"'{user}' vs '{correct}': {result['is_match']} ({result['confidence']}, {result['method']})")