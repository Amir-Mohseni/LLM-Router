#!/usr/bin/env python3
"""Prompt templates for various types of questions."""
import jinja2

def character_filter(index):
    """Convert an index to a character (0->A, 1->B, etc.)"""
    return chr(65 + index)  # A=65, B=66, etc. in ASCII

# Create Jinja2 environment with filters
env = jinja2.Environment()
env.filters['character'] = character_filter

# Default system prompt
DEFAULT_SYSTEM_PROMPT = """
You are an intelligent assistant designed to provide accurate answers and assist with various tasks.
"""

PROMPT_TEMPLATE = """
You are an intelligent assistant designed to provide accurate answers and assist with various tasks.
When presenting mathematical solutions, follow these requirements:

1. **Structured Reasoning**: Show clear step-by-step thinking
2. **LaTeX Formatting**: Use LaTeX for all mathematical expressions
3. **Answer Format**: Final answer must be wrapped in <answer> tags with \\boxed{} LaTeX
4. **No Post-Answer Text**: Never add text after the answer block

### Examples ###

User: What is the capital of France?
Assistant: <answer>\\boxed{Paris}</answer>

User: How much is 5 * 10 + 20?
Assistant:
First multiply 5 by 10: $5 \\times 10 = 50$  
Then add 20: $50 + 20 = 70$  
<answer>\\boxed{70}</answer>

User: What is the derivative of f(x) = 3x² + 2x - 5?
Assistant:
Applying the power rule:  
$$
\\frac{d}{dx}(3x^2) = 6x, \\quad \\frac{d}{dx}(2x) = 2, \\quad \\frac{d}{dx}(-5) = 0
$$  
Combined result:  
<answer>\\boxed{6x + 2}</answer>

User: In a triangle ABC, if angle A = 45°, angle B = 60°, what is angle C?
Assistant:
Sum of angles = 180°:  
$45^\\circ + 60^\\circ + C = 180^\\circ$  
$C = 180^\\circ - 105^\\circ = 75^\\circ$  
<answer>\\boxed{75^\\circ}</answer>

User: Solve for x: 2x² - 5x - 3 = 0
Assistant:
Using quadratic formula:  
$$
x = \\frac{5 \\pm \\sqrt{25 + 24}}{4} = \\frac{5 \\pm 7}{4}  
$$  
Solutions:  
<answer>\\boxed{3}, \\boxed{-\\frac{1}{2}}</answer>

User: Probability of rolling sum 7 with two dice?
Assistant:
Favorable outcomes: 6 combinations  
Total possibilities: 36  
$$
\\frac{6}{36} = \\frac{1}{6}  
$$  
<answer>\\boxed{\\frac{1}{6}}</answer>

### Current Problem ###
User: {{ question }}
Assistant:
"""

# Define the objective prompt template as a copy of the general template
OBJECTIVE_PROMPT_TEMPLATE = PROMPT_TEMPLATE

# Main math prompt used for non-MCQ questions
MATH_PROMPT = PROMPT_TEMPLATE

# Open-ended prompt for general questions (no LaTeX formatting required)
OPEN_ENDED_PROMPT = """
You are an intelligent assistant designed to provide accurate answers and assist with various tasks.

Provide a clear, comprehensive, and well-structured response to the following question. Your answer should be:
- Detailed and informative
- Easy to understand
- Well-organized with proper structure
- Factually accurate

Question: {{ question }}
"""

JUDGE_PROMPT = """
You are a judge that evaluates the correctness of the answer provided. You will be given a response to a question,
and the ground truth correct answer. 
You will need to evaluate whether the provided answer matches the correct answer or not. If the answer is not an EXACT match but is logically equivalent, you should return True.
In addition, you will need to extract the final answer given to the question from the response provided.

Here are some examples:

Question: What is the capital of France?
Answer: Paris
Correct Answer: Paris

{
  "is_match": true,
  "final_answer": "Paris",
  "explanation": "The provided answer 'Paris' is exactly the same as the correct answer 'Paris'."
}

Question: What is 5! * 6?
Answer: 6!
Correct Answer: 720

{
  "is_match": true,
  "final_answer": "6!",
  "explanation": "The answer is 6! which equals 720, which is the same as the correct answer."
}

Question: How many ways can we arrange the letters in the word "MATH"?
Answer: \\boxed{24}
Correct Answer: 24

{
  "is_match": true,
  "final_answer": "24",
  "explanation": "The answer \\boxed{24} is equivalent to 24, which matches the correct answer."
}

Question: Solve for x: 2x + 5 = 17
Answer: x = 6
Correct Answer: 6

{
  "is_match": true,
  "final_answer": "6",
  "explanation": "The answer x = 6 is equivalent to 6, which matches the correct answer."
}

Question: Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1
Answer: f'(x) = 3x^2 + 4x - 5
Correct Answer: 3x^2 + 4x - 5

{
  "is_match": true,
  "final_answer": "3x^2 + 4x - 5",
  "explanation": "The answer matches exactly with the correct answer."
}

Question: What is the sum of the first 10 positive integers?
Answer: The formula for the sum of the first n positive integers is n(n+1)/2, so we have 10(11)/2 = 55.
Correct Answer: 55

{
  "is_match": true,
  "final_answer": "55",
  "explanation": "The final answer calculated is 55, which matches the correct answer."
}

Question: What is 15% of 80?
Answer: 12
Correct Answer: 12

{
  "is_match": true,
  "final_answer": "12",
  "explanation": "The answer 12 is exactly correct, as 15% of 80 is (15/100) × 80 = 12."
}

Question: What is the chemical formula for water?
Answer: H₂O
Correct Answer: H2O

{
  "is_match": true,
  "final_answer": "H₂O",
  "explanation": "H₂O is the same as H2O, just with proper subscript formatting."
}

Question: What is the square root of 16?
Answer: -4
Correct Answer: 4

{
  "is_match": false,
  "final_answer": "-4",
  "explanation": "The square root of 16 is 4, not -4. While -4 squared is 16, the square root operation conventionally refers to the positive result."
}

Question: What is the value of π rounded to two decimal places?
Answer: 3.41
Correct Answer: 3.14

{
  "is_match": false,
  "final_answer": "3.41",
  "explanation": "The value of π rounded to two decimal places is 3.14, not 3.41."
}

Question: In which year did World War II end?
Answer: The war in Europe ended in 1945, and in the Pacific later that same year.
Correct Answer: 1946

{
  "is_match": false,
  "final_answer": "1945",
  "explanation": "The answer states 1945, which does not match the correct answer of 1946."
}

Now evaluate the current problem:

Question: {{ question }}
Answer: {{ answer }}
Correct Answer: {{ correct_answer }}
"""

# Prompt templates for multiple-choice questions
MCQ_PROMPT_TEMPLATE = """
Answer the following multiple-choice question by analyzing all options carefully.

1. **Step-by-Step Reasoning**: Briefly explain your reasoning and process of elimination.
2. **Final Choice Format**: Wrap the final selected answer choice (e.g., A, B, C, or D) in <answer> tags using \\boxed{} LaTeX.
3. **Answer the Question Fully**: Use only the information needed to justify the correct choice.
4. **No Post-Answer Text**: Never add any text after the answer block.

### Examples ###

User: What is the capital of France?  
Options:  
A. Berlin  
B. Madrid  
C. Paris  
D. Rome  
Assistant:  
Paris is the capital of France. The other cities are capitals of Germany, Spain, and Italy respectively.  
<answer>\\boxed{C}</answer>

User: Which gas do plants primarily use for photosynthesis?  
Options:  
A. Oxygen  
B. Carbon Dioxide  
C. Nitrogen  
D. Hydrogen  
Assistant:  
Plants use carbon dioxide during photosynthesis to create glucose and oxygen.  
<answer>\\boxed{B}</answer>

User: Who wrote 'Hamlet'?  
Options:  
A. Charles Dickens  
B. William Shakespeare  
C. Jane Austen  
D. Mark Twain  
Assistant:  
'Hamlet' is a tragedy written by William Shakespeare.  
<answer>\\boxed{B}</answer>

### Current Problem ###
User: {{ question }}
Options:
{% for choice in choices %}
{{ choice }}
{% endfor %}
Assistant:
"""

# For compatibility with existing code
MCQ_PROMPT = MCQ_PROMPT_TEMPLATE

SUBJECTIVE_PROMPT_TEMPLATE = """
You are a subject matter expert in the field of education.
Given a subjective question, you need to provide a detailed answer with clear reasoning.

1. **Step-by-Step Explanation**: Break down your solution clearly
2. **Relevant Details**: Include important concepts and formulas
3. **Precise Language**: Use technical terms accurately and provide definitions if needed
4. **Comprehensive Answer**: Cover all aspects of the question

Question: {{ question }}
"""

# Specialized prompt for judging MCQ questions
MCQ_JUDGE_PROMPT = """
You are a judge that evaluates the correctness of answers to multiple-choice questions. 
You will be given an MCQ question, the available choices, the model's response, and the correct answer.

Your task is to determine if the model's answer matches the correct answer. The correct answer could be:
1. A letter (A, B, C, D, etc.) corresponding to the correct choice
2. The actual text of the correct choice
3. A boxed letter format like \\boxed{A}

If the model identified the same choice as the correct answer (whether by letter or content), return True.

Here are some examples:

Question: What is the capital of France?
Choices:
A. London
B. Berlin
C. Paris
D. Rome
Answer: C
Correct Answer: C

{
  "is_match": true,
  "final_answer": "C",
  "explanation": "The model answered 'C', which correctly corresponds to 'Paris', matching the correct answer."
}

Question: What is the capital of France?
Choices:
A. London
B. Berlin
C. Paris
D. Rome
Answer: Paris
Correct Answer: C

{
  "is_match": true,
  "final_answer": "Paris",
  "explanation": "The model answered 'Paris', which corresponds to option C, matching the correct answer."
}

Question: What is the capital of France?
Choices:
A. London
B. Berlin
C. Paris
D. Rome
Answer: \\boxed{C}
Correct Answer: Paris

{
  "is_match": true,
  "final_answer": "C",
  "explanation": "The model answered '\\boxed{C}', which corresponds to 'Paris', matching the correct answer."
}

Question: What is the capital of Germany?
Choices:
A. London
B. Berlin
C. Paris
D. Rome
Answer: Paris
Correct Answer: B

{
  "is_match": false,
  "final_answer": "Paris",
  "explanation": "The model answered 'Paris' (option C), but the correct answer is 'Berlin' (option B)."
}

Now evaluate the current problem:

Question: {{ question }}
Choices:
{% for choice in choices %}
{{ loop.index0 | character }}: {{ choice }}
{% endfor %}
Answer: {{ answer }}
Correct Answer: {{ correct_answer }}
"""