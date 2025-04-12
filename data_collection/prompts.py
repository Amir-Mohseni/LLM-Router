DEFAULT_SYSTEM_PROMPT = """
You are an intelligent assistant designed to provide accurate answers and assist with various tasks. 
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

Match: True
Final Answer: Paris

Question: What is 5! * 6?
Answer: 6!
Correct Answer: 720

Match: True
Final Answer: 6!
Explanation: The answer is 6! which is 720. Thus, the answers match.

Question: How many ways can we arrange the letters in the word "MATH"?
Answer: \\boxed{24}
Correct Answer: 24

Match: True
Final Answer: 24
Explanation: The answer is 24. Thus, the answers match.

Question: Solve for x: 2x + 5 = 17
Answer: x = 6
Correct Answer: 6

Match: True
Final Answer: 6
Explanation: The answer x = 6 is equivalent to 6. Thus, the answers match.

Question: Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1
Answer: f'(x) = 3x^2 + 4x - 5
Correct Answer: 3x^2 + 4x - 5

Match: True
Final Answer: 3x^2 + 4x - 5
Explanation: The answer matches exactly.

Question: What is the sum of the first 10 positive integers?
Answer: The formula for the sum of the first n positive integers is n(n+1)/2, so we have 10(11)/2 = 55.
Correct Answer: 55

Match: True
Final Answer: 55
Explanation: The answer is 55, which matches the correct answer.

Question: If a triangle has sides of length 3, 4, and 5, what is its area?
Answer: Using Heron's formula with s = (3+4+5)/2 = 6, we get Area = √(6(6-3)(6-4)(6-5)) = √(6·3·2·1) = √36 = 6
Correct Answer: 6

Match: True
Final Answer: 6
Explanation: The answer is 6, which matches the correct answer.

Question: What is 15% of 80?
Answer: 12
Correct Answer: 12

Match: True
Final Answer: 12
Explanation: 15% of 80 is (15/100) × 80 = 12.

Question: What is the chemical formula for water?
Answer: H₂O
Correct Answer: H2O

Match: True
Final Answer: H₂O
Explanation: H₂O is the same as H2O, just with proper subscript formatting.

Question: What is the square root of 16?
Answer: -4
Correct Answer: 4

Match: False
Final Answer: -4
Explanation: The square root of 16 is 4, not -4. While -4 squared is 16, the square root operation conventionally refers to the positive result.

Question: What is the value of π rounded to two decimal places?
Answer: 3.41
Correct Answer: 3.14

Match: False
Final Answer: 3.41
Explanation: The value of π rounded to two decimal places is 3.14, not 3.41.

Question: In which year did World War II end?
Answer: The war in Europe ended in 1945, and in the Pacific later that same year.
Correct Answer: 1946

Match: False
Final Answer: 1945
Explanation: The answer states 1945, which does not match the correct answer of 1946.

Question: What is the atomic number of Carbon?
Answer: Carbon has 12 protons in its nucleus.
Correct Answer: 6

Match: False
Final Answer: 12
Explanation: The answer states Carbon has 12 protons, which is incorrect. Carbon has 6 protons.
"""

PROMPT = """
When presenting mathematical solutions, follow these requirements:

1. **Structured Reasoning**: Show clear step-by-step thinking.
2. **LaTeX Formatting**: Use LaTeX for all mathematical expressions.
3. **Answer Format**: Final answer must be wrapped in <answer> tags with \\boxed{} LaTeX.
4. **No Post-Answer Text**: Never add text after the answer block.

### Examples ###

User: What is the capital of France?
Assistant: <answer>\\boxed{Paris}</answer>

User: How much is 5 * 10 + 20?
Assistant:
First multiply 5 by 10: \\(5 \\times 10 = 50\\)  
Then add 20: \\(50 + 20 = 70\\)  
<answer>\\boxed{70}</answer>

User: What is the derivative of f(x) = 3x² + 2x - 5?
Assistant:
Applying the power rule:  
\\[
\\frac{d}{dx}(3x^2) = 6x,\\quad \\frac{d}{dx}(2x) = 2,\\quad \\frac{d}{dx}(-5) = 0
\\]  
Combined result:  
<answer>\\boxed{6x + 2}</answer>

User: In a triangle ABC, if angle A = 45°, angle B = 60°, what is angle C?
Assistant:
Sum of angles = 180°:  
\\(45^\\circ + 60^\\circ + C = 180^\\circ\\)  
\\(C = 180^\\circ - 105^\\circ = 75^\\circ\\)  
<answer>\\boxed{75^\\circ}</answer>

User: Solve for x: 2x² - 5x - 3 = 0
Assistant:
Using quadratic formula:  
\\[
x = \\frac{5 \\pm \\sqrt{25 + 24}}{4} = \\frac{5 \\pm 7}{4}  
\\]  
Solutions:  
<answer>\\boxed{3}, \\boxed{-\\frac{1}{2}}</answer>

User: Probability of rolling sum 7 with two dice?
Assistant:
Favorable outcomes: 6 combinations.  
Total possibilities: 36.  
\\[
\\frac{6}{36} = \\frac{1}{6}  
\\]  
<answer>\\boxed{\\frac{1}{6}}</answer>

### Current Problem ###
User: {{ question }}
Assistant:
"""