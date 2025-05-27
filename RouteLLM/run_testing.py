from testing import test_models

hard_question = "Can you solve the integral of the function 𝑓 ( 𝑥 ) = 𝑥 3 sin ⁡ ( 𝑥 ) f(x)=x 3 sin(x) from 0 to 𝜋 π and explain the steps involved?"
easy_question = "Can you explain the concept of quantum computing and its potential impact on industries like healthcare and finance?"
#Test:

# Need api key for OPENAI_API_KEY, GEMINI_API_KEY, TOGETHERAI_API_KEY
test_models(easy_question=easy_question,hard_question=hard_question)