from route_llm_classifier import RouteLLMClassifier

def test_models(easy_question, hard_question):
    # Define model pairs
    model_pairs = [
        ("gemini/gemini-2.5-pro-exp-03-25", "gemini/gemini-1.5-flash"),
        ("gpt-4", "gpt-3.5-turbo"),
        ("together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    ]
    
    # Loop through each model pair
    for strong_model, weak_model in model_pairs:
        print(f"\nTesting with models: {strong_model} (Strong) and {weak_model} (Weak)")
        
        # Initialize the RouteLLMClassifier with the current model pair
        router = RouteLLMClassifier(strong_model, weak_model)
        
        # Test the Easy question
        print("\nEasy Question Result:")
        try:
            easy_response = router.classify(easy_question)
            print(f"Response: {easy_response}")
        except Exception as e:
            print(f"Error with easy question: {e}")
        
        # Test the Hard question
        print("\nHard Question Result:")
        try:
            hard_response = router.classify(hard_question)
            print(f"Response: {hard_response}")
        except Exception as e:
            print(f"Error with hard question: {e}")