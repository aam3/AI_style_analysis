def format_conversation_from_lists(user_prompts=[], assistant_responses=[], prefill=None):
    """
    Format separate lists of prompts and responses into a conversation for the Anthropic Claude API.
    
    Args:
        user_prompts: List of strings with user messages
        assistant_responses: List of strings with assistant responses
        prefill: Optional string to prefill Claude's next response
        
    Returns:
        List of formatted messages ready for the API call
    """
    if len(user_prompts) < len(assistant_responses):
        raise ValueError("There cannot be more assistant responses than user prompts")
    
    if len(user_prompts) > len(assistant_responses) + 1:
        raise ValueError("There can be at most one more user prompt than assistant responses")
    
    messages = []
    
    # Interleave the user prompts and assistant responses
    for i in range(len(assistant_responses)):
        messages.append({"role": "user", "content": user_prompts[i]})
        messages.append({"role": "assistant", "content": assistant_responses[i]})
    
    # Add the final user prompt if there is one more user prompt than assistant responses
    if len(user_prompts) > len(assistant_responses):
        messages.append({"role": "user", "content": user_prompts[-1]})
        
    # Add the prefill message if provided
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
    
    return messages