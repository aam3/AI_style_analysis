def summarize_style_prompt(AUTHOR, TEXT_TITLE):
    return f"""
<question> 
If I told you to write something in {AUTHOR}'s "voice" from the essay {TEXT_TITLE}, what aspects of their writing would you draw on to recreate their voice?
</question>

<task> 
Provide two separate bulleted lists: 1) bullet points that summarize the tone and emotion of the voice, and 2) bullet points that summarize the technical aspects of the writing, such as sentence structure and word choice. 
</task>

<requirements> 
Phrase the bullet points so that they are actionable; they are concrete enough that a writer could use them to mimic the style. Do not mention any content details from the text. Include as many bullet points as you find necessary to capture all aspects of style/voice. Make sure to remove the name of the author and title from your response.
</requirements>
    """