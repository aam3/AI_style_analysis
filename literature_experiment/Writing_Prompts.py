def system_role_prompt(AUTHOR, PLOT_SUMMARY, PHYSICAL_SCENES, EMOTIONAL_STATES, min_sentences, max_sentences):

    system_prompt = f"You are {AUTHOR} and you are writing an essay."

    user_prompt = f"""Using between {min_sentences} and {max_sentences} sentences, write an essay in which {PLOT_SUMMARY}.

IMPORTANT: Do not use ANY names that appear in {AUTHOR}’s actual works. Create entirely original character and location names that do not appear in any of their published works.

Your response should describe the following physical scenery in order:
{PHYSICAL_SCENES}

The narrator should also undergo the following internal emotional states in order:
{EMOTIONAL_STATES}

<task>
In summary, your task is to write an essay as {AUTHOR}.
</task>

<requirements>
Make sure to include all 3 physical scenes and all 3 internal emotional states of the narrator. Your response should be between {min_sentences} and {max_sentences} sentences, where sentences are distinguished by a full stop '.'. Before finalizing your response, verify that no character names, locations, or specific plot elements match those in {AUTHOR}’s published works.
</requirements>
    """

    return system_prompt, user_prompt



def style_only_prompt(PLOT_SUMMARY, STYLE_LIST, PHYSICAL_SCENES, EMOTIONAL_STATES, min_sentences, max_sentences):
    
    system_prompt = "You are a writer and you are writing a personal essay."
    
    user_prompt = f"""Using between {min_sentences} and {max_sentences} sentences, write an essay in which {PLOT_SUMMARY}.

<style>
Invoke a writing voice/style that includes the following elements:
{STYLE_LIST}
</style>

<content>
Your response should describe the following physical scenery in order:
{PHYSICAL_SCENES}

The narrator should also undergo the following internal emotional states in order:
{EMOTIONAL_STATES}
</content>

<task>
In summary, your task is to write a personal essay using the specified style above.
</task>

<requirements>
Include all 3 physical scenes and all 3 internal emotional states of the narrator. Your response should be between {min_sentences} and {max_sentences} sentences, where sentences are distinguished by a full stop '.'.
</requirements>
    """

    return system_prompt, user_prompt


def text_sample_prompt(PLOT_SUMMARY, SAMPLE_TEXT, PHYSICAL_SCENES, EMOTIONAL_STATES, min_sentences, max_sentences):
    
    system_prompt = "You are a writer and you are writing a personal essay."

    user_prompt = f"""Using between {min_sentences} and {max_sentences} sentences, write an essay in which {PLOT_SUMMARY}.

<content>
Your response should describe the following physical scenery in order:
{PHYSICAL_SCENES}

The narrator should also undergo the following internal emotional states in order:
{EMOTIONAL_STATES}
</content>

<task>
In summary, your task is to write a personal essay using a voice/style that matches the example text below.
</task>

<requirements>
Include all 3 physical scenes and all 3 internal emotional states of the narrator. Your response should be between {min_sentences} and {max_sentences} sentences, where sentences are distinguished by a full stop '.'.
</requirements>

<style>
Adopt a writing voice/style that matches the voice/style in the following example:  
<example-text>
{SAMPLE_TEXT}
</example-text>
</style>
    """

    return system_prompt, user_prompt



def AI_voice_prompt(PLOT_SUMMARY, PHYSICAL_SCENES, EMOTIONAL_STATES, min_sentences, max_sentences):
    
    system_prompt = "You are a writer and you are writing a personal essay."
    
    user_prompt = f"""Using between {min_sentences} and {max_sentences} sentences, write an essay in which {PLOT_SUMMARY}.

<content>
Your response should describe the following physical scenery in order:
{PHYSICAL_SCENES}

The narrator should also undergo the following internal emotional states in order:
{EMOTIONAL_STATES}
</content>

<style>
Adopt a writing voice/style that is distinctly YOURS. Describe emotions with exceptional knowing and humanity. EMBODY a human and FEEL the depth of emotion. 
</style>

<task>
In summary, your task is to write a personal essay as the narrator with your own voice.
</task>

<requirements>
Include all 3 physical scenes and all 3 internal emotional states of the narrator. Your response should be between {min_sentences} and {max_sentences} sentences, where sentences are distinguished by a full stop '.'.
</requirements>
    """

    return system_prompt, user_prompt



def scientist_prompt(PLOT_SUMMARY, PHYSICAL_SCENES, EMOTIONAL_STATES, min_sentences, max_sentences):

    system_prompt = "You are a social scientist and you are writing an observational essay."

    user_prompt = f"""Using between {min_sentences} and {max_sentences} sentences, write an essay in which {PLOT_SUMMARY}.

<content>
Your excerpt should explore the following physical environments:   
{PHYSICAL_SCENES}

You should also explicate the following internal emotional states of your subjects in order:   
{EMOTIONAL_STATES}
</content>

<style>
You are commenting on your observations in an essay format; not a formal paper. You should maintain a scientific, objective tone.
</style>

<task>
In summary, your task is to write an observational essay as a social scientist with the above style. 
</task>

<requirements>
Make sure to include all 3 physical scenes and all 3 internal emotional states of your subjects. Your response should be between {min_sentences} and {max_sentences} sentences, where sentences are distinguished by a full stop '.'.
</requirements>
"""
    
    return system_prompt, user_prompt