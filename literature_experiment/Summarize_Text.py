
def initial_prompt(TEXT):
    return(f"""At the bottom of this prompt is a personal essay. I need you to do 3 things:

    1.	Summarize the events that occur in this excerpt.
    2.	Summarize the physical scenes of this excerpt.
    3.	Summarize the emotional state(s) of the narrator.

    Output all as numbered lists with a maximum of 3 points for each list. The headings for each list should be "## Events:", "## Physical Scenes:", and "## Emotional States:".

    <text-to-summarize>
    {TEXT}
    </text-to-summarize>
    """)


def remove_details_prompt():
    return f"Now take the three lists above and rephrase them so that there are no specific characters or names mentioned. Condense and generalize the wording as much as possible so that a person reading the list could not identify the book. For the third list, there should be no mention of specific people or places; focus only on emotions."

def condense_plot_prompt():
    return f"Condense the first list of events even further to a summary that is a single sentence. Return the full revised list."