def load_text_file(file_path):

    with open(file_path, 'r') as file:
        content = file.read()  # Reads the entire file as a single string with all formatting

    return(content)