def load_text_file(file_path):

    with open(file_path, 'r') as file:
        data = file.read().replace('\n', '')

    return(data)