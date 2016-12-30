file_name = '/Users/AdriaRomeroLopez/Desktop/utils/method3_VGG16_scores_test.txt'
with open(file_name, 'r') as my_file:
    text = my_file.read()
    text = text.replace("[", "")
    text = text.replace("]", "")

# If you wish to save the updates back into a cleaned up file
with open(file_name, 'w') as my_file:
    my_file.write(text)
