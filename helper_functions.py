import os
import glob

def add_a_underscore_between_two_word():
    """
    This is for changing "traffic light" to "traffic_light", or "stop sign" to "stop_sign" etc.
    """

    # change directory to the correct path
    directory = 'test/Label'

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            fin = open(os.path.join(directory,filename), "rt")
            data = fin.read()
            # replace all occurrences of the required string
            # change "Traffic light" and "Stop sign" to any two words
            data = data.replace("Traffic light", "Traffic_light")
            data = data.replace("Stop sign", "Stop_sign")
            fin.close()
            # open the file in write mode
            fin = open(os.path.join(directory,filename), "wt")
            # overwrite the input file with the resulting data
            fin.write(data)
            fin.close()
            continue
        else:
            continue

add_a_underscore_between_two_word()


def list_jpg_file_paths_in_a_txt():
    """
    This is for filtering the files in directory, then write the path of the file to a .txt file
    """

    # change "train" to "test", and "obj" to "test" for test.txt

    # filter the files in a directory so that only the .jpg files are in the onlyfiles
    onlyfiles = glob.glob("train/*.jpg")

    # replace string with the right directory string
    onlyfiles = [w.replace('train', 'data/obj') for w in onlyfiles]

    # write the file path in a .txt file
    fout = open("train.txt", "wt")
    for str in onlyfiles:
        fout.write(str+'\n')

    fout.close()

#list_jpg_file_paths_in_a_txt()