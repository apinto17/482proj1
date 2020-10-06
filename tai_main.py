
import json 
from tai_util import *

def main():
    document = get_docs("tai-documents-v3.json")[0]["plaintext"]
    subject = get_subject(document)

    print("Document:\n\n")
    print(document)

    print("The subject is: " + str(subject))



if(__name__ == "__main__"):
    main()


