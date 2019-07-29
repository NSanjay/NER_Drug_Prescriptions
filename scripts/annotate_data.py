import argparse
from pathlib import *
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize
import os

classes = {"AdverseReaction":"_ADR_","Severity":"_SEV_","Negation":"_NEG_","Factor":"_FAC_","Animal":"_ANM_","DrugClass":"_DCLS_"}
def annotate_file(args,file_name):
    file_text = Path(args.data_dir + file_name)
    tree = ET.parse(file_text)
    root_node = tree.getroot()
    section_dict = dict()

    entity_mention_tags = root_node.iter('Mention')
    current_entity = next(entity_mention_tags)
    print("entit:::",current_entity.attrib["str"])
    with open(Path(args.data_dir + args.output_tags_file),mode="w",encoding='utf-8') as out_tag_file, \
            open(Path(args.data_dir + args.output_words_file),mode="w",encoding='utf-8') as out_word_file:

        for section in root_node.iter('Section'):
            print(section.attrib)
            section_dict[section.attrib["id"]] = section.text
            number_of_characters = 0
            sentences = sent_tokenize(section.text)
            print("len",len(sentences))
            for i,line in enumerate(sentences):
                length_of_line = len(line)
                out_word_file.write(line+"\n")
                copy_of_line = line[:]

                #print("start::",current_entity.attrib["start"],"chars::",number_of_characters,"line_length::",length_of_line)
                while current_entity is not None and int(current_entity.attrib["start"].split(',')[0]) >= number_of_characters \
                        and int(current_entity.attrib["start"][0]) + int(current_entity.attrib["len"].split(',')[0]) <= number_of_characters + length_of_line:
                    #print("inside")
                    length_of_entity = len(current_entity.attrib["str"].split())
                    position_in_string = current_entity["str"]
                    if length_of_entity == 1:
                        replace_text = "S-" + classes[current_entity.attrib["type"]]

                    else:
                        split_text = current_entity.attrib["str"].split()
                        for i in range(length_of_entity):
                            if i == 0:
                                split_text[i] = "B-" + classes[current_entity.attrib["type"]]
                            elif i == length_of_entity - 1:
                                split_text[i] = "E-" + classes[current_entity.attrib["type"]]
                            else:
                                split_text[i] = "I" + classes[current_entity.attrib["type"]]
                        replace_text = ' '.join(split_text)

                    #print("replace",replace_text)
                    copy_of_line = copy_of_line.replace(current_entity.attrib["str"], replace_text)
                    #print("copy_of_line",copy_of_line)

                    try:
                        current_entity = next(entity_mention_tags)
                    except StopIteration:
                        break
                copy_line_split = copy_of_line.split()
                concat_list = [" "] * len(copy_line_split)
                print("line::",copy_line_split)
                for i, line_word in enumerate(copy_line_split):
                    print("line_word",line_word)
                    if not any(word in line_word for word in classes.values()):
                        concat_list[i] = "O"
                    else:
                        concat_list[i] = line_word
                out_tag_file.write(" ".join(concat_list)+"\n")
                number_of_characters += length_of_line
                #current_entity = next(entity_mention_tags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="../data/train_xml/",
        help="Enter the directory with trailing slash containing the files to be annotated")
    parser.add_argument("--output_tags_file",type=str,default="train_annotated.tags.txt",
        help="")
    parser.add_argument("--output_words_file",type=str,default="train_annotated.words.txt",
        help="")
    args = parser.parse_args()
    file_name = "ADCETRIS.xml"

    annotate_file(args,file_name)

if __name__ == "__main__":
    main()