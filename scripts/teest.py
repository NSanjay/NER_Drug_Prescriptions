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

    with open(Path(args.data_dir + args.output_tags_file),mode="w",encoding='utf-8') as out_tag_file, \
            open(Path(args.data_dir + args.output_words_file),mode="w",encoding='utf-8') as out_word_file:

        for section in root_node.iter('Section'):
            #print(section.attrib)
            section_dict[section.attrib["id"]] = section.text
            number_of_characters = 0
            #sentences = sent_tokenize(section.text)
            total_length = 0
            lines = section.text.split('\n')
            for i, line in enumerate(lines):
                #print("here", i, line=="",len(line))
                if line == "":

                    total_length += 1
                    continue
                out_word_file.write(line.strip()+"\n")
                line_length = len(line)
                replacement_list = []
                copy_of_line = line[:]
                while int(current_entity.attrib["start"].split(',')[0]) >= total_length and \
                    int(current_entity.attrib["start"].split(',')[0]) + int(current_entity.attrib["len"].split(',')[0]) <= total_length + line_length:
                    length_of_entity = len(current_entity.attrib["str"].split())
                    position_in_string = int(current_entity.attrib["start"].split(',')[0]) - total_length

                    #print("entit:::", current_entity.attrib["str"])
                    #print("position_in_string", position_in_string)

                    assert line[position_in_string:position_in_string + int(current_entity.attrib["len"].split(',')[0])] == \
                           current_entity.attrib["str"][:int(current_entity.attrib["len"].split(',')[0])], \
                        "Strings::: {} and {} not equal".format(line[position_in_string:position_in_string + int(current_entity.attrib["len"].split(',')[0])],
                                    current_entity.attrib["str"][:int(current_entity.attrib["len"].split(',')[0])])
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
                        replacement_list.append((position_in_string, current_entity.attrib["str"], replace_text))
                    try:
                        current_entity = next(entity_mention_tags)
                    except StopIteration:
                        break

                for tup in replacement_list[::-1]:
                    string_length = len(tup[1])
                    copy_of_line = copy_of_line[:tup[0]] + tup[2] + copy_of_line[tup[0]+len(tup[1]):]

                copy_line_split = copy_of_line.split()
                concat_list = [" "] * len(copy_line_split)
                #print("line::", copy_line_split)
                for i, line_word in enumerate(copy_line_split):
                    #print("line_word", line_word)
                    if not any(word in line_word for word in classes.values()):
                        concat_list[i] = "O"
                    else:
                        concat_list[i] = line_word
                out_tag_file.write(" ".join(concat_list).strip() + "\n")
                total_length += len(line) + 1
                    # print("replace",replace_text)
                    #copy_of_line = copy_of_line.replace(current_entity.attrib["str"], replace_text)

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