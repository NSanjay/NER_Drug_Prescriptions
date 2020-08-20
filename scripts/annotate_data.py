### new

import argparse
from pathlib import *
import xml.etree.ElementTree as ET
# from nltk.tokenize import sent_tokenize
import os
import sys
import re
import string

punctuation = string.punctuation
punctuation_idx = punctuation.find("_")
if punctuation_idx != -1:
    punctuation = punctuation[:punctuation_idx] + punctuation[punctuation_idx+1:]

classes = {"AdverseReaction":"_ADR_","Severity":"_SEV_","Negation":"_NEG_","Factor":"_FAC_","Animal":"_ANM_","DrugClass":"_DCLS_"}
def annotate_train_file(args, file_name, out_tag_file, out_word_file):
    # file_text = Path(args.data_dir + file_name)
    # print(file_name)
    tree = ET.parse(file_name)
    root_node = tree.getroot()
    section_dict = dict()

    entity_mention_tags = root_node.iter('Mention')
    current_entity = next(entity_mention_tags)


    # with open(Path(args.data_dir.parent / args.output_tags_file),mode="a",encoding='utf-8') as out_tag_file, \
    #         open(Path(args.data_dir.parent / args.output_words_file),mode="a",encoding='utf-8') as out_word_file:

    for section in root_node.iter('Section'):
        # print(section.attrib["id"], file_name, current_entity.attrib["str"])
        section_dict[section.attrib["id"]] = section.text
        number_of_characters = 0
        #sentences = sent_tokenize(section.text)
        total_length = 0
        lines = section.text.split('\n')
        # print(f"lines::{lines[:10]}")
        # print(lines[4].strip() == "")
        # import sys
        # sys.exit(0)
        for i, line in enumerate(lines):
            #print("here", i, line=="",len(line))
            if line.strip() == "":
                # print("here")
                total_length += len(line) + 1
                continue

            # replacement_dict = dict()
            line_length = len(line)
            replacement_list = []
            # copy_of_line = line[:]
            while int(current_entity.attrib["start"].split(',')[0]) >= total_length and \
                int(current_entity.attrib["start"].split(',')[0]) + int(current_entity.attrib["len"].split(',')[0]) <= total_length + line_length \
                    and current_entity.attrib["section"] == section.attrib["id"]:
                # length_of_entity = len(current_entity.attrib["str"].split())
                position_in_string = int(current_entity.attrib["start"].split(',')[0]) - total_length
                str_to_consider = current_entity.attrib["str"][:int(current_entity.attrib["len"].split(',')[0])]
                length_of_entity = len(str_to_consider.split())


                #print("entit:::", current_entity.attrib["str"])
                #print("position_in_string", position_in_string)

                assert line[position_in_string:position_in_string + int(current_entity.attrib["len"].split(',')[0])].split() == \
                       str_to_consider.split(), \
                    "Strings::: {} and {} not equal in file {} and section:::: {}".format(line[position_in_string:position_in_string + int(current_entity.attrib["len"].split(',')[0])],
                                current_entity.attrib["str"][:int(current_entity.attrib["len"].split(',')[0])], file_name,
                                        current_entity.attrib["section"])

                replace_text = ""
                if length_of_entity == 1:
                    replace_text = " S-" + classes[current_entity.attrib["type"]] + " "


                else:
                    # split_text = current_entity.attrib["str"].split()

                    for i in range(length_of_entity):
                        if i == 0:
                            replace_text += " B-" + classes[current_entity.attrib["type"]] + " "
                        elif i == length_of_entity - 1:
                            replace_text += "E-" + classes[current_entity.attrib["type"]] + " "
                        else:
                            replace_text += "I-" + classes[current_entity.attrib["type"]] + " "

                    # replace_text = ' '.join(split_text)
                    # replacement_list.append((position_in_string, current_entity.attrib["str"], replace_text))

                replacement_list.append((str_to_consider, replace_text, length_of_entity))
                try:
                    current_entity = next(entity_mention_tags)
                except StopIteration:
                    break

            # word_line = line.strip()
            # word_line = re.sub(r'[\s]+'," ", word_line)
            # out_word_file.write(word_line + "\n")
            i = 0
            copy_of_line = line.strip()
            for (text, entity, _) in replacement_list:
                if text in copy_of_line[i:]:
                    idx = copy_of_line[i:].index(text)
                    replacement_str = copy_of_line[i+idx:].replace(text, entity, 1)

                    copy_of_line = copy_of_line[:i+idx] + replacement_str
                    i += idx
                # else:
                #     print(text, file_name, line, i, replacement_list)
            copy_of_line = re.sub(r'[\s]+'," ", copy_of_line)

            copy_line_split = copy_of_line.split()
            # word_line_split = word_line.split()
            # concat_list = [None] * len(copy_line_split)
            # final_word_list = [None] * len(word_line_split)
            #

            # print(f"copy of line:: {copy_line_split}")

            i = 0
            j = 0
            # print(f"replacement_list:: {replacement_list}"
            #       f"\n copy_line_split:: {copy_line_split}")
            # print(f"")
            concat_list = []
            word_list = []
            while i < len(copy_line_split):
                # print(f"")
                tag_word = copy_line_split[i]
                if not any(word in tag_word for word in classes.values()):
                    concat_list.append("O")
                    word_list.append(tag_word)
                    i += 1
                else:
                    word, tag_word, length = replacement_list[j]
                    concat_list.append(tag_word)
                    word_list.append(word)
                    j += 1
                    i += length

            assert len(word_list) == len(concat_list), f"length of word:: {len(word_list)} !=" \
                                                               f"length of concat_list:: {len(concat_list)}" \
                                                               f"\n for file:: {file_name} and" \
                                                               f"\n line:: {word_list} and" \
                                                               f"\n tags:: {concat_list}"

            #print("line::", copy_line_split)
            # for i, (tag_word, line_word) in enumerate(zip(copy_line_split, word_line_split)):
            #     if "ERWINAZE:S-_SEV_" in tag_word:
            #         print(file_name, line, replacement_list)
            #     if not any(word in tag_word for word in classes.values()):
            #         concat_list[i] = "O"
            #         final_word_list[i] = line_word
            #     else:
            #         concat_list[i] = tag_word.strip(punctuation)
            #         final_word_list[i] = line_word.strip(punctuation)
            out_tag_file.write(" ".join(concat_list).strip() + "\n")
            out_word_file.write(" ".join(word_list).strip() + "\n")
            total_length += len(line) + 1


                # print("replace",replace_text)
                #copy_of_line = copy_of_line.replace(current_entity.attrib["str"], replace_text)


def annotate_test_file(args, file_name, out_tag_file, out_word_file):
    tree = ET.parse(file_name)
    root_node = tree.getroot()


    for section in root_node.iter('Section'):
        number_of_characters = 0
        # sentences = sent_tokenize(section.text)
        total_length = 0
        lines = section.text.split('\n')
        for i, line in enumerate(lines):
            # print("here", i, line=="",len(line))
            if line == "":
                total_length += 1
                continue
            out_word_file.write(line.strip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir",type=str,default="../data/train_xml/",
        help="Enter the directory with trailing slash containing the files to be annotated")
    parser.add_argument("--test_data_dir",type=str,default="../data/unannotated_xml/",
        help="Enter the directory with trailing slash containing the files to be annotated")
    parser.add_argument("--output_tags_f",type=str,default=".tags.txt",
        help="")
    parser.add_argument("--output_words_f",type=str,default=".words.txt",
        help="")
    args = parser.parse_args()
    file_name = "ADCETRIS.xml"

    args.output_words_file = "train" + args.output_words_f
    args.output_tags_file = "train" + args.output_tags_f
    args.data_dir = Path(args.train_data_dir)

    with open(Path(args.data_dir.parent / args.output_tags_file), mode="w+", encoding='utf-8') as out_tag_file, \
            open(Path(args.data_dir.parent / args.output_words_file), mode="w+", encoding='utf-8') as out_word_file:
        for file in Path(args.data_dir).glob("**/*.xml"):
            if file.is_file():
                annotate_train_file(args, file, out_tag_file, out_word_file)


    args.output_words_file = "test" + args.output_words_f
    args.output_tags_file = "test" + args.output_tags_f
    args.data_dir = Path(args.test_data_dir)

    with open(Path(args.data_dir.parent / args.output_tags_file), mode="w+", encoding='utf-8') as out_tag_file, \
            open(Path(args.data_dir.parent / args.output_words_file), mode="w+", encoding='utf-8') as out_word_file:
        for file in Path(args.data_dir).glob("**/*.xml"):
            if file.is_file():
                annotate_test_file(args, file, out_tag_file, out_word_file)




if __name__ == "__main__":
    main()
