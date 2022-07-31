import json

filename="./datasets_abstract2_15w.txt"
filename="E:/python/code/mytasks/data/finall_file.txt"
def get_entity(filename=filename):
    with open(filename,"r",encoding="utf8") as reader:
        for line in reader:
            line=line.strip()
            json_line=json.loads(line)
            types=json_line["TYPES"]
            if(len(types)>5 and len(json_line["CATEGORY"])<4 and len(json_line["ABSTRACT"].split())>50):
                # print("entityname:"+json_line["ENTITY"]+"\n")
                # print("infobox:"+json_line["INFOBOX"]+"\n")
                # print("category:"+json_line["CATEGORY"]+"\n")
                # print("abstract:"+json_line["ABSTRACT"]+"\n")
                # print("types:"+json_line["TYPES"]+"\n")

                print(line)


if __name__ == '__main__':
    get_entity()