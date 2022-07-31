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
def test():
    import torch
    a=torch.tensor([[1,2,3],[4,5,6]])#1 2 3
                                   #4  5   6
    b=torch.tensor([[2,2,2],[3,3,3],[4,4,4]])

    a=a.unsqueeze(dim=1)
    print(a)
    print(torch.mul(a,b))

if __name__ == '__main__':
    test()