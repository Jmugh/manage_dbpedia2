import re
from config import Config
import time
cfg=Config()
'''
    自定义数据迭代器
'''
class Data_Loader(object):
    def __init__(self, X,Y,Z,batch_size=128):
        self.start = 0
        self.end = len(X)
        self.batch_size=batch_size
        self.X=X
        self.Y=Y
        self.Z=Z
    def __iter__(self):
        return self

    def __next__(self):
        if self.start < self.end:
            batch_x=self.X[self.start:self.start+self.batch_size]
            batch_y=self.Y[self.start:self.start+self.batch_size]
            batch_z=self.Z[self.start:self.start+self.batch_size]
            self.start += self.batch_size
            return batch_x,batch_y,batch_z
        else:
            self.start=0
            raise StopIteration
#传入tokenizer的字典{input_ids:tensor,token_type_ids:tensor,attention_mask:tensor}
class DataLoaderTokenizer(object):
    def __init__(self,input_ids,token_type_ids,attention_mask,y,batch_size=64):
        print("create dataloader...")
        self.input_ids=input_ids
        self.token_type_ids=token_type_ids
        self.attention_mask=attention_mask
        self.y=y
        self.start = 0
        self.end = len(y)
        self.batch_size=batch_size
    def __iter__(self):
        return self
    def __next__(self):
        if self.start<self.end:
            batch_input_ids=self.input_ids[self.start:self.start+self.batch_size]
            batch_token_type_ids=self.token_type_ids[self.start:self.start+self.batch_size]
            batch_attention_mask=self.attention_mask[self.start:self.start+self.batch_size]
            batch_y=self.y[self.start:self.start+self.batch_size]
            self.start+=self.batch_size
            return batch_input_ids,batch_token_type_ids,batch_attention_mask,batch_y
        else:
            self.start=0
            raise StopIteration

# load_word2vec
def load_vocab_vector(word2vec_path=cfg.word2vec_path):
    print("start to load word to vector ...")
    start=time.time()
    import gensim
    model=gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
    word_list=[]
    vector_list=[]
    word_list.append("PAD")
    vector_list.append([0.0 for i in range(cfg.embedding_dim)])
    for word,vec in zip(model.vocab,model.vectors):
        word_list.append(word)
        vector_list.append(vec)
    word_dict={t:i for i,t in enumerate(word_list)}
    end=time.time()
    print("load word to vector successfully, used time:",end-start,"s ...")
    return word_list,vector_list,word_dict

'''
    评估指标
'''
def loose_multilabel_evaluation(y_true,y_pred):
    #计算accuracy
    strict_accurate_count=0
    for i in range(len(y_true)):
        flag = True
        for j in range(len(y_true[0])):
            if y_pred[i][j] != y_true[i][j]:
                flag = False
        if flag:
            strict_accurate_count += 1
    if len(y_true)==0:
        accuracy=0
    else:
        accuracy = strict_accurate_count / len(y_true)
    #计算micro
    TP=0
    Ti=0
    Ti_head=0
    for i in range(len(y_true)):
        for j in range(len(y_true[0])):
            if y_pred[i][j]==1 and y_true[i][j]==1:
                TP+=1
            if y_pred[i][j]==1:
                Ti_head+=1
            if y_true[i][j]==1:
                Ti+=1
    if Ti_head==0:
        micro_precision=0
    else:
        micro_precision=TP/Ti_head
    if Ti==0:
        micro_recall=0
    else:
        micro_recall=TP/Ti
    if (micro_precision+micro_recall)==0:
        micro_f1=0
    else:
        micro_f1=2*micro_precision*micro_recall/(micro_precision+micro_recall)

    # 计算macro
    precision_list=[]
    recall_list=[]
    for i in range(len(y_true)):
        TP=0
        Ti=0
        Ti_head=0
        for j in range(len(y_true[0])):
            if y_pred[i][j] == 1 and y_true[i][j] == 1:
                TP += 1
            if y_pred[i][j] == 1:
                Ti_head += 1
            if y_true[i][j] == 1:
                Ti += 1
        if Ti_head==0:
            precision=0
        else:
            precision = TP / Ti_head
        if Ti==0:
            recall=0
        else:
            recall = TP / Ti
        precision_list.append(precision)
        recall_list.append(recall)
    if sum(precision_list)==0 and sum(recall_list)==0:
        macro_precision=0
        macro_recall=0
        macro_f1=0
    else:
        macro_precision=sum(precision_list)/len(precision_list)
        macro_recall=sum(recall_list)/len(recall_list)
        macro_f1=2*macro_precision*macro_recall/(macro_precision+macro_recall)

    # result_reader = "accuracy=" + str('%.5f' % accuracy) + "  micro_precision=" + str(
    #     '%.5f' % micro_precision) + "  micro_recall=" + str('%.5f' % micro_recall) + "  micro_f1=" + str(
    #     '%.5f' % micro_f1) + "  macro_precision=" + str('%.5f' % macro_precision) + "  macro_recall=" + str(
    #     '%.5f' % macro_recall) + "  macro_f1=" + str('%.5f' % macro_f1)
    # result_writer = str('%.5f' % accuracy) + "\t" + str('%.5f' % micro_precision) + "\t" + str('%.5f' % micro_recall) + "\t" + str(
    #     '%.5f' % micro_f1) + "\t" + str('%.5f' % macro_precision) + "\t" + str('%.5f' % macro_recall) + "\t" + str('%.5f' % macro_f1)

    result_reader = "accuracy=" + str('%.5f' % accuracy) + "  micro_f1=" + str( '%.5f' % micro_f1) +"  macro_f1=" + str('%.5f' % macro_f1)
    result_writer = str('%.5f' % accuracy) + "\t" + "\t" + str('%.5f' % micro_f1) + "\t" + str('%.5f' % macro_f1)
    return result_reader, result_writer

#清理数据
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """

    # 换掉特殊字符
    string = replace_special_letter(string)

    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " is", string)
    string = re.sub(r"\'m", " am", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n\'t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"cannot", "can not", string)
    string = re.sub(r"gonna", "going to", string)
    string = re.sub(r"wanna", "want to", string)
    string = re.sub(r"gotta", "got to", string)

    # 去掉数字+单位
    string = re.sub(r"[0-9][a-z]+", "", string)
    # 去掉小数
    string = re.sub(r"[0-9]+.[0-9]+", "", string)


    string = re.sub(r"\'", "", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    #spacy 会分词 i  d  tokenizer 不会分
    string = re.sub(r"( id )", "( identity )", string)
    # ima spacy 分成i  m  a
    string = re.sub(r" ima ", "", string)
    # spacy: dict_keys([i, m, dong, hyun, (, korean, hanja, korean, pronunciation, i, m, ., do,
    # tokens: ['[CLS]', 'im', 'dong', 'hyun', '(', 'korean', 'han', '##ja', 'korean', 'pronunciation', 'im', '.', 'do

    string = re.sub(r" im ", " i am ", string)



    # 如果.不应该是分句作用 那就替换掉成空格：split(".")的item.split() 长度小于3
    sentences = string.split(".")
    stack = []
    for sentence in sentences:
        if len(sentence.split()) < 3:
            stack.append(sentence)
        else:
            stack.append(sentence)
            stack.append(".")
    string = " ".join(stack)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
#去除开头的"#"
def clean_head_special_char(string):
    string = re.sub(r"#", "", string)
    return string


#替换特殊字符áåâãäàąćéèëęíïıñóōöőøūúüřšțłńşž
def replace_special_letter(string):
    string = re.sub(r"á", "a", string)
    string = re.sub(r"å", "a", string)
    string = re.sub(r"â", "a", string)
    string = re.sub(r"ã", "a", string)
    string = re.sub(r"ä", "a", string)
    string = re.sub(r"à", "a", string)
    string = re.sub(r"ą", "a", string)

    string = re.sub(r"ć", "c", string)

    string = re.sub(r"é", "e", string)
    string = re.sub(r"è", "e", string)
    string = re.sub(r"ë", "e", string)
    string = re.sub(r"ę", "e", string)

    string = re.sub(r"í", "i", string)
    string = re.sub(r"ï", "i", string)
    string = re.sub(r"ı", "i", string)

    string = re.sub(r"ñ", "n", string)
    string = re.sub(r"ó", "o", string)
    string = re.sub(r"ō", "o", string)
    string = re.sub(r"ö", "o", string)
    string = re.sub(r"ő", "o", string)
    string = re.sub(r"ø", "o", string)

    string = re.sub(r"ū", "u", string)
    string = re.sub(r"ú", "u", string)
    string = re.sub(r"ü", "u", string)

    string = re.sub(r"ř", "r", string)
    string = re.sub(r"š", "s", string)
    string = re.sub(r"ț", "t", string)
    string = re.sub(r"ł", "l", string)
    string = re.sub(r"ń", "n", string)
    string = re.sub(r"ş", "s", string)
    string = re.sub(r"ž", "z", string)
    return string

'''
    处理实体名称
'''
def clean_entityname(entityname):
    if entityname=="9 )":#处理错了的数据
        entityname="Trap"
    if "," in entityname:
        entityname=entityname.split(",")[0]
    entityname = " ".join(entityname.split("_"))
    entityname = re.sub(u"\\(.*?\\)", "", entityname)#去掉()及里面的东西
    entityname=re.sub(r"(),", " ", entityname)#去掉(  )
    entityname=entityname.lower()
    entityname=replace_special_letter(entityname)
    if "-" in entityname:
        entityname=" ".join(entityname.split("-"))
    return entityname
'''
    获取标签树的邻接矩阵
'''
def get_heirarchy_graph():
    print("directed matrix  from root to child")
    import dgl
    import torch
    child=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231]
    root=[146,"*",17,168,"*",121,159,109,136,97,16,"*",78,40,12,147,"*",147,227,147,19,121,"*",196,19,98,57,19,19,202,147,"*",139,231,19,172,144,16,159,50,"*",218,37,149,45,"*",19,50,179,179,147,"*",147,52,231,80,17,144,154,192,155,139,19,147,12,151,226,19,109,"*",51,"*",45,147,144,"*",31,"*",196,"*",4,226,227,12,"*",203,198,78,19,43,1,139,19,191,144,154,198,19,144,19,19,37,149,118,161,37,37,226,69,16,12,155,"*",41,144,214,121,146,12,54,19,"*",178,192,147,16,144,147,12,130,19,201,139,37,136,17,227,203,85,149,146,147,143,198,4,149,231,4,147,"*",43,78,231,144,147,149,109,154,149,57,57,198,130,227,36,172,204,57,155,147,205,172,109,147,197,19,50,74,147,155,121,37,149,226,226,80,197,203,199,200,19,213,79,227,136,192,"*",144,192,144,147,"*",144,201,109,32,113,19,227,227,36,19,213,198,16,179,144,121,"*",74,37,193,179,19,69,227,19,"*",19,198,147,227]
    #root指向child
    src=[]
    dst=[]
    for i in range(len(child)):
        if root[i]!="*":
            src.append(int(root[i]))
            dst.append(int(child[i]))
    g=dgl.graph((torch.tensor(src),torch.tensor(dst)))
    g=dgl.to_bidirected(g)
    g=dgl.add_self_loop(g)
    # A = g.adjacency_matrix().to_dense()
    # sa = sparse.csr_matrix(A)
    return g,src,dst
def get_label_level():
    labels=['AcademicJournal','Activity','Actor','AdministrativeRegion','Agent','Aircraft','Airline','Airport','Album','AmericanFootballPlayer','AmusementParkAttraction','AnatomicalStructure','Animal','Anime','Arachnid','Architect','ArchitecturalStructure','Artist','Artwork','Athlete','AustralianRulesFootballPlayer','Automobile','Award','Bacteria','BadmintonPlayer','Band','Bank','BaseballPlayer','BasketballPlayer','BasketballTeam','BeautyQueen','Biomolecule','BodyOfWater','Book','Boxer','Bridge','Broadcaster','Building','BusCompany','Cardinal','Cartoon','Case','Castle','CelestialBody','ChemicalCompound','ChemicalSubstance','ChessPlayer','ChristianBishop','City','CityDistrict','Cleric','ClericalAdministrativeRegion','Coach','CollegeCoach','Comic','ComicsCharacter','ComicsCreator','Company','Congressman','Convention','Country','Crater','Cricketer','Criminal','Crustacean','CultivatedVariety','Curler','Cyclist','Dam','Device','Diocese','Disease','Drug','Economist','EducationalInstitution','Election','Enzyme','EthnicGroup','Eukaryote','Event','FictionalCharacter','FigureSkater','Film','Fish','Food','FootballLeagueSeason','FootballMatch','Fungus','GaelicGamesPlayer','Galaxy','Game','Glacier','GolfPlayer','GolfTournament','GovernmentAgency','Governor','GrandPrix','GridironFootballPlayer','Group','Gymnast','HandballPlayer','HistoricBuilding','HistoricPlace','Horse','HorseRace','Hospital','Hotel','IceHockeyPlayer','InformationAppliance','Infrastructure','Insect','Island','Language','LegalCase','Legislature','Lighthouse','Locomotive','Magazine','Mammal','Manga','MartialArtist','MeanOfTransportation','Medician','MilitaryConflict','MilitaryPerson','MilitaryStructure','MilitaryUnit','Model','Mollusca','MotorcycleRider','MotorsportRacer','MotorsportSeason','Mountain','Museum','Musical','MusicalArtist','MusicalWork','NCAATeamSeason','NationalFootballLeagueSeason','NaturalPlace','Newspaper','Noble','OlympicEvent','Olympics','Organisation','Park','PeriodicalLiterature','Person','Philosopher','Place','Planet','Plant','Play','PoliticalParty','Politician','PopulatedPlace','PowerStation','President','ProtectedArea','PublicTransitSystem','Publisher','Race','RacingDriver','RadioProgram','RadioStation','RailwayLine','RailwayStation','RecordLabel','Region','Religious','River','Road','RouteOfTransportation','Royalty','RugbyClub','RugbyPlayer','Saint','School','Scientist','Settlement','Ship','ShoppingMall','SiteOfSpecialScientificInterest','Skater','Skier','SoapCharacter','SoccerClub','SoccerClubSeason','SoccerLeague','SoccerManager','SoccerPlayer','SoccerTournament','SocietalEvent','Software','Song','SpaceMission','Species','SportsClub','SportsEvent','SportsLeague','SportsManager','SportsSeason','SportsTeam','SportsTeamSeason','Station','Stream','SupremeCourtOfTheUnitedStatesCase','Swimmer','TelevisionSeason','TelevisionShow','TelevisionStation','TennisPlayer','TennisTournament','Tournament','Tower','Town','TradeUnion','Train','UnitOfWork','University','Venue','VideoGame','Village','VolleyballPlayer','Weapon','Website','WinterSportPlayer','Work','Wrestler','WrestlingEvent','Writer','WrittenWork']
    levels=[4,1,6,4,1,2,5,3,3,7,2,1,3,2,4,5,1,5,2,5,6,2,1,2,6,4,4,6,6,4,5,1,3,3,6,4,3,2,5,6,1,2,3,2,2,1,6,6,4,4,5,5,5,6,3,3,6,3,6,3,3,3,6,5,4,4,7,6,3,1,6,1,2,5,3,3,2,1,2,1,2,7,2,4,1,3,4,3,6,3,2,3,6,5,3,6,4,6,3,6,6,3,2,5,5,1,3,7,2,2,4,3,1,3,3,3,2,4,4,4,6,1,6,3,5,2,3,5,4,7,6,2,3,3,3,6,2,3,4,2,4,5,5,4,2,2,3,4,5,1,3,3,3,3,5,2,3,6,2,4,4,4,7,2,4,4,4,4,3,5,5,4,3,5,4,6,6,4,5,3,2,3,2,7,7,3,4,3,4,6,6,5,2,2,6,3,1,3,3,3,5,1,3,2,3,4,4,6,2,2,4,6,5,4,2,4,3,2,1,4,3,3,4,6,2,2,6,1,6,4,5,2]
    return labels,levels
if __name__=="__main__":
    string="I am sun  zhong   ."
    string= clean_str(string)
    print(string)
    sentences = [sentence for sentence in string.split(".")]
    print(sentences)