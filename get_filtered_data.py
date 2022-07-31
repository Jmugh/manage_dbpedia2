import json
from sklearn.preprocessing import MultiLabelBinarizer
from impose_label_relation.common import clean_str,clean_entityname
'''
    读取数据集文件，获取摘要，实体名，类别type
    替换不用的词，特殊字符，大小写转换，过滤类别数小于20的type
'''
datasets_length=50000
text_length=500
from impose_label_relation.config import Config
cfg=Config()

#获取abstract 和  types
def get_abstract_types(filename=cfg.datasets_path):
    print("start to read dataset ...")
    min_frequency=20
    cleaned_entitynames=[]
    abstracts=[]
    types=[]
    with open(filename,"r",encoding="utf8") as reader:
        for index,line in enumerate(reader):
            # if index>=100:
            #     break
            line=line.strip()
            json_line=json.loads(line)
            entityname=json_line["ENTITY"]
            abstract=json_line["ABSTRACT"]
            type=json_line["TYPES"]
            abstract=clean_str(abstract)
            abstract=clean_str(abstract)
            if len(abstract.split())>text_length:
                abstract=" ".join(abstract.split()[:text_length])
            cleaned_entitynames.append(clean_entityname(entityname))
            abstracts.append(abstract)
            types.append(type)
        abstracts=abstracts[:datasets_length]
        types=types[:datasets_length]
        cleaned_entitynames=cleaned_entitynames[:datasets_length]
        #处理types 1.删除低频label 2.判断是否所有label同时出现在训练集测试集
        label_dict=get_label_dict(types)
        cleaned_types=[]
        for type_list in types:
            temp_list=[]
            for type in type_list:
                if type in label_dict.keys() and label_dict[type]>min_frequency:#
                    temp_list.append(type)
            cleaned_types.append(temp_list)
        all_label_frequency=0
        selected_label_frequency=0
        for label in label_dict.keys():
            if label_dict[label]>min_frequency:
                selected_label_frequency+=label_dict[label]
            all_label_frequency+=label_dict[label]
        print("标签频率大于：", min_frequency)
        print("处理后多少个标签：",selected_label_frequency)
        print("原数据集多少个标签：",all_label_frequency)
        print("处理后共多少种标签：",len(get_label_dict(cleaned_types).keys()))
        print("原数据集多少种标签：",len(get_label_dict(types).keys()))
        print("训练集多少种标签：",len(get_label_dict(cleaned_types[:int(0.8*len(cleaned_types))]).keys()))
        print("测试集多少种标签：",len(get_label_dict(cleaned_types[int(0.8*len(cleaned_types)):]).keys()))

        print("read successfully ...")

    return abstracts,cleaned_types,cleaned_entitynames

def get_label_dict(types):#获取当前需要的数据的types
    label_dict = {}
    for label_list in types:
        for label in label_list:
            if label in label_dict.keys():
                label_dict[label] += 1
            else:
                label_dict[label] = 1
    return label_dict
def get_type_dict_idx():
    type_list=['AcademicJournal','Activity','Actor','AdministrativeRegion','Agent','Aircraft','Airline','Airport','Album','AmericanFootballPlayer','AmusementParkAttraction','AnatomicalStructure','Animal','Anime','Arachnid','Architect','ArchitecturalStructure','Artist','Artwork','Athlete','AustralianRulesFootballPlayer','Automobile','Award','Bacteria','BadmintonPlayer','Band','Bank','BaseballPlayer','BasketballPlayer','BasketballTeam','BeautyQueen','Biomolecule','BodyOfWater','Book','Boxer','Bridge','Broadcaster','Building','BusCompany','Cardinal','Cartoon','Case','Castle','CelestialBody','ChemicalCompound','ChemicalSubstance','ChessPlayer','ChristianBishop','City','CityDistrict','Cleric','ClericalAdministrativeRegion','Coach','CollegeCoach','Comic','ComicsCharacter','ComicsCreator','Company','Congressman','Convention','Country','Crater','Cricketer','Criminal','Crustacean','CultivatedVariety','Curler','Cyclist','Dam','Device','Diocese','Disease','Drug','Economist','EducationalInstitution','Election','Enzyme','EthnicGroup','Eukaryote','Event','FictionalCharacter','FigureSkater','Film','Fish','Food','FootballLeagueSeason','FootballMatch','Fungus','GaelicGamesPlayer','Galaxy','Game','Glacier','GolfPlayer','GolfTournament','GovernmentAgency','Governor','GrandPrix','GridironFootballPlayer','Group','Gymnast','HandballPlayer','HistoricBuilding','HistoricPlace','Horse','HorseRace','Hospital','Hotel','IceHockeyPlayer','InformationAppliance','Infrastructure','Insect','Island','Language','LegalCase','Legislature','Lighthouse','Locomotive','Magazine','Mammal','Manga','MartialArtist','MeanOfTransportation','Medician','MilitaryConflict','MilitaryPerson','MilitaryStructure','MilitaryUnit','Model','Mollusca','MotorcycleRider','MotorsportRacer','MotorsportSeason','Mountain','Museum','Musical','MusicalArtist','MusicalWork','NCAATeamSeason','NationalFootballLeagueSeason','NaturalPlace','Newspaper','Noble','OlympicEvent','Olympics','Organisation','Park','PeriodicalLiterature','Person','Philosopher','Place','Planet','Plant','Play','PoliticalParty','Politician','PopulatedPlace','PowerStation','President','ProtectedArea','PublicTransitSystem','Publisher','Race','RacingDriver','RadioProgram','RadioStation','RailwayLine','RailwayStation','RecordLabel','Region','Religious','River','Road','RouteOfTransportation','Royalty','RugbyClub','RugbyPlayer','Saint','School','Scientist','Settlement','Ship','ShoppingMall','SiteOfSpecialScientificInterest','Skater','Skier','SoapCharacter','SoccerClub','SoccerClubSeason','SoccerLeague','SoccerManager','SoccerPlayer','SoccerTournament','SocietalEvent','Software','Song','SpaceMission','Species','SportsClub','SportsEvent','SportsLeague','SportsManager','SportsSeason','SportsTeam','SportsTeamSeason','Station','Stream','SupremeCourtOfTheUnitedStatesCase','Swimmer','TelevisionSeason','TelevisionShow','TelevisionStation','TennisPlayer','TennisTournament','Tournament','Tower','Town','TradeUnion','Train','UnitOfWork','University','Venue','VideoGame','Village','VolleyballPlayer','Weapon','Website','WinterSportPlayer','Work','Wrestler','WrestlingEvent','Writer','WrittenWork']
    type_to_idx={t:idx for idx,t in enumerate(type_list)}
    idx_to_type={idx:t for idx,t in enumerate(type_list)}
    return type_to_idx,idx_to_type
if __name__=="__main__":
    abstracts,types,cleaned_entitynames=get_abstract_types()
    train_length=int(0.8*len(types))
    train_types=types[:train_length]
    multiLabelBinarizer = MultiLabelBinarizer().fit(train_types)
    train_labels = multiLabelBinarizer.transform(train_types)
    print(multiLabelBinarizer.classes_)
'''
    20
    处理后多少个标签： 161179
    原数据集多少个标签： 162282
    处理后共多少种标签： 232
    原数据集多少种标签： 399
    训练集多少种标签： 232
    测试集多少种标签： 232
'''

'''
['AcademicJournal','Activity','Actor','AdministrativeRegion','Agent','Aircraft','Airline','Airport','Album','AmericanFootballPlayer','AmusementParkAttraction','AnatomicalStructure','Animal','Anime','Arachnid','Architect','ArchitecturalStructure','Artist','Artwork','Athlete','AustralianRulesFootballPlayer','Automobile','Award','Bacteria','BadmintonPlayer','Band','Bank','BaseballPlayer','BasketballPlayer','BasketballTeam','BeautyQueen','Biomolecule','BodyOfWater','Book','Boxer','Bridge','Broadcaster','Building','BusCompany','Cardinal','Cartoon','Case','Castle','CelestialBody','ChemicalCompound','ChemicalSubstance','ChessPlayer','ChristianBishop','City','CityDistrict','Cleric','ClericalAdministrativeRegion','Coach','CollegeCoach','Comic','ComicsCharacter','ComicsCreator','Company','Congressman','Convention','Country','Crater','Cricketer','Criminal','Crustacean','CultivatedVariety','Curler','Cyclist','Dam','Device','Diocese','Disease','Drug','Economist','EducationalInstitution','Election','Enzyme','EthnicGroup','Eukaryote','Event','FictionalCharacter','FigureSkater','Film','Fish','Food','FootballLeagueSeason','FootballMatch','Fungus','GaelicGamesPlayer','Galaxy','Game','Glacier','GolfPlayer','GolfTournament','GovernmentAgency','Governor','GrandPrix','GridironFootballPlayer','Group','Gymnast','HandballPlayer','HistoricBuilding','HistoricPlace','Horse','HorseRace','Hospital','Hotel','IceHockeyPlayer','InformationAppliance','Infrastructure','Insect','Island','Language','LegalCase','Legislature','Lighthouse','Locomotive','Magazine','Mammal','Manga','MartialArtist','MeanOfTransportation','Medician','MilitaryConflict','MilitaryPerson','MilitaryStructure','MilitaryUnit','Model','Mollusca','MotorcycleRider','MotorsportRacer','MotorsportSeason','Mountain','Museum','Musical','MusicalArtist','MusicalWork','NCAATeamSeason','NationalFootballLeagueSeason','NaturalPlace','Newspaper','Noble','OlympicEvent','Olympics','Organisation','Park','PeriodicalLiterature','Person','Philosopher','Place','Planet','Plant','Play','PoliticalParty','Politician','PopulatedPlace','PowerStation','President','ProtectedArea','PublicTransitSystem','Publisher','Race','RacingDriver','RadioProgram','RadioStation','RailwayLine','RailwayStation','RecordLabel','Region','Religious','River','Road','RouteOfTransportation','Royalty','RugbyClub','RugbyPlayer','Saint','School','Scientist','Settlement','Ship','ShoppingMall','SiteOfSpecialScientificInterest','Skater','Skier','SoapCharacter','SoccerClub','SoccerClubSeason','SoccerLeague','SoccerManager','SoccerPlayer','SoccerTournament','SocietalEvent','Software','Song','SpaceMission','Species','SportsClub','SportsEvent','SportsLeague','SportsManager','SportsSeason','SportsTeam','SportsTeamSeason','Station','Stream','SupremeCourtOfTheUnitedStatesCase','Swimmer','TelevisionSeason','TelevisionShow','TelevisionStation','TennisPlayer','TennisTournament','Tournament','Tower','Town','TradeUnion','Train','UnitOfWork','University','Venue','VideoGame','Village','VolleyballPlayer','Weapon','Website','WinterSportPlayer','Work','Wrestler','WrestlingEvent','Writer','WrittenWork']

'''