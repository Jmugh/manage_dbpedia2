from sklearn.preprocessing import MultiLabelBinarizer
from get_data import get_abstract_types
import wordninja
from config import Config
from transformers import BertTokenizer, BertModel
import torch
from common import get_heirarchy_graph
import dgl
cfg=Config()
tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)
bert_model = BertModel.from_pretrained(cfg.bert_path)
bert_model = bert_model.to(cfg.device)
def get_label_embedding():
    label_list=['AcademicJournal','Activity','Actor','AdministrativeRegion','Agent','Aircraft','Airline','Airport','Album','AmericanFootballPlayer','AmusementParkAttraction','AnatomicalStructure','Animal','Anime','Arachnid','Architect','ArchitecturalStructure','Artist','Artwork','Athlete','AustralianRulesFootballPlayer','Automobile','Award','Bacteria','BadmintonPlayer','Band','Bank','BaseballPlayer','BasketballPlayer','BasketballTeam','BeautyQueen','Biomolecule','BodyOfWater','Book','Boxer','Bridge','Broadcaster','Building','BusCompany','Cardinal','Cartoon','Case','Castle','CelestialBody','ChemicalCompound','ChemicalSubstance','ChessPlayer','ChristianBishop','City','CityDistrict','Cleric','ClericalAdministrativeRegion','Coach','CollegeCoach','Comic','ComicsCharacter','ComicsCreator','Company','Congressman','Convention','Country','Crater','Cricketer','Criminal','Crustacean','CultivatedVariety','Curler','Cyclist','Dam','Device','Diocese','Disease','Drug','Economist','EducationalInstitution','Election','Enzyme','EthnicGroup','Eukaryote','Event','FictionalCharacter','FigureSkater','Film','Fish','Food','FootballLeagueSeason','FootballMatch','Fungus','GaelicGamesPlayer','Galaxy','Game','Glacier','GolfPlayer','GolfTournament','GovernmentAgency','Governor','GrandPrix','GridironFootballPlayer','Group','Gymnast','HandballPlayer','HistoricBuilding','HistoricPlace','Horse','HorseRace','Hospital','Hotel','IceHockeyPlayer','InformationAppliance','Infrastructure','Insect','Island','Language','LegalCase','Legislature','Lighthouse','Locomotive','Magazine','Mammal','Manga','MartialArtist','MeanOfTransportation','Medician','MilitaryConflict','MilitaryPerson','MilitaryStructure','MilitaryUnit','Model','Mollusca','MotorcycleRider','MotorsportRacer','MotorsportSeason','Mountain','Museum','Musical','MusicalArtist','MusicalWork','NCAATeamSeason','NationalFootballLeagueSeason','NaturalPlace','Newspaper','Noble','OlympicEvent','Olympics','Organisation','Park','PeriodicalLiterature','Person','Philosopher','Place','Planet','Plant','Play','PoliticalParty','Politician','PopulatedPlace','PowerStation','President','ProtectedArea','PublicTransitSystem','Publisher','Race','RacingDriver','RadioProgram','RadioStation','RailwayLine','RailwayStation','RecordLabel','Region','Religious','River','Road','RouteOfTransportation','Royalty','RugbyClub','RugbyPlayer','Saint','School','Scientist','Settlement','Ship','ShoppingMall','SiteOfSpecialScientificInterest','Skater','Skier','SoapCharacter','SoccerClub','SoccerClubSeason','SoccerLeague','SoccerManager','SoccerPlayer','SoccerTournament','SocietalEvent','Software','Song','SpaceMission','Species','SportsClub','SportsEvent','SportsLeague','SportsManager','SportsSeason','SportsTeam','SportsTeamSeason','Station','Stream','SupremeCourtOfTheUnitedStatesCase','Swimmer','TelevisionSeason','TelevisionShow','TelevisionStation','TennisPlayer','TennisTournament','Tournament','Tower','Town','TradeUnion','Train','UnitOfWork','University','Venue','VideoGame','Village','VolleyballPlayer','Weapon','Website','WinterSportPlayer','Work','Wrestler','WrestlingEvent','Writer','WrittenWork']
    label_embeddings=[]
    for label_name in label_list:
        label = " ".join(wordninja.split(label_name)).lower()
        tokenized_dict = tokenizer(label, return_tensors="pt")
        tokenized_dict = tokenized_dict.to(cfg.device)
        outputs = bert_model(**tokenized_dict)
        last_hidden_states = outputs.last_hidden_state.detach().to("cpu")
        last_hidden_states=last_hidden_states.squeeze(dim=0)
        feat=torch.max(last_hidden_states,dim=0).values
        label_embeddings.append(feat.numpy().tolist())
    return label_embeddings



def get_entity_embedding_and_onehot_types():
    index=0
    abstracts, entitynames, cleaned_types = get_abstract_types()
    multiLabelBinarizer = MultiLabelBinarizer().fit(cleaned_types)
    onehot_types = multiLabelBinarizer.transform(cleaned_types).tolist()
    entity_embeddings=[]
    for abstract in abstracts:
        index+=1
        if(index%1000==0):
            print("tokenize:",index)
        output=tokenizer(abstract,return_tensors='pt', truncation=True, padding="max_length",max_length=cfg.sentence_len)
        output = output.to(cfg.device)
        output = bert_model(**output)
        output = output.last_hidden_state.detach().to("cpu")  # [batch_size,sequence_len,hidden_dim]
        cls=output[:,0].squeeze(dim=0)
        entity_embeddings.append(cls.numpy().tolist())
    return entity_embeddings,onehot_types

'''
    construct graph
'''
def construct_heterograph():
    # doc-->label
    src_doc_label=[]
    dst_doc_label=[]
    for s in range(50000):
        for t in range(232):
            src_doc_label.append(s)
            dst_doc_label.append(t)
    # label-->doc
    src_label_doc = []
    dst_label_doc = []
    for s in range(232):
        for t in range(50000):
            src_label_doc.append(s)
            dst_label_doc.append(t)
    # label-->label
    _,src_label_label,dst_label_label=get_heirarchy_graph()
    for i in range(232):
        src_label_label.append(i)
        dst_label_label.append(i)
    #doc-->doc
    src_doc_doc=[]
    dst_doc_doc=[]
    for i in range(50000):
        src_doc_doc.append(i)
        dst_doc_doc.append(i)

    graph_data = {
        ('doc', 'doc_label', 'label'):
            (torch.tensor(src_doc_label),
             torch.tensor(dst_doc_label)),
        ('label', 'label_doc', 'doc'):
            (torch.tensor(src_label_doc),
             torch.tensor(dst_label_doc)),
        ('label', 'label_label', 'label'):
            (torch.tensor(src_label_label),
             torch.tensor(dst_label_label)),
        ('doc', 'doc_doc', 'doc'):
            (torch.tensor(src_doc_doc),
             torch.tensor(dst_doc_doc))
    }

    label_embeddings=get_label_embedding()
    entity_embeddings,onehot_types=get_entity_embedding_and_onehot_types()

    g = dgl.heterograph(graph_data)
    g.nodes["doc"].data["label"] = torch.tensor(onehot_types)  # 设置节点的标签
    g.nodes["doc"].data["feats"] = torch.tensor(entity_embeddings)
    g.nodes["label"].data["feats"] = torch.tensor(label_embeddings)
    print(g)
    dgl.save_graphs(filename="./saved_graphs/hetero_graph.pth", g_list=[g])

if __name__ == '__main__':
    print()
    # get_label_embedding()
    # get_entity_embedding_and_onehot_types()
    # construct_heterograph()
    g=dgl.load_graphs("./saved_graphs/hetero_graph.pth")[0]
    print(g)
    print(g[0].nodes["doc"].data["feats"].shape)