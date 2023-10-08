import spacy #Biblioteca do PLN
import pandas as pd #Criação do DataFrame utilizando o feedback.txt
import random 
from spacy.training.example import Example #Criação da estrutura adequada para o PLN entender.
from spacy.training.example import offsets_to_biluo_tags #Verificação de tokens

#Leitura do Dataframe e armazenando os dados
fd = pd.read_table("feedback.txt", sep=".")
fd.to_html('tabela.html')
fd_text = fd["FEEDBACK"]
fd_ent = fd["AVALIAÇÃO"]
fd_start = fd["START"]
fd_end = fd["END"]

#Criação do modelo de treinamento, como se fosse uma ficha de academia
nlp = spacy.blank("pt")
ner = nlp.create_pipe("ner")
nlp.add_pipe("ner", name="custom_ner")


#Preenchendo o modelo com as frases do Dataframe para ele analisar e aprender.
training_data = []

for i in range(len(fd_text)):
    entities = [(fd_start[i], fd_end[i], fd_ent[i])]
    example = Example.from_dict(nlp.make_doc(fd_text[i].lower()), {"entities": entities})
    tags = spacy.training.offsets_to_biluo_tags(nlp.make_doc(fd_text[i].lower()), entities)
    training_data.append(example)
    tags = offsets_to_biluo_tags(nlp.make_doc(fd_text[i].lower()), entities)
    print(tags)

#Aplicando o treino
optimizer = nlp.begin_training()
for _ in range(30):
    losses = {}
    random.shuffle(training_data)
    for example in training_data:
        nlp.update([example], drop=0.2, losses=losses)
    print("Losses:", losses)

#Salvando o modelo
nlp.to_disk("modelo_ner")






