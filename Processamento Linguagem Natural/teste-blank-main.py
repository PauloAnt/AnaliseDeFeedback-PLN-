import spacy

#Carregando o modelo
nlp = spacy.load("modelo_ner")

feedback = [
    "Seu trabalho neste projeto foi excelente.",
    "Gostaria de elogiar sua habilidade de trabalho em equipe.",
    "Seu comprometimento em cumprir prazos é notável.",
    "Sua habilidade de comunicação é uma grande vantagem.",
    "Sua capacidade de se adaptar a mudanças é impressionante."
    "Seu desempenho neste projeto não atendeu às expectativas.",
    "Sua colaboração com a equipe deixou a desejar.",
    "Seus atrasos consistentes afetaram negativamente o cronograma da equipe.",
    "Sua comunicação precisa ser mais eficaz.",
    "Sua resistência a mudanças prejudicou a equipe em várias ocasiões."
]
#Testando
for i in range(5):
    frase = feedback[i]
    if frase[-1] == ".":
        frase = frase[:-1]
    doc = nlp(frase)
    for ent in doc.ents:
        print(ent.text, ent.label_)

#Resultados podem não ser totalmente precisos, pois seria necessário a criação de um DataFrame com diversas
#frases e adjetivos únicos para ser absorvido pela máquina, ele consegue identificar palavras-chaves e associa-las
#TESTES COLETADOS COM O CHATGPT PARA SIMULAR UMA ENTRADA E AS INFORMAÇÕES TAMBÉM FORAM DO CHATGPT