import spacy

#Carregando o modelo
nlp = spacy.load("modelo_ner")

while True:
    feedback = input("Insira seu feedback: ")
    print("Escreva (q) para sair.")
    print()
    if (feedback == "q"):
        break
    else:
        doc = nlp(feedback)
        for item in doc.ents:
            print(f"Palavra: {item.text}, Entidade: {item.label_}")


#Resultados podem não ser totalmente precisos, pois seria necessário a criação de um DataFrame com diversas
#frases e adjetivos únicos para ser absorvido pela máquina, ele consegue identificar palavras-chaves e associa-las
#TESTES COLETADOS COM O CHATGPT PARA SIMULAR UMA ENTRADA E AS INFORMAÇÕES TAMBÉM FORAM DO CHATGPT