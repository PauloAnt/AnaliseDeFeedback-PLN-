
import seaborn as sbr #Cria o gráfico a partir do Dataframe pandas.
import pandas as pd #Pega as informações do feedback.txt e cria um Dataframe.
import matplotlib.pyplot as plt #Mostra uma interface com o gráfico criado.
import spacy

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

fd = pd.read_table("feedback.txt", sep=".")
fd.replace(["BOM", "RUIM"], [1, 0], inplace=True)
fd.to_html('tabela.html') #Cria um HTML com o Dataframe.


x = fd["FEEDBACK"]
y = fd["AVALIAÇÃO"]
X_train, X_teste, Y_train, Y_teste = train_test_split(x, y, test_size=0.2, random_state=300) #X_train é tabela X sem as respostas, X_teste é coluna com o gabarito, Y_train é as respostas do X_train e Y_teste é a quantidade de acertos. 


vec = TfidfVectorizer()
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_teste)

model = MultinomialNB()
model.fit(X_train_vec, Y_train)

Y_pred = model.predict(X_test_vec)
Y_true = Y_teste

accuracy = accuracy_score(Y_teste, Y_pred)
print("Resultados esperados:")
print(Y_pred)
print("Resultados obtidos:")
print(Y_true.values)
print(f'Efetividade: {accuracy}')






'''
sbr.set(style="whitegrid")
sbr.countplot(y="AVALIAÇÃO", data=fd, legend=False)

plt.xlabel("Contagem")
plt.ylabel("AVALIAÇÃO")


plt.show()
'''

