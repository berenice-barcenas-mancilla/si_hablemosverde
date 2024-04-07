import re
import pandas as pd
from deep_translator import GoogleTranslator
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

# Cargar el archivo CSV
csv_file = 'HablemosVerde.csv'
mydata = pd.read_csv(csv_file)

# Renombrar las columnas
mydata.columns = ['pr1', 'pr2','pr3','pr4','pr5']


# Función para traducir texto de español a inglés
# def translate_text(text):
#     return GoogleTranslator(source="es", target="en").translate(text)

def translate_text(text):
    if pd.isna(text):
        return text
    else:
        return GoogleTranslator(source="es", target="en").translate(text)

# Aplicar la traducción a las columnas
mydata['pr1'] = mydata['pr1'].apply(translate_text)
mydata['pr2'] = mydata['pr2'].apply(translate_text)
mydata['pr3'] = mydata['pr3'].apply(translate_text)
mydata['pr4'] = mydata['pr4'].apply(translate_text)
mydata['pr5'] = mydata['pr5'].apply(translate_text)
print(mydata)

# Función para limpiar el texto del archivo
def clean(text):
    # Removemos los caracteres y números que no se ocuparon
    text = re.sub('[^A-Za-z]+', ' ', text)
    return text

# Limpiar el texto en las columnas traducidas
mydata['pr1'] = mydata['pr1'].apply(clean)
mydata['pr2'] = mydata['pr2'].apply(clean)
mydata['pr3'] = mydata['pr3'].apply(clean)
mydata['pr4'] = mydata['pr4'].apply(clean)
mydata['pr5'] = mydata['pr5'].apply(clean)
print(mydata)

# Función para tokenizar y etiquetar el texto
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

# Función para lematizar el texto
wordnet_lemmatizer = WordNetLemmatizer()
def lematize(pos_data):
    lemma_rew = ""
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

# Tokenizar y etiquetar el texto en las columnas limpiadas
mydata['POS_tagged_pr1'] = mydata['pr1'].apply(token_stop_pos)
mydata['POS_tagged_pr2'] = mydata['pr2'].apply(token_stop_pos)
mydata['POS_tagged_pr3'] = mydata['pr3'].apply(token_stop_pos)
mydata['POS_tagged_pr4'] = mydata['pr4'].apply(token_stop_pos)
mydata['POS_tagged_pr5'] = mydata['pr5'].apply(token_stop_pos)

# Lematizar el texto tokenizado y etiquetado
mydata['Lemma_pr1'] = mydata['POS_tagged_pr1'].apply(lematize)
mydata['Lemma_pr2'] = mydata['POS_tagged_pr2'].apply(lematize)
mydata['Lemma_pr3'] = mydata['POS_tagged_pr3'].apply(lematize)
mydata['Lemma_pr4'] = mydata['POS_tagged_pr4'].apply(lematize)
mydata['Lemma_pr5'] = mydata['POS_tagged_pr5'].apply(lematize)

# Mostrar el DataFrame resultante
print(mydata[['pr1', 'Lemma_pr1']])
print(mydata[['pr2', 'Lemma_pr2']])
print(mydata[['pr3', 'Lemma_pr3']])
print(mydata[['pr4', 'Lemma_pr4']])
print(mydata[['pr5', 'Lemma_pr5']])

# Función para calcular subjetividad
def getSubjectivity(comentarios):
    return TextBlob(comentarios).sentiment.subjectivity

# Función para caluclar polaridad
def getPolarity(comentarios):
    return TextBlob(comentarios).sentiment.polarity

# Función para analizar los resultados
def analysis(score):
    if score < 0:
        return 'Negativo'
    elif score == 0:
        return 'Neutro'
    else:
        return 'Positivo'

# Crear un DataFrame para el análisis final
fin_data_pr1 = pd.DataFrame(mydata[['pr1', 'Lemma_pr1']])
fin_data_pr2 = pd.DataFrame(mydata[['pr2', 'Lemma_pr2']])
fin_data_pr3 = pd.DataFrame(mydata[['pr3', 'Lemma_pr3']])
fin_data_pr4 = pd.DataFrame(mydata[['pr4', 'Lemma_pr4']])
fin_data_pr5 = pd.DataFrame(mydata[['pr5', 'Lemma_pr5']])

# Calcular subjetividad y polaridad
fin_data_pr1['Subjetividad'] = fin_data_pr1['Lemma_pr1'].apply(getSubjectivity)
fin_data_pr1['Polaridad'] = fin_data_pr1['Lemma_pr1'].apply(getPolarity)
fin_data_pr1['Resultado'] = fin_data_pr1['Polaridad'].apply(analysis)

fin_data_pr2['Subjetividad'] = fin_data_pr2['Lemma_pr2'].apply(getSubjectivity)
fin_data_pr2['Polaridad'] = fin_data_pr2['Lemma_pr2'].apply(getPolarity)
fin_data_pr2['Resultado'] = fin_data_pr2['Polaridad'].apply(analysis)

fin_data_pr3['Subjetividad'] = fin_data_pr3['Lemma_pr3'].apply(getSubjectivity)
fin_data_pr3['Polaridad'] = fin_data_pr3['Lemma_pr3'].apply(getPolarity)
fin_data_pr3['Resultado'] = fin_data_pr3['Polaridad'].apply(analysis)

fin_data_pr4['Subjetividad'] = fin_data_pr4['Lemma_pr4'].apply(getSubjectivity)
fin_data_pr4['Polaridad'] = fin_data_pr4['Lemma_pr4'].apply(getPolarity)
fin_data_pr4['Resultado'] = fin_data_pr4['Polaridad'].apply(analysis)

fin_data_pr5['Subjetividad'] = fin_data_pr5['Lemma_pr5'].apply(getSubjectivity)
fin_data_pr5['Polaridad'] = fin_data_pr5['Lemma_pr5'].apply(getPolarity)
fin_data_pr5['Resultado'] = fin_data_pr5['Polaridad'].apply(analysis)

# Imprimir el DataFrame resultante para 
print("Resultado para: ¿Qué tan preocupado/a te sientes por los problemas ambientales y la falta de sostenibilidad en Querétaro?")
print(fin_data_pr1)
print("\nResultado para:¿Crees que la mayoría de las personas en tu comunidad están tomando medidas para adoptar prácticas más sostenibles? ¿Y qué opinas al respecto?")
print(fin_data_pr2)
print("\nResultado para:¿Qué acciones has implementado en tu vida diaria para reducir tu impacto ambiental y adoptar un estilo de vida más ecológico? ")
print(fin_data_pr3)
print("\nResultado para:¿Consideras que las autoridades y empresas están haciendo lo suficiente para promover la sostenibilidad y proteger el medio ambiente en Querétaro? ¿Por qué?")
print(fin_data_pr4)
print("\nResultado para:¿Consideras que un proyecto como Hablemos Verde tendría la capacidad de concientizar, motivar y apoyar a personas en tu comunidad a transitar hacia estilos de vida más ecológicos y sostenibles? ¿Por qué sí o por qué no?")
print(fin_data_pr5)

# Gráfica de los resultados para pr1
print("Gráfica de los resultados de ¿Qué tan preocupado/a te sientes por los problemas ambientales y la falta de sostenibilidad en Querétaro?:")
tb_counts_pr1 = fin_data_pr1['Resultado'].value_counts()
print(tb_counts_pr1)

plt.figure(figsize=(10,7))
plt.title("TextBlob ¿Qué tan preocupado/a te sientes por los problemas ambientales y la falta de sostenibilidad en Querétaro?")
plt.pie(tb_counts_pr1.values, labels=tb_counts_pr1.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()

# Gráfica de los resultados para pr2
print("Gráfica de los resultados para ¿Crees que la mayoría de las personas en tu comunidad están tomando medidas para adoptar prácticas más sostenibles? ¿Y qué opinas al respecto?:")
tb_counts_pr2 = fin_data_pr2['Resultado'].value_counts()
print(tb_counts_pr2)

plt.figure(figsize=(10,7))
plt.title("TextBlob ¿Crees que la mayoría de las personas en tu comunidad están tomando medidas para adoptar prácticas más sostenibles? ¿Y qué opinas al respecto?")
plt.pie(tb_counts_pr2.values, labels=tb_counts_pr2.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()

# Gráfica de los resultados para pr3
print("¿Qué acciones has implementado en tu vida diaria para reducir tu impacto ambiental y adoptar un estilo de vida más ecológico?:")
tb_counts_pr3 = fin_data_pr3['Resultado'].value_counts()
print(tb_counts_pr3)

plt.figure(figsize=(10,7))
plt.title("TextBlob ¿Qué acciones has implementado en tu vida diaria para reducir tu impacto ambiental y adoptar un estilo de vida más ecológico?")
plt.pie(tb_counts_pr3.values, labels=tb_counts_pr3.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()


# Gráfica de los resultados para pr4
print("Gráfica de los resultados para ¿Consideras que las autoridades y empresas están haciendo lo suficiente para promover la sostenibilidad y proteger el medio ambiente en Querétaro? ¿Por qué?:")
tb_counts_pr4 = fin_data_pr4['Resultado'].value_counts()
print(tb_counts_pr4)

plt.figure(figsize=(10,7))
plt.title("TextBlob ¿Consideras que las autoridades y empresas están haciendo lo suficiente para promover la sostenibilidad y proteger el medio ambiente en Querétaro? ¿Por qué?")
plt.pie(tb_counts_pr4.values, labels=tb_counts_pr4.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()


# Gráfica de los resultados para pr5
print("Gráfica de los resultados para ¿Consideras que un proyecto como Hablemos Verde tendría la capacidad de concientizar, motivar y apoyar a personas en tu comunidad a transitar hacia estilos de vida más ecológicos y sostenibles? ¿Por qué sí o por qué no?")
tb_counts_pr5 = fin_data_pr5['Resultado'].value_counts()
print(tb_counts_pr5)

plt.figure(figsize=(10,7))
plt.title("TextBlob ¿Consideras que un proyecto como Hablemos Verde tendría la capacidad de concientizar, motivar y apoyar a personas en tu comunidad a transitar hacia estilos de vida más ecológicos y sostenibles? ¿Por qué sí o por qué no?")
plt.pie(tb_counts_pr5.values, labels=tb_counts_pr5.index, explode=(0.1, 0, 0), autopct='%1.1f%%', shadow=False)
plt.show()