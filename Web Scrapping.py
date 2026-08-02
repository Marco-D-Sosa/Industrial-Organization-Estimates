import pandas as pd
import requests
from bs4 import BeautifulSoup


# Prueba usando la pagina web "Quotes to Scrape"
url = 'https://quotes.toscrape.com'
url_actual = url + '/'
datos_extraidos = []
numero_pagina = 1

while True:
    print(f"Scrapeando página {numero_pagina}...")
    respuesta = requests.get(url_actual)
    soup = BeautifulSoup(respuesta.text, 'html.parser')
    contenedores = soup.find_all('div', class_='quote')
    for caja in contenedores:
        frase = caja.find('span', class_='text').text
        autor = caja.find('small', class_='author').text
        datos_extraidos.append({'Frase':frase, 'Autor':autor})
    
    boton_next = soup.find('li', class_='next')
    if boton_next:
        enlace_relativo = boton_next.find('a')['href']
        url_actual = url + enlace_relativo
        numero_pagina += 1
    else:
        print("No hay mas paginas")
        break #Corto el bucle while

print("------------------------------------")
print(f"Se extrajeron un total de {len(datos_extraidos)}")
df = pd.DataFrame(datos_extraidos)
print(df.head())
