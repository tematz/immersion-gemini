import numpy as np
import pandas as pd
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyAJ0lPRzEtOMh0SucCVSWRkLbsEQOx_TxY"
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)
        
title = 'A próxima geração de IA para desenvolvedores e Google Workspace'
sample_text = ('Título: A próxima geração de IA para desenvolvedores e Google Workspace'
               '\n'
               'Artigo completo:\n'
               '\n'
               'Gemini API & Google AI Studio: uma maneira acessível de explorar e criar protótipos com aplicações de IA generativa')

embeddings = genai.embed_content(model='models/embedding-001', content=sample_text, title=title, task_type='RETRIEVAL_DOCUMENT')

print(embeddings)

DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."}
DOCUMENT2 = {
    "title": "Touchscreen",
    "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."}
DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

df = pd.DataFrame(documents)
df.columns = ['Title', 'Text']
print(df)

model = 'models/embedding-001'

def embed_fn(title, text):
    return genai.embed_content(model=model, content=text, title=title, task_type='RETRIEVAL_DOCUMENT')['embedding']

df['Embeddings'] = df.apply(lambda row: embed_fn(row["Title"], row["Text"]), axis=1)
print(df)

def generate_and_fetch_query(query, base, model):
    query_embedding = genai.embed_content(model=model, content=query, task_type='RETRIEVAL_QUERY')['embedding']
    
    scalar_products = np.dot(np.stack(df['Embeddings']), query_embedding)
    
    index = np.argmax(scalar_products)
    
    return df.iloc[index]['Text']

query = 'Como faço para trocar marchas em um carro do Google?'

stretch = generate_and_fetch_query(query, df, model)
print(stretch)

generation_config = {
    'temperature': 0.5, 'candidate_count': 1
}

prompt = f'Rewrite this text in a more relaxed way, without adding information that is not part of the text: {stretch}'

model_2 = genai.GenerativeModel('gemini-1.0-pro', generation_config=generation_config)
response = model_2.generate_content(prompt)
print(response.text)