import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from keras.layers import Dense, Flatten, GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential, Model, load_model
import numpy as np
from numpy.linalg import norm
from PIL import Image
import requests
from io import BytesIO


cred = credentials.Certificate("libaasapp-firebase-adminsdk-uqyse-1f3020db8c.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'https://console.firebase.google.com/project/libaasapp/storage/libaasapp.appspot.com/files/~2Fitem'
})
db = firestore.client()

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])
for layer in base_model.layers:
    layer.trainable = False


def get_style(event):
    formal_events = ['Wedding', 'Valima', 'Business', 'Presentation', 'Convocation']
    casual_events = ['Eid', 'Party', 'Picnic', 'Friends Meetup', 'Shopping', 'Sport', 'Family Gathering', 'Hiking',
                     'Concerts', 'Outing']
    semi_formal_events = ['Mehndi/Mayon', 'Birthday', 'Anniversary']

    if event in formal_events:
        return "Formal"
    if event in casual_events:
        return "Casual"
    if event in semi_formal_events:
        return "Semi Formal"


def get_season(temperature):
    if temperature <= 25:
        return "Winter"
    return "Summer"


def get_embedding(img_url):
    image_response = requests.get(img_url)
    img = Image.open(BytesIO(image_response.content))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    expanded_img_array /= 255.0
    processed_img = preprocess_input(expanded_img_array)
    embedding = model.predict(processed_img).flatten()
    embedding = embedding / norm(embedding)
    return embedding



def get_clothing_item(item_id):

    clothe_ref = db.collection('clothes')
    clothe = clothe_ref.where(filter=FieldFilter('clotheId', '==', item_id)).stream()
    return clothe

def get_clothes(user_id, temperature, event, venue):
    clothes_ref = db.collection('clothes')
    clothes = (clothes_ref.where(filter=FieldFilter('userId', '==', user_id))
               .where(filter=FieldFilter('recentlyUsed', '==', False))).stream()
    filtered_clothes = []
    for cloth in clothes:
        cloth_dic = cloth.to_dict()
        if (cloth_dic['season'] == get_season(temperature) and cloth_dic['style'] == get_style(event)
                and cloth_dic['category'] in ['Topwear', 'Bottomwear', 'Footwear']):

            selected_clothe = {
                'clotheId': cloth_dic['clotheId'],
                'category': cloth_dic['category'],
                'imageUrl': cloth_dic['image'],
                'embedding': get_embedding(cloth_dic['image'])
            }
            #print(selected_clothe)
            filtered_clothes.append(selected_clothe)
        # print(cloth_dic)

    return filtered_clothes


def filter_by_wear_type(clothes):

    top_wears = [top_wear for top_wear in clothes if top_wear['category'] == 'Topwear']
    bottom_wears = [bottom_wear for bottom_wear in clothes if bottom_wear['category'] == 'Bottomwear']
    foot_wears = [foot_wear for foot_wear in clothes if foot_wear['category'] == 'Footwear']
    # print(topwear)
    return top_wears, bottom_wears, foot_wears


def get_recommendation(tops, bottoms, foots):
    outfits = []
    previous_score = 0.5
    for top in tops:
        for bottom in bottoms:
            for foot in foots:
                print('Making and scoring outfits')
                top_embedding = top['embedding'].reshape(1, -1)
                bottom_embedding = bottom['embedding'].reshape(1, -1)
                foot_embedding = foot['embedding'].reshape(1, -1)
                outfit_model = load_model('models/recommendation_model.h5')
                score = outfit_model.predict([top_embedding, bottom_embedding, foot_embedding])
                if score >= previous_score:
                    outfit = {
                        'topwearId': top['clotheId'],
                        'topImageUrl': top['imageUrl'],
                        'bottomwearId': bottom['clotheId'],
                        'bottomwearImageUrl': bottom['imageUrl'],
                        'footwearId': foot['clotheId'],
                        'footwearImageUrl': foot['imageUrl']
                    }
                    previous_score = score
                    outfits.append(outfit)
    return outfits


def main():
    pass


if __name__ == '__main__':
    main()

