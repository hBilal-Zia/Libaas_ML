def closets_color(image):
    from colorthief import ColorThief
    import webcolors
    ct = ColorThief(image)
    dominant_color = ct.get_color(quality=1)

    # Function to calculate the Euclidean distance between two RGB colors
    def euclidean_distance(rgb1, rgb2):
        return sum((c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)) ** 0.5

    # Find the closest CSS3 color name
    min_distance = float('inf')
    closest_color_name = None
    for color_hex, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
        r, g, b = webcolors.hex_to_rgb(color_hex)
        distance = euclidean_distance(dominant_color, (r, g, b))
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color_name

    return closest_color_name


def remove_bg(image):
    from rembg import remove
    from PIL import Image
    file_name = 'bgremoved'
    input_image = Image.open(image)
    input_image.save(file_name + '.jpg')
    output = remove(input_image)
    output.save(file_name + '.png')
    with open(file_name + '.png', "rb") as f:
        output_bytes = f.read()
        return output_bytes


def predict_features(img):
    import joblib

    from keras.preprocessing import image
    from keras.models import load_model
    from PIL import Image
    import io
    import numpy as np
    # model1 = joblib.load(open('wear_category.pkl', 'rb'))
    # model2 = joblib.load(open('style_category.pkl', 'rb'))
    # model3 = joblib.load(open('season_category.pkl', 'rb'))
    model1 = load_model("models/wear_model.h5")
    model2 = load_model('models/my_model2.h5')
    model3 = load_model('models/my_model3.h5')
    model4 = load_model('models/gender_model.h5')
    article_model = load_model('models/article_model.h5')
    img_pil1 = Image.open(img)
    img_resized1 = img_pil1.resize((60, 80))
    img_array1 = image.img_to_array(img_resized1)
    img_array1 = np.expand_dims(img_array1, axis=0)
    img_array1 /= 255.0

    img_pil2 = Image.open(img)

    img_resized2 = img_pil2.resize((224, 224))
    img_array2 = image.img_to_array(img_resized2)
    img_array2 = np.expand_dims(img_array2, axis=0)
    img_array2 /= 255.0
    # img = image.img_to_array(img_pil)

    wear_type = model1.predict(img_array2)
    style_type = model2.predict(img_array1)
    season_type = model3.predict(img_array1)
    gender_type = model4.predict(img_array2)
    article_type = article_model.predict(img_array2)

    # print(wear_type)
    wear_class_index = np.argmax(wear_type)
    style_class_index = np.argmax(style_type)
    season_class_index = np.argmax(season_type)
    gender_class_index = np.argmax(gender_type)
    articel_class_index = np.argmax(article_type)

    wear_labels = {0: 'Apparel Set', 1: 'Bottomwear', 2: 'Dress', 3: 'Saree', 4: 'Topwear'}
    style_labels = ['Casual', 'Ethnic', 'Sports', 'Formal', 'Travel', 'Smart Casual',
                    'Party']
    season_labels = {0: 'Fall', 1: 'Spring', 2: 'Summer', 3: 'Winter'}
    gender_labels = {0: 'Men', 1: 'Unisex', 2: 'Women'}
    article_labels = {0: 'flats', 1: 'heels', 2: 'jacket', 3: 'pants', 4: 'shirt', 5: 'shoes', 6: 'shorts', 7: 'skirt', 8: 'sneakers', 9: 'tshirt'}


    predicted_wear_label = wear_labels[wear_class_index]
    predicted_style_label = style_labels[style_class_index]
    predicted_season_label = season_labels[season_class_index]
    predicted_gender_label = gender_labels[gender_class_index]
    predicted_article_label = article_labels[articel_class_index]

    return {"Article": predicted_article_label,"Wear Type": predicted_wear_label, "Style": predicted_style_label, "Weather": predicted_season_label,
            "Gender": predicted_gender_label}


def main():
    pass


if __name__ == '__main__':
    main()