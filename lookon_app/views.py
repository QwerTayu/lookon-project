from django.shortcuts import render
from django.core.files.storage import default_storage
from .models import ImageClassification
import cv2
import numpy as np
from keras.models import load_model

IMG_SIZE = 64
model = load_model('my_model.h5')

def index(request):
    if request.method == 'POST':
        # 画像のアップロード処理
        image_file = request.FILES['image']
        image_path = default_storage.save('images/' + image_file.name, image_file)

        # 画像の読み込みと分類
        image_selected = cv2.imread(default_storage.path(image_path))
        image_selected = cv2.resize(image_selected, (IMG_SIZE, IMG_SIZE))
        image_selected = np.expand_dims(image_selected, axis=0)
        image_selected = image_selected / 255.0
        prediction = model.predict(image_selected)
        prob = prediction[0, 0]

        # 分類結果の判定
        if prob >= 0.9:
            classification_result = 'A'
        elif prob >= 0.8:
            classification_result = 'B'
        elif prob >= 0.6:
            classification_result = 'C'
        elif prob >= 0.3:
            classification_result = 'D'
        elif prob >= 0.1:
            classification_result = 'E'
        else:
            classification_result = 'F'

        # 分類結果の保存
        image_classification = ImageClassification(
            image=image_path,
            classification_result=classification_result,
            prediction_probability=prob
        )
        image_classification.save()

        # レンダリング時に表示するデータを準備
        context = {
            'image_path': image_path,
            'classification_result': classification_result,
            'prediction_probability': prob
        }
    else:
        context = {}

    return render(request, 'lookon_app/index.html', context)
