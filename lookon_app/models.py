from django.db import models

class ImageClassification(models.Model):
    image = models.ImageField(upload_to='images')
    classification_result = models.CharField(max_length=1)
    prediction_probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'lookon_app'
