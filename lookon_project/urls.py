from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from lookon_app.views import index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
