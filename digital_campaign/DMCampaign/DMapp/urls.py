from django.conf.urls import url
from DMapp import views
from django.urls import path

app_name = 'DMapp'

urlpatterns = [
    path('', views.dataUploadView.as_view(), name = 'DMC'),

]
