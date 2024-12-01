from django.urls import path
from home import views
from .views import ForecastAPIView,PredictWaterLevelAPIView, IrriSchedAPIView
urlpatterns=[
    # path("",views.index,name='home'),
    path('forecast',ForecastAPIView.as_view(),name="forecast"),
    path('predict',PredictWaterLevelAPIView.as_view(),name="predict"),
    path('scheduler',IrriSchedAPIView.as_view(),name="scheduler"),
]
