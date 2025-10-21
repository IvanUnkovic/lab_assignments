from django.contrib import admin
from django.urls import path
from my_website import views

urlpatterns = [
    path('', views.api_documentation),
    path('admin', admin.site.urls),
    path('posts', views.post_list),
    path('users', views.user_list),
    path('users/<int:user_id>', views.user_detail),
    path('users/<int:user_id>/posts', views.user_posts),
    path('users/<int:user_id>/posts/<int:post_id>', views.post_detail),
    path('brojevi/<int:id>', views.create_number),
    path("brojevi", views.numbers),
    path("zbroj", views.zbroj)
]

