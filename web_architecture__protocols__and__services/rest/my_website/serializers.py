from rest_framework import serializers
from .models import Post, User, Number

class PostSerializer(serializers.ModelSerializer):
    post_id = serializers.IntegerField(source='id', read_only=True)
    class Meta:
        model = Post
        fields = ['post_id', 'title', 'description', 'user']

class UserSerializer(serializers.ModelSerializer):
    user_id = serializers.IntegerField(source='id', read_only=True)
    class Meta:
        model = User
        fields = ['user_id', 'username']

class NumberSerializer(serializers.ModelSerializer):
    class Meta:
        model = Number
        fields = ['id', 'N']