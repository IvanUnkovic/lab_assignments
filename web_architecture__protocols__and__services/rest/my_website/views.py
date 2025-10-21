from .models import Post, Number
from .serializers import PostSerializer, UserSerializer, NumberSerializer
from rest_framework import status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import render

all_numbers = {}

@api_view(['PUT'])
def create_number(request, id):
    if request.method == "PUT":
        number = int(request.body.decode())
        all_numbers[id] = number
        return HttpResponse(f"{id}: {number}")
    else:
        return HttpResponse("Error", status=status.HTTP_400_BAD_REQUEST)
    
@api_view(['GET'])
def numbers(request):
    if request.method == "GET":
        return HttpResponse(f",".join([str(broj) for broj in all_numbers.values()]))

@api_view(['GET'])
def zbroj(request):
    if request.method == "GET":
        return HttpResponse(sum(all_numbers.values()))
    
def api_documentation(request):
    return render(request, 'api_documentation.html')


@api_view(['GET', 'POST', 'OPTIONS'])
@permission_classes([permissions.AllowAny])
def post_list(request):
    if request.method == 'GET':
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    if request.method == 'POST':
        if not request.user.is_authenticated:
            return Response({"error": "Authentication required"}, status=status.HTTP_401_UNAUTHORIZED)
        data = request.data.copy()
        data['user'] = request.user.id
        serializer = PostSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        
@api_view(['GET', 'PUT', 'DELETE', 'OPTIONS'])
def post_detail(request, user_id, post_id):
    try:
        post = Post.objects.get(pk=post_id, user_id=user_id)
    except Post.DoesNotExist:
        return Response({"error": "Post not found"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = PostSerializer(post)
        return Response(serializer.data)

    if request.method in ['PUT', 'DELETE']:
        if request.user != post.user and not request.user.is_superuser:
            return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)
        if request.method == 'PUT':
            serializer = PostSerializer(post, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        elif request.method == 'DELETE':
            post.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'OPTIONS'])
@permission_classes([permissions.IsAdminUser])
def user_list(request):

    if request.method == 'GET':
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

@api_view(['GET', 'PUT', 'DELETE', 'OPTIONS'])
def user_detail(request, user_id):
    try:
        user = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = UserSerializer(user)
        return Response(serializer.data)

    if request.user != user and not request.user.is_superuser:
        return Response({"error": "Permission denied"}, status=status.HTTP_403_FORBIDDEN)

    if request.method == 'PUT':
        serializer = UserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    if request.method == 'DELETE':
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

@api_view(['GET', 'OPTIONS'])
@permission_classes([permissions.AllowAny])
def user_posts(request, user_id):
    try:
        user = User.objects.get(pk=user_id)
    except User.DoesNotExist:
        return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    posts = user.posts.all()
    serializer = PostSerializer(posts, many=True)
    return Response(serializer.data)
