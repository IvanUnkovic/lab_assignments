from django.contrib.auth.models import User
from django.db import models

class Post(models.Model):
    title = models.CharField(max_length=200)
    description = models.CharField(max_length=500)
    user = models.ForeignKey(User, related_name='posts', on_delete=models.CASCADE)
    def __str__(self):
        return self.title
    
class Number(models.Model):
    broj = models.IntegerField()
    def __str__(self):
        return self.broj