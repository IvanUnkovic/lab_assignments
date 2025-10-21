import base64
from django.test import TestCase
from rest_framework.test import APIClient
from django.contrib.auth.models import User
from .models import Post
import json

class APITest(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.admin_user = User.objects.create_superuser(username="admin", password="admin123")
        self.regular_user = User.objects.create_user(username="user1", password="user1")
        self.another_user = User.objects.create_user(username="user2", password="user2")

        self.post = Post.objects.create(title="Test post", description="Test description", user=self.regular_user)

        self.posts_url = "/posts"
        self.users_url = "/users"
        self.user_detail_url = f"/users/{self.regular_user.id}"
        self.user_posts_url = f"/users/{self.regular_user.id}/posts"
        self.post_detail_url = f"/users/{self.regular_user.id}/posts/{self.post.id}"

    def set_basic_auth(self, username, password):
        credentials = base64.b64encode(f"{username}:{password}".encode()).decode("utf-8")
        self.client.credentials(HTTP_AUTHORIZATION=f"Basic {credentials}")

    #GET /posts
    def test_get_posts(self):
        response = self.client.get(self.posts_url)
        self.assertEqual(response.status_code, 200)

    #POST /posts
    def test_post_post_admin(self):
        self.set_basic_auth("admin", "admin123")
        data = {"title": "new post", "description": "new description"}
        response = self.client.post(self.posts_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 201)

    def test_post_post_any_user(self):
        self.set_basic_auth("user1", "user1")
        data = {"title": "new post", "description": "new description"}
        response = self.client.post(self.posts_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 201)

    def test_post_post_unauthorized(self):
        data = {"title": "new post", "description": "new description"}
        response = self.client.post(self.posts_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 401)

    #GET /users
    def test_get_users_authorized(self):
        self.set_basic_auth("admin", "admin123")
        response = self.client.get(self.users_url)
        self.assertEqual(response.status_code, 200)

    def test_get_users_any_user(self):
        self.set_basic_auth("user1", "user1")
        response = self.client.get(self.users_url)
        self.assertEqual(response.status_code, 403)

    def test_get_users_unauthorized(self):
        response = self.client.get(self.users_url)
        self.assertEqual(response.status_code, 401)

    #PUT /users/<user_id>
    def test_put_user_detail_admin(self):
        self.set_basic_auth("admin", "admin123")
        data = {"username": "updated_user"}
        response = self.client.put(self.user_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)

    def test_put_user_detail_right_user(self):
        self.set_basic_auth("user1", "user1")
        data = {"username": "updated_user"}
        response = self.client.put(self.user_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)

    def test_put_user_detail_another_user(self):
        self.set_basic_auth("user2", "user2")
        data = {"username": "updated_user"}
        response = self.client.put(self.user_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 403)

    def test_put_user_detail_unauthorized(self):
        data = {"username": "updated_user"}
        response = self.client.put(self.user_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 401)

    #DELETE /users/<user_id>
    def test_delete_user_admin(self):
        self.set_basic_auth("admin", "admin123")
        response = self.client.delete(self.user_detail_url)
        self.assertEqual(response.status_code, 204)
        
    def test_delete_user_right_user(self):
        self.set_basic_auth("user1", "user1")
        response = self.client.delete(self.user_detail_url)
        self.assertEqual(response.status_code, 204)

    def test_delete_user_another_user(self):
        self.set_basic_auth("user2", "user2")
        response = self.client.delete(self.user_detail_url)
        self.assertEqual(response.status_code, 403)


    def test_delete_user_unauthorized(self):
        response = self.client.delete(self.user_detail_url)
        self.assertEqual(response.status_code, 401)

    #GET /users/<user_id>/posts
    def test_get_user_posts(self):
        response = self.client.get(self.user_posts_url)
        self.assertEqual(response.status_code, 200)

    #GET /users/<user_id>/posts/<post_id>
    def test_get_post_detail_admin(self):
        self.set_basic_auth("admin", "admin123")
        response = self.client.get(self.post_detail_url)
        self.assertEqual(response.status_code, 200)

    def test_get_post_detail_right_user(self):
        self.set_basic_auth("user1", "user1")
        response = self.client.get(self.post_detail_url)
        self.assertEqual(response.status_code, 200)

    def test_get_post_detail_another_user(self):
        self.set_basic_auth("user2", "user2")
        response = self.client.get(self.post_detail_url)
        self.assertEqual(response.status_code, 200)

    def test_get_post_detail_unauthorized(self):
        response = self.client.get(self.post_detail_url)
        self.assertEqual(response.status_code, 401)

    #PUT /users/<user_id>/posts/<post_id>
    def test_put_post_detail_admin(self):
        self.set_basic_auth("admin", "admin123")
        data = {"title": "updated post", "description": "I just updated my post"}
        response = self.client.put(self.post_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)

    def test_put_post_detail_right_user(self):
        self.set_basic_auth("user1", "user1")
        data = {"title": "updated post", "description": "I just updated my post"}
        response = self.client.put(self.post_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 200)

    def test_put_post_detail_another_user(self):
        self.set_basic_auth("user2", "user2")
        data = {"title": "updated post", "description": "I just updated my post"}
        response = self.client.put(self.post_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 403)

    def test_put_post_detail_unauthorized(self):
        data = {"title": "updated post", "description": "I just updated my post"}
        response = self.client.put(self.post_detail_url, data=json.dumps(data), content_type="application/json")
        self.assertEqual(response.status_code, 401)

    #DELETE /users/<user_id>/posts/<post_id>
    def test_delete_post_detail_admin(self):
        self.set_basic_auth("admin", "admin123")
        response = self.client.delete(self.post_detail_url)
        self.assertEqual(response.status_code, 204)

    def test_delete_post_detail_right_user(self):
        self.set_basic_auth("user1", "user1")
        response = self.client.delete(self.post_detail_url)
        self.assertEqual(response.status_code, 204)

    def test_delete_post_detail_another_user(self):
        self.set_basic_auth("user2", "user2")
        response = self.client.delete(self.post_detail_url)
        self.assertEqual(response.status_code, 403)

    def test_delete_post_detail_unauthorized(self):
        response = self.client.delete(self.post_detail_url)
        self.assertEqual(response.status_code, 401)