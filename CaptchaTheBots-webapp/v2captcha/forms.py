from django import forms
from django.conf import settings
from captcha.fields import ReCaptchaField
from captcha.widgets import ReCaptchaV3, ReCaptchaV2Checkbox
from .models import Post


class CaptchaForm(forms.Form):
    your_name = forms.CharField(label='Name', max_length=100)
    your_pass = forms.CharField(label='Password', max_length=100, widget=forms.PasswordInput())
    captcha = ReCaptchaField(public_key=settings.RECAPTCHA_PUBLIC_KEY, private_key=settings.RECAPTCHA_PRIVATE_KEY, widget=ReCaptchaV2Checkbox)

class UploadForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ('title', 'image')
