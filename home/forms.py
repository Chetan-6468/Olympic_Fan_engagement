from django import forms

class inputform(forms.Form):
    year = forms.CharField(label='Year', max_length=4)
    country = forms.CharField(label='Country',  max_length=50)
