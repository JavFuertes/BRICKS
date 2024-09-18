from django import forms

class WallDictForm(forms.Form):
    walls_dict = forms.CharField(
        label='Walls Dictionary',
        widget=forms.Textarea(attrs={'rows': 10, 'cols': 80}),
        help_text='Enter the walls dictionary as a Python dictionary string'
    )
