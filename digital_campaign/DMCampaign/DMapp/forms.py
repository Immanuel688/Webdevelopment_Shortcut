from django import forms
from .models import *


class DMCForm(forms.ModelForm):
    class Meta():
        model=DMCModel
        fields=['AdSpend', 'ClickThroughRate', 'ConversionRate', 'WebsiteVisits', 'PagesPerVisit', 'TimeOnSite', 'SocialShares', 'EmailOpens', 'EmailClicks', 'PreviousPurchases', 'LoyaltyPoints', 'CampaignType_Consideration']
