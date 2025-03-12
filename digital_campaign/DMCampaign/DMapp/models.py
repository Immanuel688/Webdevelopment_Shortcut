from django.db import models

# Create your models here.
class DMCModel(models.Model):

    AdSpend=models.FloatField()
    ClickThroughRate=models.FloatField()
    ConversionRate=models.FloatField()
    WebsiteVisits=models.FloatField()
    PagesPerVisit=models.FloatField()
    TimeOnSite=models.FloatField()
    SocialShares=models.FloatField()
    EmailOpens=models.FloatField()
    EmailClicks=models.FloatField()
    PreviousPurchases=models.FloatField()
    LoyaltyPoints=models.FloatField()
    CampaignType_Consideration=models.FloatField()
