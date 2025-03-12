from django.shortcuts import render

# Create your views here.
# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
#from sklearn.tree import export_graphviz #plot tree
#from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
##from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = DMCForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_ad= request.POST.get('AdSpend')
            data_ctr=request.POST.get('ClickThroughRate')
            data_cr=request.POST.get('ConversionRate')
            data_wv=request.POST.get('WebsiteVisits')
            data_ppv=request.POST.get('PagesPerVisit')
            data_ts=request.POST.get('TimeOnSite')
            data_ss=request.POST.get('SocialShares')
            data_eo=request.POST.get('EmailOpens')
            data_ec=request.POST.get('EmailClicks')
            data_pp=request.POST.get('PreviousPurchases')
            data_lp=request.POST.get('LoyaltyPoints')
            data_ctc=request.POST.get('CampaignType_Consideration')
            #print (data)
            #dataset1=pd.read_csv("prep.csv",index_col=None)
            #dicc={'yes':1,'no':0}
            import joblib,json
            lrc_model = joblib.load('lrc_model.pkl')
            xgb_model = joblib.load('xgb_model.pkl')
            with open('blending_params.json', 'r') as f:
                best_params = json.load(f)

            best_w1 = best_params['best_w1']
            best_w2 = best_params['best_w2']
            best_threshold = best_params['best_threshold']
            data = np.array([data_ad, data_ctr, data_cr, data_wv, data_ppv, data_ts,data_ss, data_eo, data_ec, data_pp, data_lp, data_ctc], dtype=float).reshape(1, -1)
            #sc = StandardScaler()
            #data = sc.fit_transform(data.reshape(-1,1))
            LRC_probs1 = lrc_model.predict_proba(data)[:, 1]
            XGB_probs1 = xgb_model.predict_proba(data)[:, 1]
            final_sample_probs = (best_w1 * LRC_probs1) + (best_w2 * XGB_probs1)
            out = (final_sample_probs > best_threshold).astype(int)
            #out=classifier.predict(data.reshape(1,-1))
# providing an index
            #ser = pd.DataFrame(data, index =['bgr','bu','sc','pcv','wbc'])

            #ss=ser.T.squeeze()
#data_for_prediction = X_test1.iloc[0,:].astype(float)

#data_for_prediction =obj.pca(np.array(data_for_prediction),y_test)
            #obj=ckd()
            ##plt.savefig("static/force_plot.png",dpi=150, bbox_inches='tight')







            return render(request, "succ_msg.html", {'data_ad':data_ad,'data_ctr':data_ctr,'data_cr':data_cr,'data_wv':data_wv,'data_ppv':data_ppv,'data_ts':data_ts,'data_ss':data_ss,'data_eo':data_eo,'data_ec':data_ec,'data_pp':data_pp,'data_lp':data_lp,'data_ctc':data_ctc,
                                                        'out':out})


        else:
            return redirect(self.failure_url)
