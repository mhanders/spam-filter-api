from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # url(r'^$', 'spamfilter.views.home', name='home'),
    url(r'^trainham', 'spamfilter.views.train_ham', name='trainham'),
    url(r'^trainspam', 'spamfilter.views.train_spam', name='trainspam'),
    url(r'^classify', 'spamfilter.views.run_bayes', name='classify'),
    url(r'^test', 'spamfilter.views.test', name='test'),
    url(r'^admin/', include(admin.site.urls)),
)
