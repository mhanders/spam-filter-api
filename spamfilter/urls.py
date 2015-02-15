from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    url(r'^$', 'spamfilter.views.home', name='home'),
    url(r'^upload', 'spamfilter.views.upload_file', name='upload'),
    url(r'^classify', 'spamfilter.views.runbayes', name='classify'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
)
