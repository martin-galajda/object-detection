from datetime import datetime
import os

# -*- coding: utf-8 -*-

# Scrapy settings for collect_urls_for_annotation project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://doc.scrapy.org/en/latest/topics/settings.html
#     https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://doc.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'collect_annotable_urls'

SPIDER_MODULES = ['scraper.collect_annotable_urls.spiders']
NEWSPIDER_MODULE = 'scraper.collect_annotable_urls.spiders'


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'collect_urls_for_annotation (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://doc.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
# DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
# }

# Enable or disable spider middlewares
# See https://doc.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    'scraper.collect_urls_for_annotation.middlewares.limit_urls_by_domain_limit.LimitUrlsByDomainLimitMiddleware': 543,
# }

# Enable or disable downloader middlewares
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
   'scraper.collect_annotable_urls.middlewares.limit_urls_by_domain_limit.LimitUrlsByDomainLimitMiddleware': 543,
}

# Enable or disable extensions
# See https://doc.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
# }

# Configure item pipelines
# See https://doc.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'scraper.collect_annotable_urls.pipelines.matched_by_config.MatchedByConfigPipeline': 300,
}

curr_iso_timestamp = datetime.now().replace(microsecond=0).isoformat()
PATH_TO_PROJECT = '/Users/martingalajda/School/DIPLOMA-THESIS/object-detection' \
    if 'ENV' in os.environ and os.environ['ENV'] == 'local' \
    else '/storage/brno3-cerit/home/marneyko/object-detection'

FEED_URI = f'file://{PATH_TO_PROJECT}/scraper/out/collect_annotable_urls/output-{curr_iso_timestamp}-export.csv'
FEED_FORMAT = 'csv'
FEED_EXPORT_FIELDS = [
    'page_url',
    'domain',
    'url_matched_for_gallery_page',
    'url_matched_for_page_with_gallery',
    'number_of_images_by_img_src',
    'number_of_images_by_a_href',
    'has_match',
    'total_img_elements_found'
]
FEED_EXPORTERS = {
    'csv': 'scrapy.exporters.CsvItemExporter'
}
CONCURRENT_REQUESTS = 100
REACTOR_THREADPOOL_MAXSIZE = 20

LIMIT_PER_DOMAIN = {
    'fotografovani.cz': 10000,
    'extra.cz': 10000,
    'idnes.cz': 10000,
    'lifee.cz': 10000,
    'frekvence1.cz': 10000,
    'prozeny.blesk.cz': 10000,
    'blesk.cz': 10000,
    'sme.sk': 10000,
    'fotky.sme.sk': 10000,
    'kafe.cz': 10000,
    'reflex.cz': 10000,
    'vitalia.cz': 10000,
    'jenzeny.cz': 10000,
    'receptnajidlo.cz': 10000,
    'recepty.cz': 10000,
    'extralife.cz': 10000,
    'videacesky.cz': 10000,
    'televizeseznam.cz': 10000,
    'fashionmagazin.cz': 10000,
    'emefka.sk': 10000,
    'sport.cz': 10000,
    'lepsija.cz': 10000,
    'modernibyt.cz': 10000
}

# AUTOTHROTTLE_ENABLED = True
# AUTOTHROTTLE_START_DELAY = 5
# AUTOTHROTTLE_MAX_DELAY = 60
DEPTH_LIMIT=3

SCHEDULER_DEBUG = True
# Enable and configure the AutoThrottle extension (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://doc.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
