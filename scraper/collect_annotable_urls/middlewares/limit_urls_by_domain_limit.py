import urlparse
from collections import defaultdict
from scrapy.exceptions import IgnoreRequest


class FilterDomainbyLimitMiddleware(object):
    def __init__(self, domains_to_filter):
        self.domains_to_filter = domains_to_filter
        self.counter = defaultdict(int)

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        spider_name = crawler.spider.name
        domains_to_filter = settings.get('DOMAINS_TO_FILTER')
        o = cls(domains_to_filter)
        return o

    def process_request(self, request, spider):
        parsed_url = urlparse.urlparse(request.url)
        if parsed_url.netloc in self.domains_to_filter:
            if self.counter.get(parsed_url.netloc, 0) < self.domains_to_filter[parsed_url.netloc]):
                self.counter[parsed_url.netloc] += 1
            else:
                raise IgnoreRequest()
