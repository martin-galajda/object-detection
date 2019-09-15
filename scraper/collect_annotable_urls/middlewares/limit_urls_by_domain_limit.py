from urllib.parse import urlparse
from collections import defaultdict
from scrapy.exceptions import IgnoreRequest
from scraper.collect_annotable_urls.spiders.main import CollectAnnotableUrlsSpider

MAX_VISITED_REQUIRED_MATCHED_RATIO = 10


class LimitUrlsByDomainLimitMiddleware(object):
    def __init__(self, limit_per_domain: int):
        self.limit_per_domain = limit_per_domain

        for domain_key in self.limit_per_domain.keys():
            self.limit_per_domain[domain_key] = int(self.limit_per_domain[domain_key])
        self.counter_visited = defaultdict(int)
        self.counter_matched = defaultdict(int)

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        limit_per_domain = settings.get('LIMIT_PER_DOMAIN')
        o = cls(limit_per_domain)
        return o

    def process_request(self, request, spider: CollectAnnotableUrlsSpider):
        parsed_url = urlparse(request.url)
        domain = parsed_url.netloc
        domain_without_www = domain.replace('www.', '')

        if domain_without_www in self.limit_per_domain:
            max_visits_domain = self.limit_per_domain[domain_without_www] * MAX_VISITED_REQUIRED_MATCHED_RATIO
            limit_by_visits_reached = self.counter_visited.get(domain_without_www, 0) >= max_visits_domain
            limit_by_matches_reached = self.counter_matched.get(domain_without_www, 0) >= self.limit_per_domain[domain_without_www]

            if limit_by_visits_reached or limit_by_matches_reached:
                spider.logger.info(f'process_request() filtering request for domain {domain_without_www}: '
                                   f'visited: {self.counter_visited[domain_without_www]}, '
                                   f'matched: {self.counter_matched[domain_without_www]}')
                raise IgnoreRequest()
        spider.logger.info(f'process_request() accepting request for domain {domain_without_www}: '
                           f'visited: {self.counter_visited[domain_without_www]}, '
                           f'matched: {self.counter_matched[domain_without_www]}')

        return None

    def process_response(self, request, response, spider: CollectAnnotableUrlsSpider):
        parsed_url = urlparse(response.url)
        domain = parsed_url.netloc

        domain_without_www = domain.replace('www.', '')
        if domain_without_www in self.limit_per_domain:

            max_visits_domain = self.limit_per_domain[domain_without_www] * MAX_VISITED_REQUIRED_MATCHED_RATIO
            limit_by_visits_reached = self.counter_visited.get(domain_without_www, 0) >= max_visits_domain
            limit_by_matches_reached = self.counter_matched.get(domain_without_www, 0) >= self.limit_per_domain[domain_without_www]
            if not limit_by_visits_reached and not limit_by_matches_reached:

                spider.logger.info(f'process_response(): Incrementing counter_visited for domain {domain_without_www}:'
                                   f'visited: {self.counter_visited[domain_without_www]}')
                self.counter_visited[domain_without_www] += 1

                spider.logger.info('process_response(): response: {0}'.format(response))
                analyze_results = spider.website_analyzer.analyze_website(response.url, response)

                if analyze_results['has_match']:
                    spider.logger.info(
                        f'process_response(): Incrementing counter_matched for domain {domain_without_www}:'
                        f'matched: {self.counter_matched[domain_without_www]}')

                    self.counter_matched[domain_without_www] += 1

                return response
            else:
                spider.logger.info(f'process_response() filtering request for domain {domain_without_www}:'
                                   f'visited: {self.counter_visited[domain_without_www]}, '
                                   f'matched: {self.counter_matched[domain_without_www]}')
                spider.logger.info(f'process_response(): is_limited_by_matches: {limit_by_matches_reached}:')
                spider.logger.info(f'process_response(): is_limited_by_visits: {limit_by_visits_reached}:')

                raise IgnoreRequest()
        spider.logger.info(f'process_response() accepting request for domain {domain_without_www}: '
                           f'visited: {self.counter_visited[domain_without_www]}, '
                           f'matched: {self.counter_matched[domain_without_www]}')

        return response
