import scrapy

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from urllib.parse import urlparse
from scraper.collect_annotable_urls.load_configuration import load_configuration
from scraper.collect_annotable_urls.website_analyzer import WebsiteAnalyzer


configuration = load_configuration()


class CollectAnnotableUrlsSpider(CrawlSpider):
    name = 'collect_annotable_urls_spider'

    start_urls = configuration['start_urls']
    # allowed_domains = configuration['root_domains']

    rules = (
        Rule(LinkExtractor(), callback='parse_page_item'),
    )

    def __init__(self, *args, **kwargs):
        self.configuration = configuration
        self.website_analyzer = WebsiteAnalyzer(self.configuration['crawl_configs'])

        super(CollectAnnotableUrlsSpider, self).__init__(*args, **kwargs)

    def parse_page_item(self, response: scrapy.http.HtmlResponse):
        parsed_uri = urlparse(response.url)
        domain = parsed_uri.netloc

        matched_imgs_by_img_src = response.xpath("//img/@src").extract()

        self.logger.info(f'Hi, this is an item page!'
                         f'URL: {response.url}, '
                         f'Domain: {domain}. '
                         f'Images matched by img src present: {len(matched_imgs_by_img_src)}')

        analyze_results = self.website_analyzer.analyze_website(response.url, response)
        analyze_results['page_url'] = response.url
        analyze_results['domain'] = domain

        return analyze_results
