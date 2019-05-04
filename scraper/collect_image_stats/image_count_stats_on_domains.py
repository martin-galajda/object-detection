import scrapy

from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.item import Item, Field
from scraper.get_domains_and_urls import get_domains_and_urls
from urllib.parse import urlparse
from collections import Counter
import json

image_count_per_domain_counter = Counter()
websites_visited_per_domain_counter = Counter()

OUT_FILE_PATH = '/storage/brno3-cerit/home/marneyko/object-detection/scraper/out.json'

class MyItem(scrapy.Item):
    imgUrl = Field()
    domain = Field()
domains_and_urls = get_domains_and_urls()

class MySpider(CrawlSpider):
    name = 'my_spider'
    start_urls = domains_and_urls['urls']
    allowed_domains = domains_and_urls['domains']

    rules = (
        Rule(LinkExtractor(unique=True), callback='parse_item'),
    )

    def parse_item(self, response: scrapy.http.HtmlResponse):
        parsed_uri = urlparse(response.url)
        domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
        matched_imgs = response.xpath("//img/@src").extract()

        self.logger.info(f'Hi, this is an item page!'
                         f'URL: {response.url}, '
                         f'Domain: {domain}. '
                         f'Count images: {len(matched_imgs)}')

        item = MyItem()
        # item['imgUrl'] = matched_imgs

        image_count_per_domain_counter[domain] += len(matched_imgs)
        websites_visited_per_domain_counter[domain] += 1

        with open(OUT_FILE_PATH, 'w') as outfile:
            data = {
                'image_count_per_domain_counter': image_count_per_domain_counter,
                'websites_visited_per_domain_counter': websites_visited_per_domain_counter,
            }
            json.dump(data, outfile, indent=2)

        return item
