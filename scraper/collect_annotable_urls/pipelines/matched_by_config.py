from scrapy.exceptions import DropItem
from scraper.collect_annotable_urls.spiders.main import CollectAnnotableUrlsSpider
from scraper.collect_annotable_urls.items.page_analysis_results import PageAnalysisResults


class MatchedByConfigPipeline(object):
    def process_item(self, item: PageAnalysisResults, _spider: CollectAnnotableUrlsSpider):
        if item['has_match']:
            return item
        else:
            raise DropItem(f'Dropping item not matched item. URL: ' + item['page_url'])
