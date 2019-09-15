import scrapy
from scrapy.item import Field


class PageAnalysisResults(scrapy.Item):
    page_url = Field()
    domain = Field()
    number_of_images_by_img_src = Field()
    number_of_images_by_a_href  = Field()
    has_match = Field()
    url_matched_for_gallery_page = Field()
    url_matched_for_page_with_gallery = Field()
    total_img_elements_found = Field()
