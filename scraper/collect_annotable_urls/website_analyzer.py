from typing import List
from scraper.collect_annotable_urls.crawl_configuration import CrawlConfiguration
import re
import scrapy
from scraper.collect_annotable_urls.items.page_analysis_results import PageAnalysisResults


class WebsiteAnalyzer(object):

    configs: List[CrawlConfiguration] = []

    def __init__(self, configs: List[CrawlConfiguration]):
        self.configs = configs

    def analyze_website(self, website_url: str, response: scrapy.http.response.html.HtmlResponse):
        gallery_page_matched_by_url = False
        page_with_gallery_matched_by_url = False
        can_be_page_with_gallery = True

        number_of_imgs_matched_by_a_href = 0
        number_of_images_by_img_src = 0

        img_src_values = response.xpath("//img/@src").extract()
        a_href_values = response.xpath("//a/@href").extract()

        some_config_matched = False
        for config in self.configs:
            if not re.match(f'.*{config.domain}.*', website_url):
                continue
            some_config_matched = True
            if config.pageWithGalleryUrlMatchesRegexp:
                rexp_match = re.match(config.pageWithGalleryUrlMatchesRegexp, website_url)
                page_with_gallery_matched_by_url = page_with_gallery_matched_by_url or rexp_match is not None

            if config.galleryUrlMatchesRegexp:
                rexp_match = re.match(config.galleryUrlMatchesRegexp, website_url)
                gallery_page_matched_by_url = gallery_page_matched_by_url or rexp_match is not None

            if config.pageWithGalleryUrlHasToMatchRegexp:
                rexp_match = re.match(config.pageWithGalleryUrlHasToMatchRegexp, website_url)
                can_be_page_with_gallery = rexp_match is not None

            if config.pageWithGalleryContainsImgSrcRegexp:
                for img_src_val in img_src_values:
                    if re.match(config.pageWithGalleryContainsImgSrcRegexp, img_src_val):
                        number_of_images_by_img_src += 1

            if config.pageWithGalleryContainsAnchorHrefRegexp:
                for a_href_val in a_href_values:
                    if re.match(config.pageWithGalleryContainsAnchorHrefRegexp, a_href_val):
                        number_of_imgs_matched_by_a_href += 1

        has_match_for_page_with_gallery_by_imgs = number_of_imgs_matched_by_a_href >= 1 \
            or number_of_images_by_img_src >= 1

        has_match_for_page_with_gallery_by_imgs = has_match_for_page_with_gallery_by_imgs \
            and can_be_page_with_gallery

        has_match = page_with_gallery_matched_by_url \
            or gallery_page_matched_by_url \
            or has_match_for_page_with_gallery_by_imgs

        page_analysis_results = PageAnalysisResults()

        page_analysis_results['number_of_images_by_a_href'] = number_of_imgs_matched_by_a_href
        page_analysis_results['number_of_images_by_img_src'] = number_of_images_by_img_src
        page_analysis_results['has_match'] = has_match
        page_analysis_results['url_matched_for_gallery_page'] = gallery_page_matched_by_url
        page_analysis_results['url_matched_for_page_with_gallery'] = page_with_gallery_matched_by_url

        return page_analysis_results
